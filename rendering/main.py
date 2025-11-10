from OpenGL.GL import *
from glfw.GLFW import *
from glfw import _GLFWwindow as GLFWwindow
from PIL import Image
import glm
import numpy as np
import time

from shader import Shader
from shader_m import ShaderM
from camera import Camera, Camera_Movement
from dataManager import DataManager
from water import *

import platform, ctypes, os
from typing import List


print('Loading...')

# the relative path where the textures are located
IMAGE_RESOURCE_PATH = "textures"

# function that loads and automatically flips an image vertically
LOAD_IMAGE = lambda name: Image.open(os.path.join(IMAGE_RESOURCE_PATH, name))

# settings
SCR_WIDTH = 1280
SCR_HEIGHT = 720

# limit frame rate
fps = 30
spf = 1/fps

# camera
camera = Camera(glm.vec3(0.5, 0.05, 0.5))
lastX = SCR_WIDTH / 2.0
lastY = SCR_HEIGHT / 2.0
firstMouse = True
toggleMesh = False
keyPressStateM = False

# timing
deltaTime = 0.0
lastFrame = 0.0

# lighting
lightDir = glm.normalize(glm.vec3(0.5, -1.0, 0.5))

def main() -> int:
	global deltaTime, lastFrame

	# glfw: initialize and configure
	# ------------------------------
	glfwInit()
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)

	if (platform.system() == "Darwin"): # APPLE
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

	# glfw window creation
	# --------------------
	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Real-Time Terrain Generation and Rendering", None, None)
	if (window == None):

		print("Failed to create GLFW window")
		glfwTerminate()
		return -1

	glfwMakeContextCurrent(window)
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
	glfwSetCursorPosCallback(window, mouse_callback)
	glfwSetScrollCallback(window, scroll_callback)

	# tell GLFW to capture our mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED)

	# configure global opengl state
	# -----------------------------
	glEnable(GL_DEPTH_TEST)
	glEnable(GL_CULL_FACE)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 

	# build and compile shaders
	# -------------------------
	terrainShader = Shader("terrain.vert", "terrain.frag")
	skyboxShader = Shader("skybox.vert", "skybox.frag")
	waterShader = Shader("water.vert", "water.frag")
	treeShader = ShaderM("tree.vert", "tree.frag")
	# set up vertex data (and buffer(s)) and configure vertex attributes
	# ------------------------------------------------------------------
	patchSize = 128

	skyboxVertices = glm.array(glm.float32,
		# positions          
		-1.0,  1.0, -1.0,
		-1.0, -1.0, -1.0,
		 1.0, -1.0, -1.0,
		 1.0, -1.0, -1.0,
		 1.0,  1.0, -1.0,
		-1.0,  1.0, -1.0,

		-1.0, -1.0,  1.0,
		-1.0, -1.0, -1.0,
		-1.0,  1.0, -1.0,
		-1.0,  1.0, -1.0,
		-1.0,  1.0,  1.0,
		-1.0, -1.0,  1.0,

		 1.0, -1.0, -1.0,
		 1.0, -1.0,  1.0,
		 1.0,  1.0,  1.0,
		 1.0,  1.0,  1.0,
		 1.0,  1.0, -1.0,
		 1.0, -1.0, -1.0,

		-1.0, -1.0,  1.0,
		-1.0,  1.0,  1.0,
		 1.0,  1.0,  1.0,
		 1.0,  1.0,  1.0,
		 1.0, -1.0,  1.0,
		-1.0, -1.0,  1.0,

		-1.0,  1.0, -1.0,
		 1.0,  1.0, -1.0,
		 1.0,  1.0,  1.0,
		 1.0,  1.0,  1.0,
		-1.0,  1.0,  1.0,
		-1.0,  1.0, -1.0,

		-1.0, -1.0, -1.0,
		-1.0, -1.0,  1.0,
		 1.0, -1.0, -1.0,
		 1.0, -1.0, -1.0,
		-1.0, -1.0,  1.0,
		 1.0, -1.0,  1.0)

	
	# skybox VAO
	skyboxVAO = glGenVertexArrays(1)
	skyboxVBO = glGenBuffers(1)
	glBindVertexArray(skyboxVAO)
	glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO)
	glBufferData(GL_ARRAY_BUFFER, skyboxVertices.nbytes, skyboxVertices.ptr, GL_STATIC_DRAW)
	glEnableVertexAttribArray(0)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)

	# load textures
	# -------------

	faces = [
		"skybox/right.jpg",
		"skybox/left.jpg",
		"skybox/top.jpg",
		"skybox/bottom.jpg",
		"skybox/front.jpg",
		"skybox/back.jpg"
	]

	cubemapTexture = loadCubemap(faces)

	# terrain shader configuration
	# --------------------
	terrainShader.use()
	terrainShader.setVec3("light.direction", lightDir)
	
	lightColor = glm.vec3(1.0, 1.0, 1.0)
	# diffuseColor = lightColor * glm.vec3(0.7)
	diffuseColor = lightColor * glm.vec3(1.0)
	# ambientColor = lightColor * glm.vec3(0.2)
	ambientColor = lightColor * glm.vec3(1.0)
	terrainShader.setVec3("light.ambient", ambientColor)
	terrainShader.setVec3("light.diffuse", diffuseColor)
	terrainShader.setVec3("light.specular", 1.0, 1.0, 1.0)

	# material properties
	terrainShader.setVec3("material.ambient", 0.69, 0.66, 0.52)
	terrainShader.setVec3("material.diffuse", 0.69, 0.66, 0.52)
	terrainShader.setVec3("material.specular", 0.01, 0.01, 0.01) # specular lighting doesn't have full effect on this object's material
	terrainShader.setFloat("material.shininess", 0.2)

	# skybox shader configuration
	# --------------------
	skyboxShader.use()
	skyboxShader.setInt("skybox", 0)

	# tree shader configuration
	# --------------------
	treeShader.use()
	treeShader.setVec3("aLightDir", lightDir)
	treeShader.setInt("texture_diffuse1", 0)

	relevanceArea = 1

	# terrain data manager object
	dataManager = DataManager('textures/initial_dem.npy', terrainShader, treeShader, patchSize, relevanceArea)

	# water object
	water = Water(waterShader, 2, relevanceArea)

	print('Loaded!')
	
	glClearColor(1.0, 1.0, 1.0, 1.0)
	

	# render loop
	# -----------
	while not glfwWindowShouldClose(window):

		# per-frame time logic
		# --------------------
		currentFrame = glfwGetTime()
		deltaTime = currentFrame - lastFrame
		lastFrame = currentFrame
		time.sleep(max(spf - deltaTime, 0))

		# input
		# -----
		processInput(window)

		# render
		# ------
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		## draw terrain
		terrainShader.use()
		terrainShader.setVec3("viewPos", camera.Position)

		# view/projection transformations
		projection = glm.perspective(glm.radians(camera.Zoom), SCR_WIDTH / SCR_HEIGHT, camera.near, camera.far)
		view = camera.GetViewMatrix()
		terrainShader.setMat4("projection", projection)
		terrainShader.setMat4("view", view)

		treeShader.use()
		treeShader.setMat4("projection", projection)
		treeShader.setMat4("view", view)
		terrainShader.use()

		dataManager.draw(camera)

		## draw water
		waterShader.use()

		# view/projection transformations
		projection = glm.perspective(glm.radians(camera.Zoom), SCR_WIDTH / SCR_HEIGHT, camera.near, camera.far)
		view = camera.GetViewMatrix()
		waterShader.setMat4("projection", projection)
		waterShader.setMat4("view", view)

		water.draw(camera)

		## draw skybox as last
		glDepthFunc(GL_LEQUAL)  # change depth function so depth test passes when values are equal to depth buffer's content
		skyboxShader.use()
		view = glm.mat4(glm.mat3(camera.GetViewMatrix())) # remove translation from the view matrix
		skyboxShader.setMat4("view", view)
		skyboxShader.setMat4("projection", projection)
		# skybox cube
		glBindVertexArray(skyboxVAO)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture)
		glDrawArrays(GL_TRIANGLES, 0, 36)
		glBindVertexArray(0)
		glDepthFunc(GL_LESS) # set depth function back to default

		# glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		# -------------------------------------------------------------------------------
		glfwSwapBuffers(window)
		glfwPollEvents()

	# optional: de-allocate all resources once they've outlived their purpose:
	# ------------------------------------------------------------------------
	# glDeleteVertexArrays(1, (terrainVAO,))
	glDeleteVertexArrays(1, (skyboxVAO,))
	# glDeleteBuffers(1, (terrainVBO,))
	# glDeleteBuffers(1, (terrainEBO,))
	glDeleteBuffers(1, (skyboxVBO,))

	glfwTerminate()
	return 0

# process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
# ---------------------------------------------------------------------------------------------------------
def processInput(window: GLFWwindow) -> None:
	global deltaTime
	global toggleMesh
	global keyPressStateM

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS):
		glfwSetWindowShouldClose(window, True)

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS):
		camera.ProcessKeyboard(Camera_Movement.FORWARD, deltaTime)
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS):
		camera.ProcessKeyboard(Camera_Movement.BACKWARD, deltaTime)
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS):
		camera.ProcessKeyboard(Camera_Movement.LEFT, deltaTime)
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS):
		camera.ProcessKeyboard(Camera_Movement.RIGHT, deltaTime)
	if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS and keyPressStateM == False):
		if toggleMesh == False:
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
			toggleMesh = True
			keyPressStateM = True
		else:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
			toggleMesh = False
			keyPressStateM = True
	if (keyPressStateM == True and glfwGetKey(window, GLFW_KEY_M) == GLFW_RELEASE):
		keyPressStateM = False


# glfw: whenever the window size changed (by OS or user resize) this callback function executes
# ---------------------------------------------------------------------------------------------
def framebuffer_size_callback(window: GLFWwindow, width: int, height: int) -> None:
	global SCR_WIDTH
	global SCR_HEIGHT

	SCR_WIDTH = width
	SCR_HEIGHT = height
	glViewport(0, 0, width, height)

# glfw: whenever the mouse moves, this callback is called
# -------------------------------------------------------
def mouse_callback(window: GLFWwindow, xpos: float, ypos: float) -> None:
	global lastX, lastY, firstMouse

	if (firstMouse):

		lastX = xpos
		lastY = ypos
		firstMouse = False

	xoffset = xpos - lastX
	yoffset = lastY - ypos # reversed since y-coordinates go from bottom to top

	lastX = xpos
	lastY = ypos

	camera.ProcessMouseMovement(xoffset, yoffset)

# glfw: whenever the mouse scroll wheel scrolls, this callback is called
# ----------------------------------------------------------------------
def scroll_callback(window: GLFWwindow, xoffset: float, yoffset: float) -> None:

	camera.ProcessMouseScroll(yoffset)

# utility function for loading a 2D texture from file
# ---------------------------------------------------
def loadTexture(path : str) -> int:
	textureID = glGenTextures(1)

	try:
		img = LOAD_IMAGE(path)
		
		nrComponents = len(img.getbands())

		format = GL_RED if nrComponents == 1 else \
				 GL_RGB if nrComponents == 3 else \
				 GL_RGBA 

		glBindTexture(GL_TEXTURE_2D, textureID)
		glTexImage2D(GL_TEXTURE_2D, 0, format, img.width, img.height, 0, format, GL_UNSIGNED_BYTE, img.tobytes())
		glGenerateMipmap(GL_TEXTURE_2D)

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

		img.close()

	except:

		print("Texture failed to load at path: " + path)

	return textureID

# loads a cubemap texture from 6 individual texture faces
# order:
# +X (right)
# -X (left)
# +Y (top)
# -Y (bottom)
# +Z (front) 
# -Z (back)
# -------------------------------------------------------
def loadCubemap(faces : List[str]) -> int:
	textureID = glGenTextures(1)
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID)
	for i in range(len(faces)):
		try:
			img = LOAD_IMAGE(faces[i])
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
			img.close()
		except:
			print("Cubemap texture failed to load at path: " + faces[i])

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

	return textureID


main()

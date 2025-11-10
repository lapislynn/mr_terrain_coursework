from OpenGL.GL import *
from glfw.GLFW import *
from glfw import _GLFWwindow as GLFWwindow
from PIL import Image
import glm
import numpy as np
from utils import *
import assimp_py
import os
from model import Model


# Set the resource path for models
MODEL_RESOURCE_PATH = "models/tree/"


class TerrainPatch:
	'''
	Define a class for TerrainPatch
	'''
	def __init__(self, arr, shader, treeShader, x, z, patchSize, s=1.0):
		'''
		Initialize a TerrainPatch object with given parameters.
		'''
		self.arr = arr
		self.shader = shader
		self.treeShader = treeShader
		self.patchSize = patchSize
		self.x = x
		self.z = z
		
		self.s = s
		self.ne, self.nw, self.se, self.sw = None, None, None, None

		 # Calculate the normal vector for the terrain patch
		normal = self.calculateNormal(self.arr)
		
		self.scaleY = 0.16
		eps = 0.01*s
		terrainVertexRange = np.linspace(0-eps,self.s+eps,patchSize)
		terrainVertices = []

		self.treePatch = (s > 0.99)

		if self.treePatch:
			self.amount = 0
			modelMatrices = []

		# Generate terrain vertex and normal data
		for i in range(patchSize):
			for j in range(patchSize):
				# positions
				terrainVertices.append(terrainVertexRange[i])
				terrainVertices.append(self.arr[i][j] * self.scaleY)
				terrainVertices.append(terrainVertexRange[j])
				# normals
				terrainVertices.append(normal[i,j,0])
				terrainVertices.append(normal[i,j,1])
				terrainVertices.append(normal[i,j,2])

				if self.treePatch and np.random.random() > 0.9 and self.arr[i][j] * self.scaleY > 0.005 and normal[i,j,0] < 0.2 and normal[i,j,2] < 0.2:
					self.amount += 1
					model = glm.mat4(1.0)
					model = glm.translate(model, glm.vec3(x+terrainVertexRange[i], self.arr[i][j]*self.scaleY, z+terrainVertexRange[j]))
					model = glm.scale(model, glm.vec3(0.001,0.001,0.001))
					modelMatrices.append(model)
		
		if self.treePatch and self.amount==0:
			self.treePatch = False
		
		if self.treePatch:
			modelMatrices = glm.array(modelMatrices)

		terrainVertices = glm.array(glm.float32, *terrainVertices)

		self.NTerrainVertices = 0
		terrainIndices = []
		getTerrainIndex = lambda x,y: x*patchSize+y;
		for i in range(patchSize-1):
			for j in range(patchSize-1):
				# first triangle
				terrainIndices.append(getTerrainIndex(i,j))
				terrainIndices.append(getTerrainIndex(i,j+1))
				terrainIndices.append(getTerrainIndex(i+1,j))
				# second triangle
				terrainIndices.append(getTerrainIndex(i,j+1))
				terrainIndices.append(getTerrainIndex(i+1,j+1))
				terrainIndices.append(getTerrainIndex(i+1,j))
				self.NTerrainVertices += 6
		terrainIndices = glm.array(glm.uint32, *terrainIndices)


		# Create VAO, VBO, and EBO for terrain rendering
		self.terrainVAO = glGenVertexArrays(1)
		self.terrainVBO = glGenBuffers(1)
		self.terrainEBO = glGenBuffers(1)
	
		glBindVertexArray(self.terrainVAO)
	
		glBindBuffer(GL_ARRAY_BUFFER, self.terrainVBO)
		glBufferData(GL_ARRAY_BUFFER, terrainVertices.nbytes, terrainVertices.ptr, GL_STATIC_DRAW)
	
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.terrainEBO)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, terrainIndices.nbytes, terrainIndices.ptr, GL_STATIC_DRAW)

		# position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
		glEnableVertexAttribArray(0)
		# normal attribute
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
		glEnableVertexAttribArray(1)

		self.model = glm.mat4(1.0)
		self.offestY = 0.0
		self.model = glm.translate(self.model, glm.vec3(x, self.offestY, z))
		self.model = glm.scale(self.model, glm.vec3(1.0,1.0,1.0))

		self.normalMatrix = glm.mat3(glm.transpose(glm.inverse(self.model)))


		if self.treePatch:

			self.treeModel = Model(os.path.join(MODEL_RESOURCE_PATH, "lowpolypine.obj"))

			self.treeVAO = self.treeModel.meshes[0].VAO
			glBindVertexArray(self.treeVAO)
		
			# configure instanced array
			# -------------------------
			treebuffer = glGenBuffers(1)
			glBindBuffer(GL_ARRAY_BUFFER, treebuffer)
			glBufferData(GL_ARRAY_BUFFER, modelMatrices.nbytes, modelMatrices.ptr, GL_STATIC_DRAW)

			glEnableVertexAttribArray(3)
			glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, glm.sizeof(glm.mat4), None)
			glEnableVertexAttribArray(4)
			glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, glm.sizeof(glm.mat4), ctypes.c_void_p(glm.sizeof(glm.vec4)))
			glEnableVertexAttribArray(5)
			glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, glm.sizeof(glm.mat4), ctypes.c_void_p(2 * glm.sizeof(glm.vec4)))
			glEnableVertexAttribArray(6)
			glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, glm.sizeof(glm.mat4), ctypes.c_void_p(3 * glm.sizeof(glm.vec4)))

			glVertexAttribDivisor(3, 1)
			glVertexAttribDivisor(4, 1)
			glVertexAttribDivisor(5, 1)
			glVertexAttribDivisor(6, 1)


	def draw(self, camera, terrainEnhancementModels):
		'''
        Draw the terrain patch and tree instances using specified shaders.
		'''
		if not self.intersectFrustum(camera):
			return

		if self.treePatch:
			self.treeShader.use()
			glActiveTexture(GL_TEXTURE0)
			glBindTexture(GL_TEXTURE_2D, self.treeModel.textures_loaded[0].id)
			glBindVertexArray(self.treeModel.meshes[0].VAO)
			glDrawElementsInstanced(GL_TRIANGLES, len(self.treeModel.meshes[0].indices), GL_UNSIGNED_INT, None, self.amount)
			glBindVertexArray(0)
			self.shader.use()

		if not self.shouldDivide(camera):
			self.shader.setMat4("model", self.model)
			self.shader.setMat3("normalMatrix", self.normalMatrix)

			glBindVertexArray(self.terrainVAO)
			glDrawElements(GL_TRIANGLES, self.NTerrainVertices, GL_UNSIGNED_INT, None)
		else:
			if self.ne == None: # rest are also None
				if self.s < 0.25 + 1e-3:
					arr = terrainEnhancementModels[4].predict(self.arr)
				elif self.s < 0.5 + 1e-3:
					arr = terrainEnhancementModels[3].predict(self.arr)
				elif self.s < 0.5 + 1e-3:
					arr = terrainEnhancementModels[2].predict(self.arr)
				elif self.s < 1.00 + 1e-3:
					arr = terrainEnhancementModels[1].predict(self.arr)
				else:
					print('Error in terrain patch draw, should not be here!')

				self.ne = TerrainPatch(arr[:self.patchSize,self.patchSize:], self.shader, None, self.x         , self.z+self.s/2, self.patchSize, self.s/2.0)
				self.nw = TerrainPatch(arr[:self.patchSize,:self.patchSize], self.shader, None, self.x         , self.z         , self.patchSize, self.s/2.0)
				self.se = TerrainPatch(arr[self.patchSize:,self.patchSize:], self.shader, None, self.x+self.s/2, self.z+self.s/2, self.patchSize, self.s/2.0)
				self.sw = TerrainPatch(arr[self.patchSize:,:self.patchSize], self.shader, None, self.x+self.s/2, self.z         , self.patchSize, self.s/2.0)
			self.ne.draw(camera, terrainEnhancementModels)
			self.nw.draw(camera, terrainEnhancementModels)
			self.se.draw(camera, terrainEnhancementModels)
			self.sw.draw(camera, terrainEnhancementModels)


	def shouldDivide(self, camera):
		'''
		Function to determine if the current patch should be further divided
		'''
		if self.s < 0.125 + 1e-3:  # cannot divide further since not enhancement models
			return False 
		
		if self.arr.std() < 0.08:  # not much detail
			return False 

		patchPos = glm.vec3(self.x+self.s/2, self.offestY, self.z+self.s/2)
		dist = glm.length(camera.Position - patchPos)
		if dist > self.s / 1.5:
			return False 
		else:
			return True


	def intersectFrustum(self, camera):
		'''
		Function to check if the patch intersects with the camera frustum
		'''
		v1, v2, v3 = giveTriangleVertices(camera.Position.xz, camera.Front.xz, camera.Zoom, camera.far)
		if PointInTriangle(glm.vec2(self.x, self.z), v1, v2, v3):
			return True 
		if PointInTriangle(glm.vec2(self.x+self.s, self.z), v1, v2, v3):
			return True 
		if PointInTriangle(glm.vec2(self.x, self.z+self.s), v1, v2, v3):
			return True 
		if PointInTriangle(glm.vec2(self.x+self.s, self.z+self.s), v1, v2, v3):
			return True
		return False

	
	def loadTexture(self, arr):
		'''
		Function to load a texture from a NumPy array into OpenGL
		'''
		textureID = glGenTextures(1)

		img = Image.fromarray(arr)
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
	
		return textureID


	def calculateNormal(self, patch):
		'''
		Function to calculate the normal vector for a given patch using gradient information
		'''
		# Calculate gradient in x and y directions
		grad_x, grad_y = np.gradient(patch)
	
		# Calculate normal vector components
		normal_x = -grad_x
		normal_y = -grad_y
		normal_z = np.ones_like(patch) * 0.1 * self.s
	
		# Normalize the normal vector
		norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
		normal_x /= norm
		normal_y /= norm
		normal_z /= norm

		normal = np.dstack((normal_x, normal_z, normal_y))
		return normal


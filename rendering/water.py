import numpy as np
from OpenGL.GL import *
from glfw.GLFW import *
from glfw import _GLFWwindow as GLFWwindow
import glm
from math import floor, sin
import time

class Water:
	'''
	Define the Water class
	'''
	def __init__(self, shader, patchSize, relevanceArea):
		self.shader = shader
		self.patchSize = patchSize
		self.relevanceArea = relevanceArea
		
		waterVertexRange = np.linspace(0,1,patchSize)
		waterVertices = []
		for i in range(patchSize):
			for j in range(patchSize):
				# positions
				waterVertices.append(waterVertexRange[i])
				waterVertices.append(-0.02)
				waterVertices.append(waterVertexRange[j])
		waterVertices = glm.array(glm.float32, *waterVertices)

		self.NWaterVertices = 0
		waterIndices = []
		getWaterIndex = lambda x,y: x*patchSize+y;

		# Define indices for triangles forming the water surface
		for i in range(patchSize-1):
			for j in range(patchSize-1):
				# first triangle
				waterIndices.append(getWaterIndex(i,j))
				waterIndices.append(getWaterIndex(i,j+1))
				waterIndices.append(getWaterIndex(i+1,j))
				# second triangle
				waterIndices.append(getWaterIndex(i,j+1))
				waterIndices.append(getWaterIndex(i+1,j+1))
				waterIndices.append(getWaterIndex(i+1,j))
				self.NWaterVertices += 6
		waterIndices = glm.array(glm.uint32, *waterIndices)


		# Create VAO, VBO, and EBO for water rendering
		self.waterVAO = glGenVertexArrays(1)
		self.waterVBO = glGenBuffers(1)
		self.waterEBO = glGenBuffers(1)
	
		glBindVertexArray(self.waterVAO)
	
		glBindBuffer(GL_ARRAY_BUFFER, self.waterVBO)
		glBufferData(GL_ARRAY_BUFFER, waterVertices.nbytes, waterVertices.ptr, GL_STATIC_DRAW)
	
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.waterEBO)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, waterIndices.nbytes, waterIndices.ptr, GL_STATIC_DRAW)

		# position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)
		glEnableVertexAttribArray(0)



	def draw(self, camera):
		'''
		Method for rendering the water patch
		'''
		model = glm.mat4(1.0)
		model = glm.translate(model, glm.vec3(-self.relevanceArea, 0, -self.relevanceArea))
		model = glm.translate(model, glm.vec3(floor(camera.Position.x), 0, floor(camera.Position.z)))
		model = glm.scale(model, glm.vec3(2*self.relevanceArea+1, 1.0, 2*self.relevanceArea+1))

		self.shader.setFloat("time", sin(time.time()/5))
		self.shader.setMat4("model", model)

		glBindVertexArray(self.waterVAO)
		glDrawElements(GL_TRIANGLES, self.NWaterVertices, GL_UNSIGNED_INT, None)

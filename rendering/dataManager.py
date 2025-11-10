import numpy as np
import glm
from math import floor
from terrainPatch import *
from terrainCompletionModel import *
from terrainEnhancementModel import *


class DataManager:
	'''
	A class to manage terrain data, patches, completion, and rendering.
	'''

	def __init__(self, initialDemPath, terrainShader, treeShader, patchSize, relevanceArea):
		self.terrainShader = terrainShader
		self.treeShader = treeShader
		self.patchSize = patchSize

		self.relevanceArea = relevanceArea
		self.maxPatches1D = 1024  # equivalent to ~ 2k km x 2k km
		self.patchExist = np.zeros((self.maxPatches1D,self.maxPatches1D), dtype=np.uint8)
		self.terrainPatchPtr = [[None for j in range(self.maxPatches1D)] for i in range(self.maxPatches1D)]

		# Initialize the initial terrain patch and completion model
		self.terrainPatchPtr[0][0] = TerrainPatch(np.rot90(np.load(initialDemPath),3), self.terrainShader, self.treeShader, 0.0, 0.0, self.patchSize)
		self.patchExist[0][0] = 1

		print('Initializing terrain completion model...')
		self.terrainCompletionModel = TerrainCompletionModel(gt_size = self.patchSize)
		print('Terrain completion model initialized!')

		# Complete initial patches
		print('Calculating initial patches...')
		for i in range(1,self.relevanceArea+1):
			self.completePatch(-i,-i+1)
			self.completePatches(0,0,i)
		print('Initial patches calculated!')

		# Set up enhancement models
		print('Initializing terrain enhancement models...')
		self.terrainEnhancementModels = [TerrainEnhancementModel('weights/enhancement_%d.pt'%(i)) for i in range(5)]
		print('Terrain enhancement models initialized!')


	def completePatches(self, x, y, relevanceArea):
		'''
		Completes a group of patches around a given point (x, y)
		'''
		x = floor(x)
		y = floor(y)

		for i in range(-relevanceArea, relevanceArea+1):
			for j in range(-relevanceArea, relevanceArea+1):
				if self.patchExist[x+i][y+j] == 0:
					self.completePatch(x+i,y+j)


	def completePatch(self, x, y):
		'''
		Completes a single patch at the specified coordinates (x, y)
		'''
		if self.patchExist[x,y] == 1:
			print('Warning: Patch already exists, check completePatch in dataManager!')
			return

		x = floor(x)
		y = floor(y)

		ip = [None for i in range(9)]
		allNone = True

		for i in range(3):
			for j in range(3):
				if i == 1 and j == 1:
					pass
				elif self.patchExist[x+i-1][y+j-1] == 1:
					allNone = False
					ip[i*3 + j] = self.terrainPatchPtr[x+i-1][y+j-1].arr
		if allNone:
			print('Warning: All None for completion, check completePatch in dataManager!', x, y)

		pr = self.terrainCompletionModel.predict(ip)

		self.terrainPatchPtr[x][y] = TerrainPatch(pr, self.terrainShader, self.treeShader, float(x), float(y), self.patchSize)
		self.patchExist[x,y] = 1


	def getPatchExist(self, x, y):
		'''
		Checks if a patch exists at the specified coordinates (x, y)
		'''
		assert abs(x) < self.maxPatches1D//2 and abs(y) < self.maxPatches1D//2, 'Error in DataManager, increase maxPatches1D, limit reached!'

		if self.patchExist[floor(x), floor(y)] == 1:
			return True
		else:
			return False


	def setPatchExist(self, x, y):
		'''
		Marks a patch as existing at the specified coordinates (x, y)
		'''
		self.patchExist[floor(x), floor(y)] = 1


	def draw(self, camera):
		'''
		Draws the terrain using the given camera and enhancement models
		'''
		x = floor(camera.Position[0])
		z = floor(camera.Position[2])
		self.completePatches(x, z, self.relevanceArea)

		for i in range(-self.relevanceArea, self.relevanceArea+1):
			for j in range(-self.relevanceArea, self.relevanceArea+1):
				self.terrainPatchPtr[x+i][z+j].draw(camera, self.terrainEnhancementModels)

import numpy as np
from Utils import *
import cv2
# from dask.array.routines import gradient
import matplotlib.pyplot as plt

SMALL_FLOAT = 0.00001
BIG_NUMBER = 999999999
SIN45 = 0.70710678118  #  in degrees


class SecondLevel:
	gradients = []

	def setImage(self, image):
		image = image
		
	def computeGradients(self, imageGray):
		
		height, width = imageGray.shape  
		
		# --, |, \, /
		self.gradients = np.zeros((height, width, 4), np.float64)  
		
		self.gradients[:, :, 1] = np.gradient(imageGray, axis=1)  # horizontal
		self.gradients[:, :, 0] = np.gradient(imageGray, axis=0)  # vertical
		
		for i in range(0, width):  # \
			diagonalTmp = np.diagonal(imageGray, i)
			if len(diagonalTmp) > 1:
				gradientSuperior = np.gradient(diagonalTmp)	
				for j in range(0, len(gradientSuperior)):
					self.gradients[j, i + j, 2] = gradientSuperior[j]
				
		for i in range(1, height):
			diagonalTmp = np.diagonal(imageGray, -i)
			if len(diagonalTmp) > 1:
				gradientInferior = np.gradient(diagonalTmp)
				for j in range(0, len(gradientInferior)):
					self.gradients[i + j, j, 2] = gradientInferior[j]
			
		gradientDiag2Tmp = np.zeros((width, height), np.float64)  # /
		imageGrayRotated = np.rot90(imageGray, 1)
		height, width = imageGrayRotated.shape 
		
		for i in range(0, width):
			diagonalTmp = np.diagonal(imageGrayRotated, i)
			if len(diagonalTmp) > 1:
				gradientSuperior = np.gradient(diagonalTmp)	
				for j in range(0, len(gradientSuperior)):
					gradientDiag2Tmp[j, i + j] = gradientSuperior[j]
				
		for i in range(1, height):
			diagonalTmp = np.diagonal(imageGrayRotated, -i)
			if len(diagonalTmp) > 1:
				gradientInferior = np.gradient(diagonalTmp)
				for j in range(0, len(gradientInferior)):
					gradientDiag2Tmp[i + j, j] = gradientInferior[j]
							
		self.gradients[:, :, 3] = np.rot90(gradientDiag2Tmp, -1)
		
		print 'Computing gradients [OK]' 
		
		# plt.imshow(self.gradients[:, :, 3], cmap='jet')
		# plt.show()

	'''
	@gradients is a [h,w,4]-dimention matrix
		
	'''	

	def ERSTransform(self, imageGray, radius=1):
		
		#imageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
				
		self.computeGradients(imageGray)
		
		height, width = imageGray.shape  
		
		# epsilon vectors
		gradientMax = np.zeros((height, width, 8), np.float64)
		gradientMin = np.zeros((height, width, 8), np.float64)
		
		# r vectors
		positionsMax = np.zeros((height, width, 8), np.int32)
		positionsMin = np.zeros((height, width, 8), np.int32)
		
		for h in range(0 + radius, height - radius):
			for w in range(0 + radius, width - radius): 
				
				neighborhood = np.array(self.gradients[h - radius:h + radius, w - radius:w + radius, :], dtype=float)
				
				# eight directions (x,y) in Cartesian coordinates
				#
				# x = 0
				# y = 1
				# \ = 2
				# / = 3
				
				# direction 1 ->
				gradient = neighborhood[radius, radius:radius * 2, 0]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 0] = maxValue
				gradientMin[h, w, 0] = minValue
				positionsMax[h, w, 0] = list(gradient).index(maxValue)
				positionsMin[h, w, 0] = list(gradient).index(minValue)
				
				# direction 5 <-
				gradient = neighborhood[radius, 0:radius, 0]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 4] = maxValue
				gradientMin[h, w, 4] = minValue
				positionsMax[h, w, 4] = list(gradient).index(maxValue)
				positionsMin[h, w, 4] = list(gradient).index(minValue)
				
				# direction ^
				gradient = neighborhood[0:radius, radius, 1]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 6] = maxValue
				gradientMin[h, w, 6] = minValue
				positionsMax[h, w, 6] = list(gradient).index(maxValue)
				positionsMin[h, w, 6] = list(gradient).index(minValue)
				
				# direction 3 v
				gradient = neighborhood[radius:radius * 2, radius, 1]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 2] = maxValue
				gradientMin[h, w, 2] = minValue
				positionsMax[h, w, 2] = list(gradient).index(maxValue)
				positionsMin[h, w, 2] = list(gradient).index(minValue)
								
				# direction 6 ^\
				gradient = neighborhood[:, :, 2].diagonal()[0:radius]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 5] = maxValue
				gradientMin[h, w, 5] = minValue
				positionsMax[h, w, 5] = list(gradient).index(maxValue)
				positionsMin[h, w, 5] = list(gradient).index(minValue)
				
				# direction 2 \v
				gradient = neighborhood[:, :, 2].diagonal()[radius:radius * 2]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 1] = maxValue
				gradientMin[h, w, 1] = minValue
				positionsMax[h, w, 1] = list(gradient).index(maxValue)
				positionsMin[h, w, 1] = list(gradient).index(minValue)
				
				# direction 8 /^
				gradient = np.flip(neighborhood, 0)[:, :, 3].diagonal()[0:radius]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 7] = maxValue
				gradientMin[h, w, 7] = minValue
				positionsMax[h, w, 7] = list(gradient).index(maxValue)
				positionsMin[h, w, 7] = list(gradient).index(minValue)
				
				# direction 4 v/
				gradient = np.flip(neighborhood, 0)[:, :, 3].diagonal()[radius:radius * 2]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 3] = maxValue
				gradientMin[h, w, 3] = minValue
				positionsMax[h, w, 3] = list(gradient).index(maxValue)
				positionsMin[h, w, 3] = list(gradient).index(minValue)
				
		print 'Computing max and min points [OK]' 
				
		# self.quantities(gradientMax, gradientMin, positionsMax, positionsMin, radius)
		
		# my quantities
		self.myQuantities(positionsMax, positionsMin, gradientMax, gradientMin)
		
	def myQuantities(self, positionsMax, positionsMin, gradientMax, gradientMin):
			
		height, width = positionsMax[:, :, 0].shape
		
		areasMax = np.zeros(8, np.float32)
		areasMin = np.zeros(8, np.float32)
		areas = np.zeros((height, width), np.float32)
		strength = np.zeros((height, width), np.float32)
		rank = np.zeros((height, width), np.float32)
		
		minGradientsAbs = np.absolute(gradientMin)
		maxGradientsAbs = np.absolute(gradientMax)
		
		for h in range(0, height):
			for w in range(0, width):
				# area
				for i in range(0, 8):
					nextPoint = i + 1 if i < 7 else 0
					areasMax[i] = (positionsMax[h, w, i] * positionsMax[h, w, nextPoint] * SIN45) / 2.0
					areasMin[i] = (positionsMin[h, w, i] * positionsMin[h, w, nextPoint] * SIN45) / 2.0
				
				areasMeanTmp = np.mean(areasMax)
				areasMaxTmp = np.std(areasMax) / areasMeanTmp if areasMeanTmp > 0.0 else SMALL_FLOAT
				areasMeanTmp = np.mean(areasMin)   
				areasMinTmp = np.std(areasMin) / areasMeanTmp if areasMeanTmp > 0.0 else SMALL_FLOAT
				areas[h, w] = areasMaxTmp + areasMinTmp  # small values points indide 
					
					
				# strength
				strength[h, w] = np.mean(maxGradientsAbs[h, w, :]) + np.mean(minGradientsAbs[h, w, :]) 
				#rank[h,w] =  areas[h,w]* strength[h,w] 
				rank[h,w] =  areas[h,w]+strength[h,w] if strength[h,w] > 0.0 else 0.0 
		
		
		toPlot = rank
		#toPlot = np.ma.masked_where(toPlot < 50, toPlot)
		cmap = plt.get_cmap('jet')
		#cmap.set_bad('white')
		plt.imshow(toPlot, cmap=cmap)
		plt.show()

	def quantities(self, maxGradients, minGradients, positionsMax, positionsMin, radius):
		height, width = maxGradients[:, :, 0].shape  
		# S quantity, see paper
		symmetry = np.zeros((height, width), np.float64) 
		strength = np.zeros((height, width), np.float64)
		uniformity = np.zeros((height, width), np.float64)
		
		# to avoid reiterative computations, strenght function
		minGradientsAbs = np.absolute(minGradients)
		maxGradientsAbs = np.absolute(maxGradients)
				
		for h in range(0 + radius, height - radius):
			for w in range(0 + radius, width - radius):
				
				# symmetry, unit test [ok]
				deltasMax = [self.delta(h, w, maxGradients, 0), self.delta(h, w, maxGradients, 1), self.delta(h, w, maxGradients, 2), self.delta(h, w, maxGradients, 3)]
				deltasMin = [self.delta(h, w, minGradients, 0), self.delta(h, w, minGradients, 1), self.delta(h, w, minGradients, 2), self.delta(h, w, minGradients, 3)]
				symmetry[h, w] = np.mean(deltasMax) + np.mean(deltasMin)
						
				# strength
				# sumMeansTmp = (self.mean(h, w, maxGradientsAbs) + self.mean(h, w, minGradientsAbs));
				sumMeansTmp = np.mean(maxGradientsAbs[h, w, :]) + np.mean(minGradientsAbs[h, w, :]) 
				strength[h, w] = 1.0 / (sumMeansTmp if  sumMeansTmp != 0.0 else SMALL_FLOAT)  # to avoid division by zero 
				
				# print maxGradients[h,w,:], minGradients[h,w,:], strength[h, w]
				
				# uniformity [ok]
				sigmaMaxTmp = self.averageDeviation(positionsMax[h, w, :])
				sigmaMinTmp = self.averageDeviation(positionsMin[h, w, :])
				meanTmpMax = np.mean(positionsMax[h, w, :])
				meanTmpMin = np.mean(positionsMin[h, w, :])
				
				# meanTmpMax = self.mean(h, w, positionsMax)
				# meanTmpMin = self.mean(h, w, positionsMin)
				
				# print sigmaMaxTmp
				
				# avoid division by zero
				meanTmpMax = meanTmpMax if meanTmpMax != 0.0 else SMALL_FLOAT
				meanTmpMin = meanTmpMin if meanTmpMin != 0.0 else SMALL_FLOAT
				
				uniformity[h, w] = (sigmaMaxTmp / meanTmpMax) + (sigmaMinTmp / meanTmpMin) 
				
				# test[h, w] = uniformity[h, w] + strength[h, w] + symmetry[h, w]
		
		'''imageTmp = np.zeros((height, width), np.float64)+255.0
		for i in range(0,20):
			imageTmp[np.where(test == np.max(test))] = 0.0
			test[np.where(test == np.max(test))] = np.min(test)
		'''	
		uniformity = minMax(uniformity, np.min(uniformity), np.max(uniformity), 0.0, 1.0)
		strength = minMax(strength, np.min(strength), np.max(strength), 0.0, 1.0)
		symmetry = minMax(symmetry, np.min(symmetry), np.max(symmetry), 0.0, 1.0)	
		
		test = uniformity + strength + symmetry
		
		# cv2.imshow("gradient 1", test)
		# cv2.waitKey(0)  
		
		# test = np.ma.masked_where(test < 0.0, test)
		cmap = plt.get_cmap('jet')
		# cmap.set_bad('white')
		plt.imshow(test, cmap=cmap)
		plt.show()
		
		# cv2.imshow("gradient 1", test)
		# cv2.waitKey(0)
			
	'''
	using the paper notation
	@matrix is a [h,w,8]-dimensional matrix 
	'''

	def delta(self, h, w, matrix, k):
		minTmp = min(matrix[h, w, k], matrix[h, w, k + 4])
		return (max(matrix[h, w, k], matrix[h, w, k + 4]) / (minTmp if minTmp != 0.0 else SMALL_FLOAT)) - 1.0  # to avoid division by zero
	
	'''
	Discretize version of equation (4) -> sigma function
	
	'''

	def averageDeviation(self, values):
		
		meanTmp = np.mean(values)
		return np.mean(np.abs(values - meanTmp))


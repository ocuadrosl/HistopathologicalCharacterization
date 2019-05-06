import numpy as np
from Utils import *
import cv2
# from dask.array.routines import gradient
import matplotlib.pyplot as plt
from dask.array.routines import gradient

SMALL_FLOAT = 0.00001
BIG_NUMBER = 999999999
SIN45 = 0.70710678118  #  in degrees


class SecondLevel:

	def setImage(self, image):
		image = image
				
	'''
	computing gradients for the entire image
	'''	

	def computeGradients(self, imageGray):
		
		height, width = imageGray.shape  
		
		# --, |, \, /
		gradients = np.zeros((height, width, 4), np.float32)  
		
		gradients[:, :, 1] = np.gradient(imageGray, axis=1)  # horizontal
		gradients[:, :, 0] = np.gradient(imageGray, axis=0)  # vertical
		
		for i in range(0, width):  # \
			diagonalTmp = np.diagonal(imageGray, i)
			if len(diagonalTmp) > 1:
				gradientSuperior = np.gradient(diagonalTmp)	
				for j in range(0, len(gradientSuperior)):
					gradients[j, i + j, 2] = gradientSuperior[j]
				
		for i in range(1, height):
			diagonalTmp = np.diagonal(imageGray, -i)
			if len(diagonalTmp) > 1:
				gradientInferior = np.gradient(diagonalTmp)
				for j in range(0, len(gradientInferior)):
					gradients[i + j, j, 2] = gradientInferior[j]
			
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
							
		gradients[:, :, 3] = np.rot90(gradientDiag2Tmp, -1)
			
		# plot results
		'''
		axeslist = plt.subplots(ncols=2, nrows=2)[1]
		axeslist.ravel()[0].imshow(gradients[:, :, 0], cmap=plt.jet())
		axeslist.ravel()[1].imshow(gradients[:, :, 1], cmap=plt.jet())
		axeslist.ravel()[2].imshow(gradients[:, :, 2], cmap=plt.jet())
		axeslist.ravel()[3].imshow(gradients[:, :, 3], cmap=plt.jet())
		plt.tight_layout()
		'''
		#plt.imshow(gradients[:, :, 0] + gradients[:, :, 1] + gradients[:, :, 2] + gradients[:, :, 3], cmap=plt.jet())
		#plt.show()
		
		print 'Computing gradients [OK]'
		return gradients[:, :, 0] + gradients[:, :, 1] + gradients[:, :, 2] + gradients[:, :, 3]

	'''
	@gradients is a [h,w,4]-dimention matrix
		
	'''	
	
	def findMaxMinGradients(self, gradients, radius):
		height, width = gradients.shape  
		
		# epsilon vectors
		gradientMax = np.zeros((height, width, 8), np.float32)
		gradientMin = np.zeros((height, width, 8), np.float32)
		
		# r vectors
		positionsMax = np.zeros((height, width, 8), np.int32)
		positionsMin = np.zeros((height, width, 8), np.int32)
		
		for h in range(0 + radius, height - radius):
			for w in range(0 + radius, width - radius): 
				
				neighborhood = np.array(gradients[h - radius:h + radius, w - radius:w + radius], dtype=float)
					
				# eight directions (x,y) in Cartesian coordinates
				#
				# x = 0
				# y = 1
				# \ = 2
				# / = 3
				
				# direction 1 ->
				gradient = neighborhood[radius, radius:radius * 2]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 0] = maxValue
				gradientMin[h, w, 0] = minValue
				positionsMax[h, w, 0] = list(gradient).index(maxValue)
				positionsMin[h, w, 0] = list(gradient).index(minValue)
				
				# direction 5 <-
				gradient = neighborhood[radius, 0:radius]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 4] = maxValue
				gradientMin[h, w, 4] = minValue
				positionsMax[h, w, 4] = list(gradient).index(maxValue)
				positionsMin[h, w, 4] = list(gradient).index(minValue)
				
				# direction ^
				gradient = neighborhood[0:radius, radius]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 6] = maxValue
				gradientMin[h, w, 6] = minValue
				positionsMax[h, w, 6] = list(gradient).index(maxValue)
				positionsMin[h, w, 6] = list(gradient).index(minValue)
				
				# direction 3 v
				gradient = neighborhood[radius:radius * 2, radius]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 2] = maxValue
				gradientMin[h, w, 2] = minValue
				positionsMax[h, w, 2] = list(gradient).index(maxValue)
				positionsMin[h, w, 2] = list(gradient).index(minValue)
								
				# direction 6 ^\
				gradient = neighborhood[:, :].diagonal()[0:radius]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 5] = maxValue
				gradientMin[h, w, 5] = minValue
				positionsMax[h, w, 5] = list(gradient).index(maxValue)
				positionsMin[h, w, 5] = list(gradient).index(minValue)
				
				# direction 2 \v
				gradient = neighborhood[:, :].diagonal()[radius:radius * 2]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 1] = maxValue
				gradientMin[h, w, 1] = minValue
				positionsMax[h, w, 1] = list(gradient).index(maxValue)
				positionsMin[h, w, 1] = list(gradient).index(minValue)
				
				# direction 8 /^
				gradient = np.flip(neighborhood, 0)[:, :].diagonal()[0:radius]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 7] = maxValue
				gradientMin[h, w, 7] = minValue
				positionsMax[h, w, 7] = list(gradient).index(maxValue)
				positionsMin[h, w, 7] = list(gradient).index(minValue)
				
				# direction 4 v/
				gradient = np.flip(neighborhood, 0)[:, :].diagonal()[radius:radius * 2]
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 3] = maxValue
				gradientMin[h, w, 3] = minValue
				positionsMax[h, w, 3] = list(gradient).index(maxValue)
				positionsMin[h, w, 3] = list(gradient).index(minValue)
				
		print 'Computing max and min points [OK]' 
		
		cmap = plt.get_cmap('jet')
		axeslist = plt.subplots(ncols=4, nrows=2)[1]
		axeslist.ravel()[0].imshow(gradientMax[:, :, 0], cmap=cmap)
		axeslist.ravel()[1].imshow(gradientMax[:, :, 1], cmap=cmap)
		axeslist.ravel()[2].imshow(gradientMax[:, :, 2], cmap=cmap)
		axeslist.ravel()[3].imshow(gradientMax[:, :, 3], cmap=cmap)
		axeslist.ravel()[4].imshow(gradientMax[:, :, 4], cmap=cmap)
		axeslist.ravel()[5].imshow(gradientMax[:, :, 5], cmap=cmap)
		axeslist.ravel()[6].imshow(gradientMax[:, :, 6], cmap=cmap)
		axeslist.ravel()[7].imshow(gradientMax[:, :, 7], cmap=cmap)
		plt.tight_layout()
		plt.show()

	def ERSTransform(self, image, radius=1):
					
		gradients = self.computeGradients(image)
						
		# my quantities
		# self.myQuantities(positionsMax, positionsMin, gradientMax, gradientMin)
		self.histograms(gradients, radius)
	
	def histograms(self, gradients, radius):
		
		h = 504
		w = 412
		
		neighborhood = np.array(gradients[h - radius:h + radius, w - radius:w + radius], dtype=float)
		plt.imshow(neighborhood, cmap = 'jet')
		plt.show()
				
		plt.hist(neighborhood.ravel(),100); 
		plt.show()
		
		
		height, width = neighborhood.shape
		result = np.zeros((height, width), np.float32)
		
		for h in range(0, height/2):
			for w in range(0, width/2):
				nW = neighborhood[h,w]
				#nE = neighborhood[h , width-w-1]
				#sW = neighborhood[height-h-1, w]
				sE = neighborhood[height-h-1, width-w-1]
				
				result[h, w] = np.abs(nW - sE)
				result[height-h-1, width-w-1] = result[h, w]
		
		
		for h in range(0, height/2):
			for w in range(width/2, width):
				nE = neighborhood[h,w]
				#nE = neighborhood[h , width-w-1]
				#sW = neighborhood[height-h-1, w]
				sW = neighborhood[height-h-1, width-w-1]
				
				result[h, w] = np.abs(nE - sW) 
				result[height-h-1, width-w-1] = result[h, w]
				
				
				
		plt.imshow(result, cmap = 'jet')
		plt.show()		 
		 	
		 
		 
		
		
		
		
		return
		height, width = gradients.shape 
				
		gradientMean = np.zeros((height, width), np.float32)
		gradientStd = np.zeros((height, width), np.float32)
		
		# r vectors
		positionsMean = np.zeros((height, width), np.int32)
		positionsStd = np.zeros((height, width), np.int32)
		
		for h in range(0 + radius, height - radius):
			for w in range(0 + radius, width - radius): 
				
				neighborhood = np.array(gradients[h - radius:h + radius, w - radius:w + radius], np.float32)
				#hist = np.histogram(neighborhood, bins=255)[0]
				gradientMean[h, w] = np.mean(neighborhood)
				# print np.mean(hist)
				
		plt.imshow(gradientMean, cmap='jet')
		plt.show()	
	
	def myQuantities(self, positionsMax, positionsMin, gradientMax, gradientMin):
			
		height, width = positionsMax[:, :, 0].shape
		
		areasMax = np.zeros(8, np.float32)
		areasMin = np.zeros(8, np.float32)
		areas = np.zeros((height, width), np.float32)
		strength = np.zeros((height, width), np.float32)
		uniformity = np.zeros((height, width), np.float32)
		rank = np.zeros((height, width), np.float32)
		
		minGradientsAbs = np.absolute(gradientMin)
		maxGradientsAbs = np.absolute(gradientMax)
		
		for h in range(0, height):
			for w in range(0, width):
				# area
				'''
				for i in range(0, 8):
					nextPoint = i + 1 if i < 7 else 0
					areasMax[i] = (positionsMax[h, w, i] * positionsMax[h, w, nextPoint] * SIN45) / 2.0
					areasMin[i] = (positionsMin[h, w, i] * positionsMin[h, w, nextPoint] * SIN45) / 2.0
				'''
				
				uniformity[h, w] = self.averageDeviation(positionsMax[h, w, :])			
				
				'''
				areasMeanTmp = np.mean(areasMax)
				areasMaxTmp = np.std(areasMax) / areasMeanTmp if areasMeanTmp > 0.0 else SMALL_FLOAT
				areasMeanTmp = np.mean(areasMin)   
				areasMinTmp = np.std(areasMin) / areasMeanTmp if areasMeanTmp > 0.0 else SMALL_FLOAT
				areas[h, w] = areasMaxTmp + areasMinTmp  # small values points indide 
				'''
					
				# strength
				strength[h, w] = np.mean(maxGradientsAbs[h, w, :]) + np.mean(minGradientsAbs[h, w, :]) 
				# rank[h,w] =  areas[h,w]* strength[h,w] 
				rank[h, w] = areas[h, w] + strength[h, w] if strength[h, w] > 0.0 else 0.0 
		
		toPlot = uniformity
		# toPlot = np.ma.masked_where(toPlot > 1, toPlot)
		cmap = plt.get_cmap('jet')
		# cmap.set_bad('white')
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


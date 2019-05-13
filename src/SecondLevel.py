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
		plt.imshow(gradients[:, :, 0] + gradients[:, :, 1] + gradients[:, :, 2] + gradients[:, :, 3], cmap=plt.jet())
		plt.show()
		
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

	def ERSTransform(self, image, radiusMin=1, radiusMax=1):
					
		 	# gradients = self.computeGradients(image)
						
		# my quantities
		self.myQuantities(image, radiusMin, radiusMax)
		# self.histograms(gradients, radius)
	
	def histograms(self, gradients, radius):
		
		'''
		h = 504
		w = 412
		
		neighborhood = np.array(gradients[h - radius:h + radius, w - radius:w + radius], dtype=float)
		plt.imshow(neighborhood, cmap = 'jet')
		plt.show()
				
		plt.hist(neighborhood.ravel(),100); 
		plt.show()
		
		
		height, width = neighborhood.shape
		result = np.zeros((height, width), np.float32)
		'''
		
		height, width = gradients.shape
		result = np.zeros((height, width), np.float32)
		for h in range(0, height - radius):
			for w in range(0, width - radius):
				
				index = gradients[h, w]
				
				neighborhoodWidth = w + radius
				neighborhoodHeight = h + radius
				
				# nW index
				for nH in range(h - radius, h):
					for nW in range(w - radius, w):
					
						nE = gradients[h, neighborhoodWidth - nW]
						sW = gradients[neighborhoodHeight - nH, w]
						sE = gradients[neighborhoodHeight - nH, neighborhoodWidth - nW]
						
						result[h, w] = np.mean((np.abs(index - nE) + np.abs(index - sW) + np.abs(index - sE)))  		
						
		plt.imshow(result, cmap='jet')
		plt.show()		 
		
		return
	
	def myQuantities(self, edges, radiusMin, radiusMax):
			
		height, width = edges.shape
		
		rank = np.zeros((height, width, radiusMax - radiusMin, radiusMax / 2), np.int16)
		
		x = -width
		y = height	
		t1 = []
		t2 = []	
		#possible improvements
		# evaluate only non edge pixels
		# 
		for h in range(radiusMax, height - radiusMax):
			x = -width
			for w in range(radiusMax, width - radiusMax):
				
				for R in range(radiusMin, radiusMax): #possible radius
					r = R / 2
					
					for d in range(0, r): # distance from the interior circle
						for g in np.linspace(0.0, (3/2)*np.pi, 4): # 7 possible orientation
							for t in np.linspace(0.0, 2.0 * np.pi, r): # R angle of rolling circle
							
								# a = h - int((r) * np.cos(t) + d * np.cos(t))
								# b = w - int((r) * np.sin(t) - d * np.sin(t))
								#a = (h - (np.cos(t) * (r + d)))
								#b = (w - (np.sin(t) * (r - d)))
								
								a = r*np.cos(g-t) + d*np.cos(t)
								b = r*np.sin(t-g) - d*np.sin(t)
								
								a = int(np.around(a))
								b = int(np.around(b))
								
								#t1.append(a)
								#t2.append(b)
								
								if edges[a, b] == 255:
									rank[a, b, R-radiusMin, d] = rank[a, b, R-radiusMin, d] + 1
						
							#print t1;
							#print t2;
							#print ""
							
							#plt.scatter(t1,t2)
							#plt.axis('equal')
							#plt.show()
							#t1 = []
							#t2 = []
										
				x = x + 1	
			y = y + 1	
				
		# rank = np.ma.masked_where(rank > 1000000, rank)
		cmap = plt.get_cmap('jet')
		# cmap.set_bad('white')
		plt.imshow(rank[:,:,0, 1], cmap=cmap)
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


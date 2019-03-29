import numpy as np
from Utils import *
import cv2
from dask.array.routines import gradient
import matplotlib.pyplot as plt

SMALL_FLOAT = 0.0000000001


class SecondLevel:
	image = []

	def setImage(self, image):
		image = image

	def ERSTransform(self, inputImage, radius=1):
		imageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
		height, width = imageGray.shape  
		
		# epsilon vectors
		gradientMax = np.zeros((height, width, 8), np.float64)
		gradientMin = np.zeros((height, width, 8), np.float64)
		
		# r vectors
		positionsMax = np.zeros((height, width, 8), np.int32)
		positionsMin = np.zeros((height, width, 8), np.int32)
		
		for h in range(0 + radius, height - radius):
			for w in range(0 + radius, width - radius): 
				
				neighborhood = np.array(imageGray[h - radius:h + radius, w - radius:w + radius], dtype=float)
				
				# eight directions (x,y) in cartesian coordinates
				
				# direction 1 (1,0)
				gradient = np.gradient(neighborhood[radius, radius:radius * 2])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 0] = maxValue
				gradientMin[h, w, 0] = minValue
				positionsMax[h, w, 0] = list(gradient).index(maxValue)
				positionsMin[h, w, 0] = list(gradient).index(minValue)
				
				# direction 5 (-1,0)
				gradient = np.gradient(neighborhood[radius, 0:radius])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 4] = maxValue
				gradientMin[h, w, 4] = minValue
				positionsMax[h, w, 4] = list(gradient).index(maxValue)
				positionsMin[h, w, 4] = list(gradient).index(minValue)
			
				# direction 7 (0,1)
				gradient = np.gradient(neighborhood[0:radius, radius])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 6] = maxValue
				gradientMin[h, w, 6] = minValue
				positionsMax[h, w, 6] = list(gradient).index(maxValue)
				positionsMin[h, w, 6] = list(gradient).index(minValue)
				
				# direction 3 (0,-1)
				gradient = np.gradient(neighborhood[radius:radius * 2, radius])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 2] = maxValue
				gradientMin[h, w, 2] = minValue
				positionsMax[h, w, 2] = list(gradient).index(maxValue)
				positionsMin[h, w, 2] = list(gradient).index(minValue)
								
				# direction 6 (-1,1)
				gradient = np.gradient(neighborhood.diagonal()[0:radius])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 5] = maxValue
				gradientMin[h, w, 5] = minValue
				positionsMax[h, w, 5] = list(gradient).index(maxValue)
				positionsMin[h, w, 5] = list(gradient).index(minValue)
				
				# direction 2 (-1,1)
				gradient = np.gradient(neighborhood.diagonal()[radius:radius * 2])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 1] = maxValue
				gradientMin[h, w, 1] = minValue
				positionsMax[h, w, 1] = list(gradient).index(maxValue)
				positionsMin[h, w, 1] = list(gradient).index(minValue)
				
				# direction 8 (1,1)
				gradient = np.gradient(np.flip(neighborhood, 0).diagonal()[0:radius])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 7] = maxValue
				gradientMin[h, w, 7] = minValue
				positionsMax[h, w, 7] = list(gradient).index(maxValue)
				positionsMin[h, w, 7] = list(gradient).index(minValue)
				
				# direction 4 (1,-1)
				gradient = np.gradient(np.flip(neighborhood, 0).diagonal()[radius:radius * 2])
				maxValue = max(gradient)
				minValue = min(gradient)
				gradientMax[h, w, 3] = maxValue
				gradientMin[h, w, 3] = minValue
				positionsMax[h, w, 3] = list(gradient).index(maxValue)
				positionsMin[h, w, 3] = list(gradient).index(minValue)
				
		print 'Computing gradients [OK]' 
				
		# cv2.imshow("gradient 1", positionsMin[:, :, 2])
		# cv2.waitKey(0)
		# cv2.imshow("gradient 1", test)				
		# cv2.waitKey(0)
		self.quantities(gradientMax, gradientMin, positionsMax, positionsMin, radius)
	
	def quantities(self, maxGradients, minGradients, positionsMax, positionsMin, radius):
		height, width = maxGradients[:, :, 0].shape  
		# S quantity, see paper
		symmetry = np.zeros((height, width), np.float64) 
		strength = np.zeros((height, width), np.float64)
		uniformity = np.zeros((height, width), np.float64)
		
		# to avoid reiterative computations, strenght function
		minGradientsAbs = np.absolute(minGradients)
		maxGradientsAbs = np.absolute(maxGradients)
		
		test = np.zeros((height, width), np.float64)
		
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
				
				# uniformity
				sigmaMaxTmp = self.sigma(positionsMax[h, w, :])
				sigmaMinTmp = self.sigma(positionsMin[h, w, :])
				meanTmpMax = np.mean(positionsMax[h, w, :])
				meanTmpMin = np.mean(positionsMin[h, w, :])
				
				# meanTmpMax = self.mean(h, w, positionsMax)
				# meanTmpMin = self.mean(h, w, positionsMin)
				
				# print sigmaMaxTmp
				
				# avoid division by zero
				meanTmpMax = meanTmpMax if meanTmpMax != 0.0 else SMALL_FLOAT
				meanTmpMin = meanTmpMin if meanTmpMin != 0.0 else SMALL_FLOAT
				
				uniformity[h, w] = (sigmaMaxTmp / meanTmpMax) + (sigmaMinTmp / meanTmpMin) 
											
				test[h, w] = uniformity[h, w] + strength[h, w] + symmetry[h, w]
			
		
		imageTmp = np.zeros((height, width), np.float64)+255.0
		for i in range(0,20):
			imageTmp[np.where(test == np.max(test))] = 0.0
			test[np.where(test == np.max(test))] = np.min(test)
			
		
		cv2.imshow("gradient 1", imageTmp)
		cv2.waitKey(0)  
		
		# test = np.ma.masked_where(test < 0.0, test)
		#cmap = plt.get_cmap('jet')
		# cmap.set_bad('white')
		#plt.imshow(test, cmap=cmap)
		#plt.show()
		
		# cv2.imshow("gradient 1", test)
		# cv2.waitKey(0)
			
	'''
	using the paper notation
	'''
	def delta(self, h, w, gradientJ, k):
		minTmp = min(gradientJ[h, w, k], gradientJ[h, w, k + 4])
		return (max(gradientJ[h, w, k], gradientJ[h, w, k + 4]) / (minTmp if minTmp != 0.0 else SMALL_FLOAT)) - 1.0  # to avoid division by zero
	
	'''
	Discretize version of equation (4) -> sigma function
	'''
	def sigma(self, positions):
		return  (float(np.absolute(sum(positions - np.mean(positions))))) / 8.0
		

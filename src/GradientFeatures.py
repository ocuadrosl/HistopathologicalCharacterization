import cv2
import numpy as np
import matplotlib.pyplot as plt


class GradientFeatures:
    
    magnitude=[]
    direction = []

    
    '''
    image must be grayscale
    '''
    def computeGradient(self, image):
        
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Find x and y gradients
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
     
        self.magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
        self.direction = np.arctan2(sobely, sobelx) * (180.0 / np.pi)
        
        #mask here
        self.magnitude[np.where(image == [0])] = [0] 
        self.direction[np.where(image == [0])] = [0]   
        

    '''
    just plot...
    another useless function
    '''
    def plotGradient(self):
        #mask to plot only
        self.direction = np.ma.masked_where(self.magnitude == 0, self.direction)
        self.magnitude = np.ma.masked_where(self.magnitude == 0, self.magnitude)
        
        cmap = plt.get_cmap('jet')
        cmap.set_bad('white')
        plt.figure(1)
        plt.imshow(self.magnitude, cmap=cmap)
        plt.figure(2)
        plt.imshow(self.direction , cmap=cmap)
        
        height = np.linspace(0, self.magnitude.shape[0], self.magnitude.shape[0] - 1);
        width = np.linspace(0, self.magnitude.shape[1], self.magnitude.shape[1] - 1);
        x, y = np.meshgrid(width, height)
  
        plt.figure(3)
        plt.quiver(x, y, cv2.flip(self.magnitude,1), cv2.flip(self.direction,1), units='dots', scale=100)
        
        plt.show()

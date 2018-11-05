import cv2 as cv2
import numpy as np
import copy 
from cv2 import threshold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from PIL.ImageOps import grayscale
#from pyatspi import component
from PIL import Image
from compiler.ast import Printnl

'''
In this class are implemented all methods to identify and categorize 
regions of interest in first level of microscopy resolution 
'''


class FirstLevel:
    components = []
    labels = []
    
    '''
    Main function to execute all the process and then extract connected components, 
    @param image:
    @param radius: to identify high density regions
    @param threshold: to extract the high density regions   
    '''

    def connectedComponents(self, image, radius=5, threshold=50):       
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        otsu = self.segmetBackground(imageGray, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        density = self.identifyHighDensity(otsu, radius)
        roi = self.extractHightDensityRegions(density, threshold)
        self.components = cv2.connectedComponentsWithStats(roi, 8, cv2.CV_32S)
    
    '''
    Ad hoc bachground segmentation  
    bests methods are otsu and triangle
    '''

    def segmetBackground(self, image, method):
    
        ret, otsu = cv2.threshold(image, 0, 255, method)
    
        image[np.where(otsu == [255])] = [0]
        
        return image

    '''
    This function computes the probality [0-100] of a pixel belongs to a 
    high density region
    '''

    def identifyHighDensity(self, image, radius):
        height = image.shape[0]
        width = image.shape[1]
        
        maxDensity = pow(radius * 2, 2) 
       
        output = copy.deepcopy(image)
        for h in range(0, height):
            for w in range(0, width):
                if image[h, w] != 0:  # 255
                    count = 0
                    for i in range(h - radius, h + radius):
                        for j in range(w - radius, w + radius):
                            try:  # out of bound, improve it... 
                                if image[i, j] > 0:  # <255
                                    count = count + 1
                                    # print count
                            except:
                                pass  # do nothing
                    
                    # if (count * 100) / maxDensity > 100:
                        # print (count * 100) / maxDensity
                    output[h, w] = ((count * 100) / maxDensity)  # 255- ...
                           
        return output
   
    '''
    This funciton extracts the hight density regions
    @param densityImage: output of the identifyHighDensity function
    @param threhold: simple threshold value to separate the regions of interest  
    '''

    def extractHightDensityRegions(self, densityImage, threshold):
        
        ret, roi = cv2.threshold(densityImage, threshold, 255, cv2.THRESH_BINARY)
        return roi
    
    '''
    To plot components, background is setup as wh    ite 
    '''

    def plotComponents(self):
        components = np.ma.masked_where(self.components[1] == 0, self.components[1])
        cmap = plt.get_cmap('hot')
        cmap.set_bad('white')
        plt.imshow(components, cmap=cmap)
        plt.show()
        
    def writeComponentsAsImages(self, fileName):
        height, width = self.components[1].shape
        
        for i in range(0, self.components[0]):
            component = np.zeros((height, width, 3), np.uint8) + 255
            component[np.where(self.components[1] == [i])] = [i]
            cv2.imwrite(fileName + "_" + str(i) + ".png", component)
    

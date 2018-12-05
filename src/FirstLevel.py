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
# from pyatspi import component
from PIL import Image
from compiler.ast import Printnl
from time import ctime
import os


'''
In this class are implemented all methods to identify and categorize 
regions of interest at lower magnification 
'''


class FirstLevel:
    components = []
    labels = []
    imageGray = []
    
    '''
    Main function to execute all process and then extract connected components, 
    @param image:
    @param radius: to identify high density regions
    @param threshold: to extract the high density regions   
    '''

    def connectedComponents(self, image, radius=5, threshold=50):       
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.imageGray = copy.deepcopy(imageGray)
        foreground = self.segmetBackground(imageGray, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        density = self.identifyHighDensity(foreground, radius)
        # first threhold higth dendity regions     
        ret, hight = cv2.threshold(density, threshold, 255, cv2.THRESH_BINARY)  
        ret, low = cv2.threshold(density, threshold, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('low', low)
        #cv2.imshow('hight', hight)
        #cv2.waitKey(0)
        # roi = self.extractHightDensityRegions(density, threshold)
        # components
        self.components = cv2.connectedComponentsWithStats(hight, 8, cv2.CV_32S)
    
    
	'''
    Ad hoc bachground segmentation  
    best methods are otsu and triangle
    '''

    def segmetBackground(self, image, method):
    
        ret, otsu = cv2.threshold(image, 0, 255, method)
    
        image[np.where(otsu == [255])] = [0]
        
       #cv2.imshow('image', image)
       #cv2.waitKey(0)
        
        return image

    '''
    This function computes the probality [0-100] of a pixel belongs to a 
    high density region
    '''

    def identifyHighDensity(self, foreground, radius):
        height = foreground.shape[0]
        width = foreground.shape[1]
        
        maxDensity = pow(radius * 2, 2) 
       
        output = copy.deepcopy(foreground)
        for h in range(0, height):
            for w in range(0, width):
                if foreground[h, w] != 0:  # 255
                    count = 0
                    for i in range(h - radius, h + radius):
                        for j in range(w - radius, w + radius):
                            try:  # out of bound, improve it... 
                                if foreground[i, j] > 0:  # <255
                                    count = count + 1
                                    # print count
                            except:
                                pass  # do nothing
                    
                    # if (count * 100) / maxDensity > 100:
                        # print (count * 100) / maxDensity
                    output[h, w] = ((count * 100) / maxDensity)  # 255- ...
                           
        return output
   
    '''
    This funciton extracts the high density regions
    @param densityImage: output of the identifyHighDensity function
    @param threhold: simple threshold value to separate the regions of interest  
    '''

    def extractHightDensityRegions(self, densityImage, threshold):
				ret, hight = cv2.threshold(densityImage, threshold, 255, cv2.THRESH_BINARY)
				ret, low = cv2.threshold(densityImage, threhold, 255, cv2.THRESH_BINARY_INV)
				return hight  
    


    '''
    To plot components, background is setup as wh    ite 
    '''

    def plotComponents(self):
        components = np.ma.masked_where(self.components[1] == 0, self.components[1])
        cmap = plt.get_cmap('hot')
        cmap.set_bad('white')
        plt.imshow(components, cmap=cmap)
        plt.show()
        
    def writeComponentsAsImages(self, mainDir, imageName):
               
        # create  dir
        dirName = mainDir + "/" + imageName + "_" + str(ctime())
        try:
            os.mkdir(dirName)
        except FileExistsError:
            print("Directory " , dirName , " error creating the output directory")
                
        height, width = self.components[1].shape
        fileName = dirName + "/" + imageName
        #extra image to write all low density regions
        lowDensity = copy.deepcopy(self.imageGray)
        highDensity = cv2.cvtColor(np.zeros((height, width, 3), np.uint8), cv2.COLOR_RGB2GRAY)
       
        for i in range(1, self.components[0]):
            component = np.zeros((height, width, 3), np.uint8)
            component[np.where(self.components[1] == [i])] = [255] #[i]
            #cv2.imwrite(fileName + "_" + str(i) + ".png", component)
            
            #convex hull here
            mask = self.findConvexHull(component)
            
            lowDensity[np.where(mask == [255])] = [0]
            highDensity[np.where(mask == [255])] = self.imageGray[np.where(mask == [255])] 
            mask[np.where(mask == [255])] = self.imageGray[np.where(mask == [255])]
            
            cv2.imwrite(fileName + "_" + str(i) + ".png", mask)
       
        cv2.imwrite(fileName + "_low"  + ".png", lowDensity)
        cv2.imwrite(fileName + "_high"  + ".png", highDensity)


        
    
    
    
    
    def findConvexHull(self, image):
        imageGray =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, threshed_img = cv2.threshold(imageGray, 0, 1, cv2.THRESH_BINARY) #orignal 1 
        img, contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        for cnt in contours:
            epsilon = 0.0001*cv2.arcLength(cnt,True) #0.01
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            hull = cv2.convexHull(approx)
            cv2.drawContours(img, [hull], -1, (255, 255, 255),-1)
        return img   
        
    
            
        
        
    
    

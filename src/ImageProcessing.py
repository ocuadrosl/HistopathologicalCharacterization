import cv2 as cv2
import numpy as np
import copy 

#bests methods are otsu and triangle
def segmetBackground(image, method):
    
    ret, otsu = cv2.threshold(image, 0, 255, method)
    
    image[np.where(otsu == [255])] = [0]
        
    return image

'''
This function computes the probality [0-100] of a pixel belongs to a 
high density region
'''
def identifyHighDensity(image, radius):
    height = image.shape[0]
    width = image.shape[1]
    
    maxDensity = pow(radius * 2, 2) 
   
    output = copy.deepcopy(image)
    for h in range(0, height):
        for w in range(0, width):
            if image[h, w] != 0: # 255
                count = 0
                for i in range(h - radius, h + radius):
                    for j in range(w - radius, w + radius):
                        try:  # out of bound, improve it... 
                            if image[i, j] > 0: # <255
                                count = count + 1
                                # print count
                        except:
                            pass  # do nothing
                
                #if (count * 100) / maxDensity > 100:
                   # print (count * 100) / maxDensity
                output[h, w] = ((count * 100) / maxDensity) # 255- ...
                       
    return output



'''
This funciton extracts the hight density regions
@param densityImage: output of the identifyHighDensity function
@param threhold: simple threshold value to separate the regions of interest  
'''
def extractHightDensityRegions(densityImage, threshold):
    
    ret, roi = cv2.threshold(densityImage, threshold, 255, cv2.THRESH_BINARY)
    return roi


    
    
    
    

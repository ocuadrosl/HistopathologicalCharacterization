import cv2 as cv
import numpy as np
import copy 


def segmetBackground(image):
    # otsu es una bosta
    ret, otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    image[np.where(otsu == [255])] = [255]
        
    return image


def identifyHighDensity(image, radius):
    height = image.shape[0]
    width = image.shape[1]
    
    maxDensity = pow(radius*2,2) 
   
    output = copy.deepcopy(image)
    for h in range(0, height):
        for w in range(0, width):
            if image[h, w] != 255:
                count = 0
                for i in range(h - radius, h + radius):
                    for j in range(w - radius, w + radius):
                        try:  # out of bound, improve it... 
                            if image[i, j] < 255:
                                count = count + 1
                                # print count
                        except:
                            pass  # do nothing
                
                if  (count * 100) / maxDensity > 100:
                    print (count * 100) / maxDensity
                output[h, w] = (count * 100) / maxDensity
                       
    return output
    

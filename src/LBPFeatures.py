from skimage import feature
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

class LBPFeatures:
    def computeLBP(self, image):
        
        mask =  np.ma.masked_where(image == 0, image)
        plt.imshow(mask)
        #plt.imshow(image)
        plt.show()
        #cv2.imshow('low', mask)
        #cv2.waitKey(0)        
        lbp = feature.local_binary_pattern(image, 8, 1, method="uniform")
        
        
        
        return lbp
        #cv2.imshow('low', lbp)
        #cv2.waitKey(0)
    
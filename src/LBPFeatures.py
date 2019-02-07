from skimage import feature
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import copy
import mahotas

class LBPFeatures:
    
    '''
    This function computes the LBP pattern on the masked RO images (output of FirstLevel)
    after computing LBP, a mask (weights) is created in which zero value is assigned to the background (black)
    then the  histogram (output) is computed ignoring the background
    TODO parameters...  
    '''
    def computeLBP(self, image):
        nPoints = 8#8
        radius = 24#1
        #lbp = feature.local_binary_pattern(image, nPoints, radius, method="uniform", ignore_zeros=True)
        lbp = mahotas.features.lbp(image, radius, nPoints, ignore_zeros=True)
        #print lbp
        '''
        mask = copy.deepcopy(lbp)
        mask[np.where(image != [0])] = [1] 
        mask[np.where(image == [0])] = [0]   
        '''
        #cv2.imshow('mask', mask)
        #cv2.waitKey(0)
        
        #TODO optimize bins and range
        nBins = 10# int(lbp.max() + 1)
                
        hist, _ = np.histogram(lbp,  bins=nBins)#, weights=mask)  
        return hist
    
   
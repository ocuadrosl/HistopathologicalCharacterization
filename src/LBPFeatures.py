from skimage import feature
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import copy

class LBPFeatures:
    
    '''
    This function computes the LBP pattern on the masked RIO images (output of FirstLevel)
    after computing LBP  a mask (weights) is created in which zero velue is assigned to the background (black)
    then the  histogram (output) is computed ignoring the background  
    '''
    def computeLBP(self, image):
        lbp = feature.local_binary_pattern(image, 8, 1, method="uniform")
        
        mask = copy.deepcopy(lbp)
        mask[np.where(image != [0])] = [1] 
        mask[np.where(image == [0])] = [0]   
        
        #TODO optimize bins and range
        nBins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, normed=True, bins=nBins, range=(0, nBins + 2), weights=mask)        
        
        return hist
    
   
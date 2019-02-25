from skimage import feature
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import copy
import mahotas

'''
    This function computes the LBP pattern on the masked RO images (output of FirstLevel)
    after computing LBP, a mask (weights) is created in which zero value is assigned to the background (black)
    then the  histogram (output) is computed ignoring the background
    TODO parameters...  
    '''
def computeLBP(image):
    
    radius = 1#5
    nPoints = 8#8 #* radius
    
    return mahotas.features.lbp(image, radius, nPoints, ignore_zeros=True)
    
   
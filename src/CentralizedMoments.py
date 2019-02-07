from skimage import feature
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import copy

class CentralizedMoments:
    
    '''
    This function computes Centralized Moments: Raw or Spatial Moments; Central Moments; 
    Central Standardized or Normalized or Scale Invariant Moments
    TODO parameters...  
    '''
    def computeCentralizedMoments(self, image):
        x, y = np.mgrid[:image.shape[0],:image.shape[1]]
        moments = []
         # raw or spatial moments
        mean_x = np.sum(x*image)/np.sum(image) 
        moments.append(mean_x)  
        mean_y = np.sum(y*image)/np.sum(image)
        moments.append(mean_y) 
        m00 = np.sum(image)
        moments.append(m00) 
        m01 = np.sum(x*image)
        moments.append(m01)  
        m10 = np.sum(y*image)
        moments.append(m10)
        m11 = np.sum(y*x*image) 
        moments.append(m11) 
        m02 = np.sum(x**2*image)
        moments.append(m02)
        m20 = np.sum(y**2*image)
        moments.append(m20)
        m12 = np.sum(x*y**2*image) 
        moments.append(m12)
        m21 = np.sum(x**2*y*image)
        moments.append(m21)
        m03 = np.sum(x**3*image)
        moments.append(m03)
        m30 = np.sum(y**3*image)
        moments.append(m30)
         # central moments
        mu11 = np.sum((x-mean_x)*(y-mean_y)*image)
        moments.append(mu11)
        mu02 = np.sum((y-mean_y)**2*image) # variance
        moments.append(mu02)
        mu20 = np.sum((x-mean_x)**2*image) # variance
        moments.append(mu20)
        mu12 = np.sum((x-mean_x)*(y-mean_y)**2*image)
        moments.append(mu12)
        mu21 = np.sum((x-mean_x)**2*(y-mean_y)*image) 
        moments.append(mu21)
        mu03 = np.sum((y-mean_y)**3*image) 
        moments.append(mu03)
        mu30 = np.sum((x-mean_x)**3*image) 
        moments.append(mu30)
        
        
        # central standardized or normalized or scale invariant moments
        nu11 = mu11 / np.sum(image)**(2/2+1)
        moments.append(nu11)
        nu12 = mu12 / np.sum(image)**(3/2+1)
        moments.append(nu12)
        nu21 = mu21 / np.sum(image)**(3/2+1)
        moments.append(nu21)
        nu20 = mu20 / np.sum(image)**(2/2+1)
        moments.append(nu20)
        nu03 = mu03 / np.sum(image)**(3/2+1) # skewness
        moments.append(nu03)
        nu30 = mu30 / np.sum(image)**(3/2+1) # skewness
        moments.append(nu30)
               
        return moments   
   
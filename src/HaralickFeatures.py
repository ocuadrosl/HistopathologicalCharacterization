import numpy as np
import mahotas

class HaralickFeatures:
    
    '''
    This function computes the LBP pattern on the masked RO images (output of FirstLevel)
    after computing LBP, a mask (weights) is created in which zero value is assigned to the background (black)
    then the  histogram (output) is computed ignoring the background
    TODO parameters...  
    '''
    def computeHaralick(self, image):
        moments = mahotas.features.haralick(image, ignore_zeros=True)
        return moments
    
   
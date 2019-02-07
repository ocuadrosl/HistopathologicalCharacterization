import numpy as np
import mahotas

class PFTASFeatures:
    
    '''
    This function computes the LBP pattern on the masked RO images (output of FirstLevel)
    after computing LBP, a mask (weights) is created in which zero value is assigned to the background (black)
    then the  histogram (output) is computed ignoring the background
    TODO parameters...  
    '''
    def computePFTASFeatures(self, image):
        moments = mahotas.features.pftas(image)
        return moments
    
   
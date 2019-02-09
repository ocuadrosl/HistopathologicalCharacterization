
import numpy as np
'''
physical in micrometers
'''


def computeResolution(physicalX, physicalY, sizeX, sizeY, inputMagnification, outputMagnification):
    
    eyePice   = 10.0
    magnificationDifference = ((inputMagnification*eyePice) - (outputMagnification*eyePice)) + 0.00000000001 
        
    outputPhysicalX = (physicalX * magnificationDifference) + physicalX
    outputPhysicalY = (physicalY * magnificationDifference) + physicalY
        
    sizeXOutput = sizeX / outputPhysicalX
    sizeYOutput = sizeY / outputPhysicalY
    
    return sizeXOutput * sizeYOutput
    
   
def minMax(inputValue, orgMin, orgMax, newMin, newMax):
    den = 0.00000001 if  orgMax == orgMin else orgMax - orgMin 
    return  ((newMax - newMin) * (inputValue - orgMin) / den) + newMin

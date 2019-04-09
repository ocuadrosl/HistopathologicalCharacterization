
import numpy as np
'''
physical in micrometers
'''


def computeResolution(physicalX, physicalY, sizeX, sizeY, inputMagnification, outputMagnification):
    
    eyePice   = 10.0
    magnificationDifference = ((inputMagnification*eyePice) - (outputMagnification*eyePice)) + 0.00000000001 
        
    outputPhysicalX = (physicalX * magnificationDifference) + physicalX
    outputPhysicalY = (physicalY * magnificationDifference) + physicalY
    
    #print outputPhysicalY, outputPhysicalY
        
    sizeXOutput = sizeX / outputPhysicalX
    sizeYOutput = sizeY / outputPhysicalY
    
   #print sizeXOutput, sizeYOutput
    
    return sizeXOutput * sizeYOutput
    
   
def minMax(inputValue, orgMin, orgMax, newMin, newMax):
    den = 0.00000001 if  orgMax == orgMin else orgMax - orgMin 
    #print inputValue, ( ((newMax - newMin) * (inputValue - orgMin)) / den) + newMin
    return  ( ((newMax - newMin) * (inputValue - orgMin)) / den) + newMin

def cartesianToPolar(coordinates):
    
    r = np.sqrt(coordinates[0]**2 + coordinates[1]**2)
    t = np.arctan2(coordinates[1],coordinates[0])
    return (r,t) 
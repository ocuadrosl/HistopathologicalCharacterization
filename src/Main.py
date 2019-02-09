import os
import javabridge
import bioformats
from bioformats import log4j
import sys
import pylab
import numpy as np
from bioformats.omexml import OMEXML
from Utils import *
from FirstLevel import FirstLevel

import cv2
from ImageProcessing import *

fileName = "/home/oscar/src/HistopathologicalCharacterization/input/B526-18  B 20181107/Image01B526-18  B .vsi"

firstLevel = FirstLevel()
firstLevel.compute(fileName, 7, 0.5, 9,9)


'''

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True, max_heap_size='8G')
try:
    log4j.basic_config()
    directoryPath = "/home/oscar/src/HistopathologicalCharacterization/input/B526-18  B 20181107/Image01B526-18  B .vsi"
  
      
    ImageReader = bioformats.formatreader.make_image_reader_class()
    reader = ImageReader()
    reader.setId(directoryPath)
    ##
    rdr = bioformats.get_image_reader(None, path=directoryPath)
    totalseries = 1
    try:
        totalseries = np.int(rdr.rdr.getSeriesCount())
    except:
        print("exc")
        totalseries = 1  # in case there is only ONE series
    ##
    
    
   # print(reader.getSizeY(), reader.getSizeX())
    width = 6000
    height = 6000
    ome = OMEXML(bioformats.get_omexml_metadata(path=directoryPath))
    x = ome.image().Pixels.SizeY
    y = ome.image().Pixels.SizeX
    c = ome.image().Pixels.get_SizeC()
    t = ome.image().Pixels.get_SizeT()
    z = ome.image().Pixels.get_SizeZ()
    
    
    #print ome.instrument(0).Objective.get_NominalMagnification()
    #print width*height
    physicalX = ome.image().Pixels.get_PhysicalSizeX()
    physicalY = ome.image().Pixels.get_PhysicalSizeY()
   
    newResolution = computeScaleFactor(physicalX, physicalY, width, height, 40, 5)
       
        
    rdr = bioformats.ImageReader(directoryPath)
    format_reader = rdr.rdr
    data = reader.openBytesXYWH(0,50098,50098,width,height)
      

    data.shape = (width, height, 3)
    
    image = adaptiveResize(data, newResolution)
    
    
    cv2.imwrite("/home/oscar/src/HistopathologicalCharacterization/input/B526-18  B 20181107/Image01B526-18  B .tiff", image)
    cv2.imshow("tile", image)
    cv2.waitKey()
    #pylab.imshow(data)
    #pylab.gca().set_title("tile")
    #pylab.show()
    
finally:
    javabridge.kill_vm()
    
'''
import cv2
from ImageProcessing import *
from FirstLevel import FirstLevel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from PIL.ImageOps import grayscale

print "START"

# TODO read input files and parameters from command line or input file...
imageDir = "/home/oscar/MEGA/post-doc/src/input/rp/patient_1/00529 (1).jpg"
#imageDir = "/home/oscar/MEGA/post-doc/src/input/rp/patient_1/small_1.jpg"
image = cv2.imread(imageDir)

# verify original size before
height, width = image.shape[:2]
scaleFactor =   250000.0 / (height * width)   
scaleFactor = ((scaleFactor*100.0)/1.0)
print height ,  width, scaleFactor 
if scaleFactor < 1.0:
    image = rezise(image, 0.2)  # 0.2
    print scaleFactor


imageName = imageDir.split("/").pop().split(".")[0]
firstLevel = FirstLevel()
firstLevel.connectedComponents(image, radius=10, threshold=50)
firstLevel.writeComponentsAsImages("../output", imageName)
firstLevel.plotComponents()

# cv2.waitKey(0)

print "DONE"

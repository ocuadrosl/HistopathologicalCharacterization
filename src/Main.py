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

#TODO read input files and parameters from command line or input file...
imageDir = "/home/oscar/MEGA/post-doc/src/input/rp/patient_2/06960 (1).jpg"
image = cv2.imread(imageDir)

#TODO verify original size before
image = rezise(image, 0.2)


imageName = imageDir.split("/").pop().split(".")[0]
firstLevel = FirstLevel()
firstLevel.connectedComponents(image, radius=10, threshold=60)
firstLevel.writeComponentsAsImages("../output", imageName)
#firstLevel.plotComponents()

#cv2.waitKey(0)

print "DONE"

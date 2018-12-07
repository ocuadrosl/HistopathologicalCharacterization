import cv2
from ImageProcessing import *
from FirstLevel import FirstLevel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from PIL.ImageOps import grayscale
from  GradientFeatures import GradientFeatures
from LBPFeatures import LBPFeatures


imageDir = "../input/high2.png"
image = cv2.imread(imageDir)
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lbpFearures = LBPFeatures()
hist= lbpFearures.computeLBP(imageGray)

plt.hist(hist)
plt.show()


quit()

gF = GradientFeatures()
gF.computeGradient(imageGray)
gF.plotGradient()





print "START"

# TODO read input files and parameters from command line or input file...
imageDir = "/home/oscar/MEGA/post-doc/src/input/rp/patient_1/00529 (1).jpg"
image = cv2.imread(imageDir)

image = adaptiveResize(image)  # 0.2


imageName = imageDir.split("/").pop().split(".")[0]
firstLevel = FirstLevel()
firstLevel.connectedComponents(image, radius=6, threshold=60)
firstLevel.writeComponentsAsImages("../output", imageName)
firstLevel.plotComponents()

# cv2.waitKey(0)

print "DONE"

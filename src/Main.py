import cv2
from ImageProcessing import *


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg


imageDir = "/home/oscar/MEGA/post-doc/src/input/rp/patient_1/00529 (1).jpg"
image = cv2.imread(imageDir)

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


otsu = segmetBackground(imageGray)

density = identifyHighDensity(otsu, 7)

cv2.imwrite('00529 (1).png',density)

cv2.imshow("input Image", density);


#plt.imshow(density, cmap='hot')


cv2.waitKey(0)
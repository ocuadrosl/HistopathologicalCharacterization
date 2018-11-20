import cv2
from ImageProcessing import *
from FirstLevel import FirstLevel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from PIL.ImageOps import grayscale


'''
imageDir = "../input/others/roi2.png"
image = cv2.imread(imageDir)
imageGray =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
imageGray[np.where(imageGray == [255])] = [0]
imageGray[np.where(imageGray > [0])] = [255]

ret, threshed_img = cv2.threshold(imageGray, 0, 1, cv2.THRESH_BINARY)
img, contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

for cnt in contours:
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    hull = cv2.convexHull(approx)
    cv2.drawContours(img, [hull], -1, (255, 255, 0))


cv2.imshow("rectangle", img)
cv2.waitKey(0)

quit()
'''



imageDir = "../input/others/patient2.vsi"
image = cv2.imread(imageDir)
#image = rezise(image, 0.2)

imageName = imageDir.split("/").pop().split(".")[0]
firstLevel = FirstLevel()
firstLevel.connectedComponents(image, radius=10)
firstLevel.writeComponentsAsImages("../output", imageName)
firstLevel.plotComponents()


plt.show()

#cv2.waitKey(0)

print "DONE"
import cv2
from ImageProcessing import *
import numpy as np

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg

imageDir = "/home/oscar/MEGA/post-doc/src/input/rp/patient_1/small_4.jpg"
image = cv2.imread(imageDir)

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

otsu = segmetBackground(imageGray, cv2.THRESH_TRIANGLE) #cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE 

cv2.imwrite('../output/otsu.png', otsu)

density = identifyHighDensity(otsu, 5)

# cd cv2.imwrite('../output/small_1.png',density)

# cv2.imshow("input Image", density);

density = np.ma.masked_where(density == 255, density)
cmap = plt.get_cmap('seismic')
cmap.set_bad('white')

plt.imsave('../output/small_4.png', density, cmap=cmap)
plt.imshow(density, cmap=cmap)

# plt.colorbar()
plt.show()

# cv2.waitKey(0)

print "DONE"

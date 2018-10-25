import cv2
from ImageProcessing import *
from FirstLevel import FirstLevel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from PIL.ImageOps import grayscale



imageDir = "/home/oscar/MEGA/post-doc/src/input/rp/patient_1/small_1.jpg"
image = cv2.imread(imageDir)


firstLevel = FirstLevel()
firstLevel.connectedComponents(image)
#firstLevel.plotComponents()
firstLevel.writeComponentsAsImages("../output/component")





#cd cv2.imwrite('../output/small_1.png',density)

#cv2.imshow("input Image", density);

#TODO falta threshold aqui 

#density = np.ma.masked_where(density == 0, density)
#cmap = plt.get_cmap('seismic')
#cmap.set_bad('black')


#plt.imsave('../output/small_1.png', density, cmap=cmap)
#plt.imshow(density, cmap=cmap)


#plt.colorbar()
plt.show()

#cv2.waitKey(0)

print "DONE"
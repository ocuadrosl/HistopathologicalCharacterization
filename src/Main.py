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
from SecondLevel import SecondLevel
from LBPFeatures import *
import matplotlib.pyplot as plt
from gui.MainGui import *
from gui.ActionsGui import *
import cv2
from ImageProcessing import *
from PyQt4.QtGui import (QMainWindow, QApplication)


fileName = "/home/aurea/eclipse-workspace/HistopathologicalCharacterization/input/others/test.png"
image = cv2.imread(fileName)
#gaussianImage = cv2.GaussianBlur(image,(15,15),0)
#cv2.imshow("gaussian", gaussianImage)
#cv2.waitKey(0)
secondLevel = SecondLevel()
secondLevel.ERSTransform(image, 10)

quit()




        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


#
quit()





fileName = "/home/oscar/eclipse-workspace/HistopathologicalCharacterization/input/B2046_18 B20181107/Image01B2046_18 B.vsi"
firstLevel = FirstLevel()



# for large samples
high, low, density, gray = firstLevel.identifyHighDensityLargeSample(fileName, 7, 0.05, 9,9, 60)
cv2.imwrite("/home/oscar/eclipse-workspace/HistopathologicalCharacterization/output//Image01B2046_18 B_high.png" , high)
cv2.imwrite("/home/oscar/eclipse-workspace/HistopathologicalCharacterization/output/Image01B2046_18 B_low.png" , low)
firstLevel.writeDensityImage(density, "/home/oscar/eclipse-workspace/HistopathologicalCharacterization/output/Image01B2046_18 B_colormap.png")
cv2.imwrite("/home/oscar/eclipse-workspace/HistopathologicalCharacterization/output/Image01B2046_18 B_gray.png" , gray)



'''

## for small samples
fileName  = "/home/oscar/src/HistopathologicalCharacterization/input/rp/patient_1/00529 (2).jpg"
image = cv2.imread(fileName)
image = adaptiveResize(image)  # 0.2
imageName = fileName.split("/").pop().split(".")[0]
density = firstLevel.connectedComponents(image, radius=7, threshold=60)
firstLevel.writeComponentsAsImages("../output", imageName)
cv2.imwrite("/home/oscar/src/HistopathologicalCharacterization/output/test/00529 (2)_density.tiff" , density)
firstLevel.writeDensityImage(density, "/home/oscar/src/HistopathologicalCharacterization/output/test/00529 (2)_colormap.tiff")
#firstLevel.plotComponents()

'''


'''

fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/00529 (2)_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/00529 (2)_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)
plt.figure(1)
plt.bar(np.arange(len(lbpRoi)), lbpRoi, align='center', alpha=0.5)
#plt.title('LOW')
plt.figure(2)
plt.bar(np.arange(len(lbpNoRoi)), lbpNoRoi, align='center', alpha=0.5)
plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi - np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')


plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])
#plt.show()


fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/00529 (4)_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/00529 (4)_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)
plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi - np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])

fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/00529_1_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/00529_1_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)
plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])

fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (1)_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (1)_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)
plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi - np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])

fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (2)_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (2)_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)
plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi - np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])

fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (3)_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (3)_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)
plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi - np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])

fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (4)_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/06960 (4)_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)
plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi - np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])

fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/Image01B2046_18 B_high.tiff"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/Image01B2046_18 B_low.tiff"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)

plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi- np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])



fileNameRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/Image01B526-18  B_high.png"
fileNameNoRoi = "/home/oscar/src/HistopathologicalCharacterization/output/test/Image01B526-18  B_low.png"

roi  = cv2.imread(fileNameRoi, cv2.IMREAD_GRAYSCALE)
noRoi = cv2.imread(fileNameNoRoi, cv2.IMREAD_GRAYSCALE)

lbpRoi = computeLBP(roi)
lbpNoRoi = computeLBP(noRoi)

plt.figure(3)
newlbpRoi = (lbpRoi - np.min(lbpRoi)) / (np.max(lbpRoi) - np.min(lbpRoi))
newlbpNoRoi = (lbpNoRoi- np.min(lbpNoRoi)) / (np.max(lbpNoRoi) - np.min(lbpNoRoi))
plt.plot(newlbpNoRoi)
#plt.title('LOW')
plt.figure(4)
plt.plot(newlbpRoi)
#plt.title('HIGH')
print(["%.8f" % v for v in newlbpRoi])
print(["%.8f" % v for v in newlbpNoRoi])


plt.show()

'''
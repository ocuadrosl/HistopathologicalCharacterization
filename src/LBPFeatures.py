from skimage import feature
import numpy as np
import cv2 as cv2

class LBPFeatures:
    def computeLBP(self, image):
        lbp = feature.local_binary_pattern(image, 24, 20, method="uniform")
        cv2.imshow('low', lbp)
        cv2.waitKey(0)
    
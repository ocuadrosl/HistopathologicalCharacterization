import cv2 as cv2
import numpy as np
import copy 
from cv2 import threshold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from PIL.ImageOps import grayscale
from PIL import Image
from compiler.ast import Printnl
from time import ctime
import os
import javabridge
import bioformats
from bioformats import log4j
from bioformats.omexml import OMEXML
from Utils import *
from ImageProcessing import *
from Xlib.Xutil import HeightValue
import thread

import gc
import PIL as pil

'''
In this class are implemented all methods to identify and categorize 
regions of interest at lower magnification 
'''


class FirstLevel:
    components = []
    labels = []
    imageGray = []
    roiMasks = []  # vector of ROI components
    roi = []  # Region of interest
    nonROi = []  # non ROI
    
    def compute(self, fileName, radius, outputMagnification, numberOfTilesX, numberOfTilesY):
        
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True, max_heap_size='8G')
        
        try:
            log4j.basic_config()
            imageReader = bioformats.formatreader.make_image_reader_class()
            reader = imageReader()
            reader.setId(fileName)
    # #
            rdr = bioformats.get_image_reader(None, path=fileName)
            totalseries = 1
            try:
                totalseries = np.int(rdr.rdr.getSeriesCount())
            except:
                print("exc")
                totalseries = 1  # in case there is only ONE series
        
            # Get image Metadata 
            ome = OMEXML(bioformats.get_omexml_metadata(path=fileName))
            sizeX = ome.image().Pixels.get_SizeX()
            sizeY = ome.image().Pixels.get_SizeY()
            print sizeX, sizeY
            
            physicalX = ome.image().Pixels.get_PhysicalSizeX()
            physicalY = ome.image().Pixels.get_PhysicalSizeY()
             
            inputMagnification = np.round(np.float(ome.instrument(0).Objective.get_NominalMagnification()), 0)
                      
            # initialize variables         
            tileBeginX = 0
            tileBeginY = 0
            format_reader = bioformats.ImageReader(fileName).rdr
            imageNumber = 1;
            
            hMosaic = []
            vMosaic = []
            
            for y in range(0, numberOfTilesY):  # <=
                        
                # computing begin and Y size 
                tileBeginY = minMax(y , 0, numberOfTilesY, 0, sizeY)
                height = minMax(y + 1 , 0, numberOfTilesY, 0, sizeY) - tileBeginY
                    
                for x in range(0, numberOfTilesX):  # <=   
                                       
                    # computing begin and X size
                    tileBeginX = minMax(x , 0, numberOfTilesX, 0, sizeX)
                    width = minMax(x + 1 , 0, numberOfTilesX, 0, sizeX) - tileBeginX
                    # print x, y 
                    print tileBeginX, tileBeginY, width , height
                                        
                    newResolution = computeResolution(physicalX, physicalY, width, height, inputMagnification , outputMagnification)
                    
                    # extracting tile                    
                    tile = reader.openBytesXYWH(0, tileBeginX, tileBeginY, width, height)
                    tile.shape = (height, width, 3)
                    
                    # resize tile
                    image = adaptiveResize(tile, newResolution)
                    
                    if(x > 0):
                        hMosaic  = np.concatenate((hMosaic, image), axis=0)
                    else:
                        hMosaic = image
                    # parallel computation goes here 
                    # self.connectedComponents(image, radius)  
                    # self.writeComponentsAsImages("/home/oscar/src/HistopathologicalCharacterization/output/tiles", "tile_"+str(imageNumber))                  
                    # cv2.imwrite("/home/oscar/src/HistopathologicalCharacterization/input/B526-18  B 20181107/tiles/tile_" + str(imageNumber) + ".tiff" , image)
                    # cv2.imshow("tile", image)
                    # cv2.waitKey()
                    
                    imageNumber = imageNumber + 1
                                
                # free memory               
                gc.collect()
                if(y > 0 ):
                    vMosaic  = np.concatenate((vMosaic, hMosaic), axis=1)
                else:
                    vMosaic = hMosaic
                
                hMosaic = []
                #mosaic = pil.Image.fromarray(vMosaic)
                #mosaic.save("/home/oscar/src/HistopathologicalCharacterization/output/mosaic.tiff")
                cv2.imwrite("/home/oscar/src/HistopathologicalCharacterization/output/mosaic.tiff" , vMosaic)
                
        finally:
            javabridge.kill_vm()
    
        print "succes"    
    
    '''
    Main function to execute all process and then extract connected components, 
    @param image:
    @param radius: to identify high density regions
    @param threshold: to extract the high density regions   
    '''

    def connectedComponents(self, image, radius=5, threshold=50):       
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('gray.png', imageGray)
        self.imageGray = copy.deepcopy(imageGray)
        foreground = self.segmentBackground(imageGray, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imwrite('otsu.png', foreground)
       
        density = self.identifyHighDensity(foreground, radius)
        # first threhold higth dendity regions     
        ret, high = cv2.threshold(density, threshold, 255, cv2.THRESH_BINARY)  
        ret, low = cv2.threshold(density, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # cv2.imshow('low', low)
        # cv2.imshow('high', high)
        # cv2.waitKey(0)
        
        # cv2.imwrite('high.png', high)
        # cv2.imwrite('low.png', low)
        # roi = self.extractHightDensityRegions(density, threshold)
        # components
        self.components = cv2.connectedComponentsWithStats(high, 8, cv2.CV_32S)
        
        # find convex hulls
        height, width = self.components[1].shape        
        self.roi = copy.deepcopy(self.imageGray)
        self.nonRoi = cv2.cvtColor(np.zeros((height, width, 3), np.uint8), cv2.COLOR_RGB2GRAY)
       
        for i in range(1, self.components[0]):
            component = np.zeros((height, width, 3), np.uint8)
            component[np.where(self.components[1] == [i])] = [255]  
                    
            mask = self.findConvexHull(component)
            
            self.roi[np.where(mask == [255])] = [0]
            self.nonRoi[np.where(mask == [255])] = self.imageGray[np.where(mask == [255])] 
            mask[np.where(mask == [255])] = self.imageGray[np.where(mask == [255])]
            
            self.roiMasks.append(mask);
    
    '''
    Ad hoc bachground segmentation  
    best methods are otsu and triangle
    '''

    def segmentBackground(self, image, method):
    
        ret, otsu = cv2.threshold(image, 0, 255, method)
    
        image[np.where(otsu == [255])] = [0]
        
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        
        return image

    '''
    This function computes the probality [0-100] of a pixel belongs to a 
    high density region
    '''

    def identifyHighDensity(self, foreground, radius):
        height = foreground.shape[0]
        width = foreground.shape[1]
        
        maxDensity = pow(radius * 2, 2) 
       
        output = copy.deepcopy(foreground)
        for h in range(0, height):
            for w in range(0, width):
                if foreground[h, w] != 0:  # 255
                    count = 0
                    for i in range(h - radius, h + radius):
                        for j in range(w - radius, w + radius):
                            try:  # out of bound, improve it... 
                                if foreground[i, j] > 0:  # <255
                                    count = count + 1
                                    # print count
                            except:
                                pass  # do nothing
                    
                    # if (count * 100) / maxDensity > 100:
                        # print (count * 100) / maxDensity
                    output[h, w] = ((count * 100) / maxDensity)  # 255- ...
                           
        return output
   
    '''
    This funciton extracts the high density regions
    @param densityImage: output of the identifyHighDensity function
    @param threhold: simple threshold value to separate the regions of interest  
    '''

    def extractHightDensityRegions(self, densityImage, threshold):

	    ret, hight = cv2.threshold(densityImage, threshold, 255, cv2.THRESH_BINARY)
	    ret, low = cv2.threshold(densityImage, threhold, 255, cv2.THRESH_BINARY_INV)
	    return hight  

    '''
    Simple plot of components, background is setup as white 
    '''

    def plotComponents(self):
        components = np.ma.masked_where(self.components[1] == 0, self.components[1])
        cmap = plt.get_cmap('hot')
        cmap.set_bad('white')
        plt.imshow(components, cmap=cmap)
        plt.show()
        
    def writeComponentsAsImages(self, mainDir, imageName):
               
        # create dir
        dirName = mainDir + "/" + imageName + "_" + str(ctime())
        try:
            os.mkdir(dirName)
        except FileExistsError:
            print("Directory " , dirName , " error creating the output directory")
               
        fileName = dirName + "/" + imageName
        
        # for i in range(1, len(self.roiMasks)):
            # cv2.imwrite(fileName + "_" + str(i) + ".png", self.roiMasks[i])
       
        cv2.imwrite(fileName + "_low" + ".png", self.roi)
        cv2.imwrite(fileName + "_high" + ".png", self.nonRoi)
    
    '''
    Find convex hull and fill it white
    low epsilon to more convex hull
    '''

    def findConvexHull(self, image):
        imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, threshed_img = cv2.threshold(imageGray, 0, 1, cv2.THRESH_BINARY)  # orignal 1 
        img, contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        for cnt in contours:
            epsilon = 0.0001 * cv2.arcLength(cnt, True)  # 0.0001
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(approx)
            cv2.drawContours(img, [hull], -1, (255, 255, 255), -1)
        return img   
    

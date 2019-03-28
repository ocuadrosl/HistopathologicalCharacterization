import sys
import threading
from HighDensityActions import *

#from PyQt4.QtGui import (QMainWindow, QApplication, QFileDialog, QMessageBox,  QPixmap, QGraphicsScene, QImage)

#from PyQt4.QtCore import SIGNAL, QRectF




from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)



from MainGui import Ui_MainWindow
from FirstLevel import * 




class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    
    
    resultSubwindows = []
   
    
    mainImageSubWindow=None
    
    moduleFrame = None
    
    vsiFileName = None
        
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)                                         
        self.setupUi(self)
        
        #bacis definitions
        
        
       
       
        #Action connections
        self.actionOpen.triggered.connect(self.openImage)
        self.actionHigh_density.triggered.connect(self.highDensityAction)
        self.resizeEvent = self.rezize
    
    
        
    def rezize(self, args):
        del args
        #update modulesFrame size
        if self.moduleFrame is not None:
            self.moduleFrame.resize(self.modulesFrame.width(), self.modulesFrame.height())      
   
    def highDensityAction(self):
        
        if self.moduleFrame is None: 
            self.moduleFrame = HighDensity(self.modulesFrame)
            self.moduleFrame.setMainWindow(self)
            self.moduleFrame.resize(self.modulesFrame.width(), self.modulesFrame.height())    
            self.moduleFrame.show()
                
    
    def showResultSubWindows(self, images):
        
        graphicsViews=[]
        pixmaps = []
        scenes = []
        qImages = []
        
        for i in range(0, len(images)):
            self.resultSubwindows.append(QtGui.QMdiSubWindow())

            self.resultSubwindows[i].setObjectName(_fromUtf8("subWindow_"+str(i)))

            height, width = images[i].shape 
           
            qImages.append(QtGui.QImage(images[i].data, width, height, QtGui.QImage.Format_Indexed8))
            pixmaps.append(QtGui.QPixmap(qImages[i]))
                   
            graphicsViews.append(QtGui.QGraphicsView(self.resultSubwindows[i]))
            self.mdiArea.addSubWindow(self.resultSubwindows[i])
            graphicsViews[i].setGeometry(QtCore.QRect(30, 30, width+10, height+10))
            graphicsViews[i].setObjectName(_fromUtf8("graphicView_"+str(i)))
            
            scenes.append(QtGui.QGraphicsScene()) 
            scenes[i].setSceneRect(QtCore.QRectF(pixmaps[i].rect()))
            scenes[i].addPixmap(pixmaps[i])
            graphicsViews[i].setScene(scenes[i])
            scenes[i].update()
            
            self.resultSubwindows[i].resize(width+60,height+60)
            self.resultSubwindows[i].setWindowTitle(str(i))
            self.resultSubwindows[i].show()
        
       
        
            
    
    def openImage(self):
       
        dialog  =  QtGui.QFileDialog(self)
        dialog.setNameFilter("Images (*.png *.vsi *.jpg)")
        if dialog.exec_():
            self.vsiFileName = str(dialog.selectedFiles()[0])
            
               
        #open thumbnail
        image = QtGui.QImage(self.vsiFileName)
       
        
        pixmap =  QtGui.QPixmap(image) 
        
        scene =  QtGui.QGraphicsScene() 
        scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        scene.addPixmap(pixmap)
        
        
        if self.mainImageSubWindow is None:
            self.mainImageSubWindow = QtGui.QMdiSubWindow()
            self.mainImageSubWindow.setObjectName(_fromUtf8("Thumbnail"))
            self.graphicsView = QtGui.QGraphicsView(self.mainImageSubWindow)
            self.mdiArea.addSubWindow(self.mainImageSubWindow)
       
                     
        self.graphicsView.setGeometry(QtCore.QRect(30, 30, image.width()+10, image.height()+10))
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        
        
        
        self.graphicsView.setScene(scene)
        scene.update()
        
        
        self.mainImageSubWindow.resize(image.width()+60, image.height()+60)
        self.mainImageSubWindow.setWindowTitle("VSI Thumbnail")
        self.mainImageSubWindow.show()
        
       
        
               
            
        '''
        msg = QMessageBox()
        msg.setInformativeText(fileNames[0])
        msg.exec_()
        ''' 

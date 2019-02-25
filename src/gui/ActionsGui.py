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
from FirstLevel import FirstLevel 




class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    
    
    subwindow = None
        
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)                                         
        self.setupUi(self)
        
        #bacis definitions
        
        self.vsiFileName = ""
       
       
        #Action connections
        self.actionOpen.triggered.connect(self.openImage)
        self.actionHigh_density.triggered.connect(self.highDensityAction)
        
        
        
   
    def highDensityAction(self):
        
        frame = HighDensity()
        frame.setMainWindow(self)
        frame.fileName = self.vsiFileName
           
        self.scrollArea.setWidget(frame)
        frame.show()
       
               
        
    
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
        
        
        if self.subwindow is None:
            self.subwindow = QtGui.QMdiSubWindow()
            self.subwindow.setObjectName(_fromUtf8("Thumbnail"))
            self.graphicsView = QtGui.QGraphicsView(self.subwindow)
            self.mdiArea.addSubWindow(self.subwindow)
       
                     
        self.graphicsView.setGeometry(QtCore.QRect(30, 30, image.width()+10, image.height()+10))
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        
        
        self.graphicsView.setScene(scene)
        scene.update()
        
        
        self.subwindow.resize(image.width()+60, image.height()+60)
        self.subwindow.setWindowTitle("VSI Thumbnail")
        self.subwindow.show()
        
       
        
               
            
        '''
        msg = QMessageBox()
        msg.setInformativeText(fileNames[0])
        msg.exec_()
        ''' 

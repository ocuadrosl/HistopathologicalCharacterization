from Highdensity import *
from FirstLevel import FirstLevel
from PyQt4 import QtCore, QtGui
import threading
from ActionsGui import *

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




class HighDensity(QtGui.QWidget, Ui_highDensity):
    
        
    mainWindow = None
    def __init__(self, parent=None):
        super(HighDensity, self).__init__(parent)                                         
        self.setupUi(self)
        
        
        self.firstLevel = FirstLevel()
        
        self.processButton.clicked.connect(self.process)
        
    def setMainWindow(self, mainWindow):
        self.mainWindow = mainWindow
    
    def process(self):
        
        thread = threading.Thread(target=self.firstLevel.identifyHighDensityLargeSample, args=(self.mainWindow.vsiFileName,7, 0.05, 9,9, 60))
        thread.daemon = True                            
        thread.start()
        
        
        
        #high, low, density, gray = self.firstLevel.identifyHighDensityLargeSample(self.fileName, 7, 0.05, 9,9, 60)     
        
        
        
    
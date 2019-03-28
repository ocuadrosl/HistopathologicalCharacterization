from Highdensity import *
from FirstLevel import *
from PyQt4 import QtCore, QtGui
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


class HighDensity(QtGui.QFrame, Ui_highDensity):
        
    mainWindow = None

    def __init__(self, parent=None):
        super(HighDensity, self).__init__(parent)                                         
        self.setupUi(self)
               
        self.firstLevel = FirstLevel()
        
        self.processPushButton.clicked.connect(self.process)
        self.clearTextPushButton.clicked.connect(self.clearMessages)
        
    def setMainWindow(self, mainWindow):
        self.mainWindow = mainWindow
    
    def clearMessages(self):
        self.messagesPlainTextEdit.clear()
      
    def process(self):
        if self.mainWindow.vsiFileName is None:
            self.messagesPlainTextEdit.setPlainText("Open a VSI file")

            return
        
        self.messagesPlainTextEdit.setPlainText("Processing, please wait")
        radius = self.radiusSpinBox.value()
        magnification = self.magnificationDoubleSpinBox.value()
        tilesY = self.tilesYSpinBox.value()
        tilesX = self.tilesXSpinBox.value()
        threshold = self.thresholdSpinBox.value()
               
        high, low, density, gray = self.firstLevel.identifyHighDensityLargeSample(self.mainWindow.vsiFileName, radius, magnification, tilesX, tilesY, threshold, self.progressBar)
        
        self.messagesPlainTextEdit.setPlainText("Succes")
        
        self.mainWindow.showResultSubWindows((high, low, density, gray))
        
        self.progressBar.setValue(0)
        
        
        
        '''
        thread = threading.Thread(target=self.threadAdmin)
        thread.daemon = True                            
        thread.start()   
        '''
        
        # high, low, density, gray = self.firstLevel.identifyHighDensityLargeSample(self.fileName, 7, 0.05, 9,9, 60)     
        
    

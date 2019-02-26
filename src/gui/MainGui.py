# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1227, 777)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.mdiArea = QtGui.QMdiArea(self.centralwidget)
        self.mdiArea.setGeometry(QtCore.QRect(220, 40, 911, 551))
        self.mdiArea.setObjectName(_fromUtf8("mdiArea"))
        self.scrollArea = QtGui.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(10, 40, 201, 581))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 199, 579))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(210, 630, 921, 80))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1227, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menumenu = QtGui.QMenu(self.menubar)
        self.menumenu.setObjectName(_fromUtf8("menumenu"))
        self.menuModules = QtGui.QMenu(self.menubar)
        self.menuModules.setObjectName(_fromUtf8("menuModules"))
        MainWindow.setMenuBar(self.menubar)
        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionHigh_density = QtGui.QAction(MainWindow)
        self.actionHigh_density.setObjectName(_fromUtf8("actionHigh_density"))
        self.menumenu.addSeparator()
        self.menumenu.addAction(self.actionOpen)
        self.menumenu.addAction(self.actionSave)
        self.menuModules.addAction(self.actionHigh_density)
        self.menubar.addAction(self.menumenu.menuAction())
        self.menubar.addAction(self.menuModules.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.menumenu.setTitle(_translate("MainWindow", "File", None))
        self.menuModules.setTitle(_translate("MainWindow", "Modules", None))
        self.actionOpen.setText(_translate("MainWindow", "Open", None))
        self.actionSave.setText(_translate("MainWindow", "Save", None))
        self.actionHigh_density.setText(_translate("MainWindow", "High density", None))


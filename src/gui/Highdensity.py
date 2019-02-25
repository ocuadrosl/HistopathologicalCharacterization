# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'highdensity.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
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

class Ui_highDensity(object):
    def setupUi(self, highDensity):
        highDensity.setObjectName(_fromUtf8("highDensity"))
        highDensity.resize(191, 608)
        self.verticalLayoutWidget = QtGui.QWidget(highDensity)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 160, 591))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.lineEdit = QtGui.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.verticalLayout.addWidget(self.lineEdit)
        self.processButton = QtGui.QPushButton(self.verticalLayoutWidget)
        self.processButton.setObjectName(_fromUtf8("processButton"))
        self.verticalLayout.addWidget(self.processButton)

        self.retranslateUi(highDensity)
        QtCore.QMetaObject.connectSlotsByName(highDensity)

    def retranslateUi(self, highDensity):
        highDensity.setWindowTitle(_translate("highDensity", "Form", None))
        self.processButton.setText(_translate("highDensity", "Process", None))


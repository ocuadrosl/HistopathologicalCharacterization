# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'highdensity.ui'
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

class Ui_highDensity(object):
    def setupUi(self, highDensity):
        highDensity.setObjectName(_fromUtf8("highDensity"))
        highDensity.resize(213, 632)
        self.groupBox = QtGui.QGroupBox(highDensity)
        self.groupBox.setGeometry(QtCore.QRect(10, 400, 191, 221))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.textBrowser = QtGui.QTextBrowser(self.groupBox)
        self.textBrowser.setGeometry(QtCore.QRect(10, 30, 171, 181))
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))
        self.groupBox_2 = QtGui.QGroupBox(highDensity)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 0, 191, 221))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.formLayoutWidget = QtGui.QWidget(self.groupBox_2)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 30, 171, 181))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label_4 = QtGui.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_4)
        self.radiusSpinBox = QtGui.QSpinBox(self.formLayoutWidget)
        self.radiusSpinBox.setMinimum(1)
        self.radiusSpinBox.setMaximum(50)
        self.radiusSpinBox.setProperty("value", 5)
        self.radiusSpinBox.setObjectName(_fromUtf8("radiusSpinBox"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.radiusSpinBox)
        self.label = QtGui.QLabel(self.formLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label)
        self.tilesXSpinBox = QtGui.QSpinBox(self.formLayoutWidget)
        self.tilesXSpinBox.setMinimum(1)
        self.tilesXSpinBox.setMaximum(100)
        self.tilesXSpinBox.setProperty("value", 10)
        self.tilesXSpinBox.setObjectName(_fromUtf8("tilesXSpinBox"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.tilesXSpinBox)
        self.label_5 = QtGui.QLabel(self.formLayoutWidget)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_5)
        self.tilesYSpinBox = QtGui.QSpinBox(self.formLayoutWidget)
        self.tilesYSpinBox.setMinimum(1)
        self.tilesYSpinBox.setMaximum(100)
        self.tilesYSpinBox.setProperty("value", 10)
        self.tilesYSpinBox.setObjectName(_fromUtf8("tilesYSpinBox"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.tilesYSpinBox)
        self.label_6 = QtGui.QLabel(self.formLayoutWidget)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_6)
        self.thresholdSpinBox = QtGui.QSpinBox(self.formLayoutWidget)
        self.thresholdSpinBox.setMaximum(100)
        self.thresholdSpinBox.setProperty("value", 5)
        self.thresholdSpinBox.setObjectName(_fromUtf8("thresholdSpinBox"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.thresholdSpinBox)
        self.label_7 = QtGui.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_7)
        self.magnificationDoubleSpinBox = QtGui.QDoubleSpinBox(self.formLayoutWidget)
        self.magnificationDoubleSpinBox.setMaximum(100.0)
        self.magnificationDoubleSpinBox.setSingleStep(0.5)
        self.magnificationDoubleSpinBox.setProperty("value", 5.0)
        self.magnificationDoubleSpinBox.setObjectName(_fromUtf8("magnificationDoubleSpinBox"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.magnificationDoubleSpinBox)
        self.groupBox_3 = QtGui.QGroupBox(highDensity)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 230, 191, 161))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.verticalLayoutWidget = QtGui.QWidget(self.groupBox_3)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 30, 171, 111))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.processPushButton = QtGui.QPushButton(self.verticalLayoutWidget)
        self.processPushButton.setObjectName(_fromUtf8("processPushButton"))
        self.verticalLayout.addWidget(self.processPushButton)
        self.progressBar = QtGui.QProgressBar(self.verticalLayoutWidget)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout.addWidget(self.progressBar)

        self.retranslateUi(highDensity)
        QtCore.QMetaObject.connectSlotsByName(highDensity)

    def retranslateUi(self, highDensity):
        highDensity.setWindowTitle(_translate("highDensity", "Form", None))
        self.groupBox.setTitle(_translate("highDensity", "Messages", None))
        self.groupBox_2.setTitle(_translate("highDensity", "Parameters", None))
        self.label_4.setText(_translate("highDensity", "Radius", None))
        self.label.setText(_translate("highDensity", "Tiles X", None))
        self.label_5.setText(_translate("highDensity", "Tiles Y", None))
        self.label_6.setText(_translate("highDensity", "Threshold", None))
        self.label_7.setText(_translate("highDensity", "Magnification", None))
        self.groupBox_3.setTitle(_translate("highDensity", "Actions", None))
        self.processPushButton.setText(_translate("highDensity", "Process", None))


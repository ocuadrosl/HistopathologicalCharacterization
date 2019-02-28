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
        highDensity.resize(400, 600)
        self.gridLayout_2 = QtGui.QGridLayout(highDensity)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.scrollArea = QtGui.QScrollArea(highDensity)
        self.scrollArea.setFrameShape(QtGui.QFrame.NoFrame)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 382, 559))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.gridLayout_3 = QtGui.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_3.setMargin(0)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.groupBox_2 = QtGui.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_6 = QtGui.QGridLayout(self.groupBox_2)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label_4 = QtGui.QLabel(self.groupBox_2)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_4)
        self.radiusSpinBox = QtGui.QSpinBox(self.groupBox_2)
        self.radiusSpinBox.setMinimum(1)
        self.radiusSpinBox.setMaximum(50)
        self.radiusSpinBox.setProperty("value", 5)
        self.radiusSpinBox.setObjectName(_fromUtf8("radiusSpinBox"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.radiusSpinBox)
        self.label = QtGui.QLabel(self.groupBox_2)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label)
        self.tilesXSpinBox = QtGui.QSpinBox(self.groupBox_2)
        self.tilesXSpinBox.setMinimum(1)
        self.tilesXSpinBox.setMaximum(100)
        self.tilesXSpinBox.setProperty("value", 10)
        self.tilesXSpinBox.setObjectName(_fromUtf8("tilesXSpinBox"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.tilesXSpinBox)
        self.label_5 = QtGui.QLabel(self.groupBox_2)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_5)
        self.tilesYSpinBox = QtGui.QSpinBox(self.groupBox_2)
        self.tilesYSpinBox.setMinimum(1)
        self.tilesYSpinBox.setMaximum(100)
        self.tilesYSpinBox.setProperty("value", 10)
        self.tilesYSpinBox.setObjectName(_fromUtf8("tilesYSpinBox"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.tilesYSpinBox)
        self.label_6 = QtGui.QLabel(self.groupBox_2)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_6)
        self.thresholdSpinBox = QtGui.QSpinBox(self.groupBox_2)
        self.thresholdSpinBox.setMaximum(100)
        self.thresholdSpinBox.setProperty("value", 60)
        self.thresholdSpinBox.setObjectName(_fromUtf8("thresholdSpinBox"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.thresholdSpinBox)
        self.label_7 = QtGui.QLabel(self.groupBox_2)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_7)
        self.magnificationDoubleSpinBox = QtGui.QDoubleSpinBox(self.groupBox_2)
        self.magnificationDoubleSpinBox.setMaximum(100.0)
        self.magnificationDoubleSpinBox.setSingleStep(0.5)
        self.magnificationDoubleSpinBox.setProperty("value", 5.0)
        self.magnificationDoubleSpinBox.setObjectName(_fromUtf8("magnificationDoubleSpinBox"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.magnificationDoubleSpinBox)
        self.gridLayout_6.addLayout(self.formLayout, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.groupBox_3 = QtGui.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.gridLayout_5 = QtGui.QGridLayout(self.groupBox_3)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.processPushButton = QtGui.QPushButton(self.groupBox_3)
        self.processPushButton.setEnabled(True)
        self.processPushButton.setObjectName(_fromUtf8("processPushButton"))
        self.verticalLayout.addWidget(self.processPushButton)
        self.progressBar = QtGui.QProgressBar(self.groupBox_3)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout.addWidget(self.progressBar)
        self.gridLayout_5.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_3, 1, 0, 1, 1)
        self.groupBox = QtGui.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout_4 = QtGui.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.messagesPlainTextEdit = QtGui.QPlainTextEdit(self.groupBox)
        self.messagesPlainTextEdit.setReadOnly(True)
        self.messagesPlainTextEdit.setOverwriteMode(True)
        self.messagesPlainTextEdit.setObjectName(_fromUtf8("messagesPlainTextEdit"))
        self.gridLayout_4.addWidget(self.messagesPlainTextEdit, 0, 0, 1, 1)
        self.clearTextPushButton = QtGui.QPushButton(self.groupBox)
        self.clearTextPushButton.setObjectName(_fromUtf8("clearTextPushButton"))
        self.gridLayout_4.addWidget(self.clearTextPushButton, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 2, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_2.addWidget(self.scrollArea, 1, 0, 1, 1)
        self.label_2 = QtGui.QLabel(highDensity)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)

        self.retranslateUi(highDensity)
        QtCore.QMetaObject.connectSlotsByName(highDensity)

    def retranslateUi(self, highDensity):
        highDensity.setWindowTitle(_translate("highDensity", "Form", None))
        self.groupBox_2.setTitle(_translate("highDensity", "Parameters", None))
        self.label_4.setText(_translate("highDensity", "Radius", None))
        self.label.setText(_translate("highDensity", "Tiles X", None))
        self.label_5.setText(_translate("highDensity", "Tiles Y", None))
        self.label_6.setText(_translate("highDensity", "Threshold", None))
        self.label_7.setText(_translate("highDensity", "Magnification", None))
        self.groupBox_3.setTitle(_translate("highDensity", "Actions", None))
        self.processPushButton.setText(_translate("highDensity", "Process", None))
        self.groupBox.setTitle(_translate("highDensity", "Messages", None))
        self.clearTextPushButton.setText(_translate("highDensity", "Clear messages", None))
        self.label_2.setText(_translate("highDensity", "High Density", None))


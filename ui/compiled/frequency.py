# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'frequency.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(280, 170)
        mainwindow.setMinimumSize(QSize(280, 170))
        mainwindow.setMaximumSize(QSize(280, 170))
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.title = QLabel(mainwindow)
        self.title.setObjectName(u"title")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title.sizePolicy().hasHeightForWidth())
        self.title.setSizePolicy(sizePolicy)
        self.title.setMinimumSize(QSize(0, 40))
        self.title.setMaximumSize(QSize(16777215, 40))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.title.setFont(font)
        self.title.setStyleSheet(u"border: 1px solid grey;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.title)

        self.line_2 = QFrame(mainwindow)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShadow(QFrame.Shadow.Plain)
        self.line_2.setFrameShape(QFrame.Shape.HLine)

        self.verticalLayout.addWidget(self.line_2)

        self.freq_box = QWidget(mainwindow)
        self.freq_box.setObjectName(u"freq_box")
        self.input_box = QHBoxLayout(self.freq_box)
        self.input_box.setSpacing(20)
        self.input_box.setObjectName(u"input_box")
        self.freq_lab = QLabel(self.freq_box)
        self.freq_lab.setObjectName(u"freq_lab")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.freq_lab.setFont(font1)

        self.input_box.addWidget(self.freq_lab)

        self.freq_edit = QLineEdit(self.freq_box)
        self.freq_edit.setObjectName(u"freq_edit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.freq_edit.sizePolicy().hasHeightForWidth())
        self.freq_edit.setSizePolicy(sizePolicy1)
        self.freq_edit.setMinimumSize(QSize(0, 0))
        self.freq_edit.setMaximumSize(QSize(100, 16777215))

        self.input_box.addWidget(self.freq_edit)


        self.verticalLayout.addWidget(self.freq_box)

        self.line = QFrame(mainwindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShadow(QFrame.Shadow.Plain)
        self.line.setFrameShape(QFrame.Shape.HLine)

        self.verticalLayout.addWidget(self.line)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.ok = QPushButton(mainwindow)
        self.ok.setObjectName(u"ok")
        sizePolicy1.setHeightForWidth(self.ok.sizePolicy().hasHeightForWidth())
        self.ok.setSizePolicy(sizePolicy1)
        self.ok.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.horizontalLayout_3.addWidget(self.ok)


        self.verticalLayout.addLayout(self.horizontalLayout_3)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Mimimal Frequency", None))
        self.title.setText(QCoreApplication.translate("mainwindow", u"Minimal Frequency", None))
        self.freq_lab.setText(QCoreApplication.translate("mainwindow", u"Minimal Frequency :", None))
        self.freq_edit.setText("")
        self.freq_edit.setPlaceholderText(QCoreApplication.translate("mainwindow", u"3", None))
        self.ok.setText(QCoreApplication.translate("mainwindow", u"Ok", None))
    # retranslateUi


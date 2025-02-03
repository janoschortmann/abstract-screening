# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'stem.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDialogButtonBox,
    QFrame, QLabel, QListView, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(340, 320)
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.stem_lab = QLabel(mainwindow)
        self.stem_lab.setObjectName(u"stem_lab")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stem_lab.sizePolicy().hasHeightForWidth())
        self.stem_lab.setSizePolicy(sizePolicy)
        self.stem_lab.setMinimumSize(QSize(0, 40))
        self.stem_lab.setMaximumSize(QSize(16777215, 40))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.stem_lab.setFont(font)
        self.stem_lab.setStyleSheet(u"border: 1px solid grey;")
        self.stem_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.stem_lab)

        self.stem_list = QListView(mainwindow)
        self.stem_list.setObjectName(u"stem_list")

        self.verticalLayout.addWidget(self.stem_list)

        self.line = QFrame(mainwindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.button_box = QDialogButtonBox(mainwindow)
        self.button_box.setObjectName(u"button_box")
        self.button_box.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout.addWidget(self.button_box)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Keywords", None))
        self.stem_lab.setText(QCoreApplication.translate("mainwindow", u"Stem Found", None))
    # retranslateUi


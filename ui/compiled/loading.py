# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'loading.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QProgressBar,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(300, 130)
        mainwindow.setMinimumSize(QSize(300, 130))
        mainwindow.setMaximumSize(QSize(300, 130))
        mainwindow.setStyleSheet(u"b")
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.loading_lab = QLabel(mainwindow)
        self.loading_lab.setObjectName(u"loading_lab")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loading_lab.sizePolicy().hasHeightForWidth())
        self.loading_lab.setSizePolicy(sizePolicy)
        self.loading_lab.setMinimumSize(QSize(0, 40))
        self.loading_lab.setMaximumSize(QSize(16777215, 40))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.loading_lab.setFont(font)
        self.loading_lab.setStyleSheet(u"border: 1px solid grey;")
        self.loading_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.loading_lab)

        self.bar = QProgressBar(mainwindow)
        self.bar.setObjectName(u"bar")
        self.bar.setValue(0)

        self.verticalLayout.addWidget(self.bar)

        self.process = QLabel(mainwindow)
        self.process.setObjectName(u"process")
        self.process.setMinimumSize(QSize(0, 20))
        self.process.setMaximumSize(QSize(16777215, 20))
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(8)
        font1.setItalic(True)
        self.process.setFont(font1)
        self.process.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.verticalLayout.addWidget(self.process)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Loading", None))
        mainwindow.setWindowFilePath("")
        self.loading_lab.setText(QCoreApplication.translate("mainwindow", u"Stemming the files", None))
        self.process.setText(QCoreApplication.translate("mainwindow", u"Starting", None))
    # retranslateUi


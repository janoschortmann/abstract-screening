# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'paper.ui'
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
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
    QTextBrowser, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(340, 320)
        self.verticalLayout = QVBoxLayout(mainwindow)
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
        font.setItalic(False)
        self.title.setFont(font)
        self.title.setStyleSheet(u"border: 1px solid grey;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.title)

        self.abstract = QTextBrowser(mainwindow)
        self.abstract.setObjectName(u"abstract")

        self.verticalLayout.addWidget(self.abstract)

        self.info = QHBoxLayout()
        self.info.setObjectName(u"info")
        self.info.setContentsMargins(-1, 5, -1, -1)
        self.doi = QVBoxLayout()
        self.doi.setObjectName(u"doi")
        self.doi_lab = QLabel(mainwindow)
        self.doi_lab.setObjectName(u"doi_lab")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.doi_lab.setFont(font1)
        self.doi_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.doi.addWidget(self.doi_lab)

        self.line = QFrame(mainwindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.doi.addWidget(self.line)

        self.doi_inp = QLabel(mainwindow)
        self.doi_inp.setObjectName(u"doi_inp")
        self.doi_inp.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.doi.addWidget(self.doi_inp)


        self.info.addLayout(self.doi)

        self.jour = QVBoxLayout()
        self.jour.setObjectName(u"jour")
        self.jour_lab = QLabel(mainwindow)
        self.jour_lab.setObjectName(u"jour_lab")
        self.jour_lab.setFont(font1)
        self.jour_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.jour.addWidget(self.jour_lab)

        self.line_2 = QFrame(mainwindow)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.jour.addWidget(self.line_2)

        self.jour_inp = QLabel(mainwindow)
        self.jour_inp.setObjectName(u"jour_inp")
        self.jour_inp.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.jour.addWidget(self.jour_inp)


        self.info.addLayout(self.jour)

        self.date = QVBoxLayout()
        self.date.setObjectName(u"date")
        self.date_lab = QLabel(mainwindow)
        self.date_lab.setObjectName(u"date_lab")
        self.date_lab.setFont(font1)
        self.date_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.date.addWidget(self.date_lab)

        self.line_3 = QFrame(mainwindow)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.date.addWidget(self.line_3)

        self.date_inp = QLabel(mainwindow)
        self.date_inp.setObjectName(u"date_inp")
        self.date_inp.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.date.addWidget(self.date_inp)


        self.info.addLayout(self.date)


        self.verticalLayout.addLayout(self.info)

        self.buttons = QHBoxLayout()
        self.buttons.setObjectName(u"buttons")
        self.spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.buttons.addItem(self.spacer)

        self.yes_but = QPushButton(mainwindow)
        self.yes_but.setObjectName(u"yes_but")
        self.yes_but.setMinimumSize(QSize(100, 25))
        self.yes_but.setMaximumSize(QSize(100, 25))
        self.yes_but.setStyleSheet(u"background-color: rgb(155, 255, 155);")

        self.buttons.addWidget(self.yes_but)

        self.no_but = QPushButton(mainwindow)
        self.no_but.setObjectName(u"no_but")
        self.no_but.setMinimumSize(QSize(100, 25))
        self.no_but.setMaximumSize(QSize(100, 25))
        self.no_but.setStyleSheet(u"background-color: rgb(255, 90, 90);")

        self.buttons.addWidget(self.no_but)

        self.cancel_but = QPushButton(mainwindow)
        self.cancel_but.setObjectName(u"cancel_but")
        self.cancel_but.setMinimumSize(QSize(100, 25))
        self.cancel_but.setMaximumSize(QSize(100, 25))

        self.buttons.addWidget(self.cancel_but)


        self.verticalLayout.addLayout(self.buttons)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Paper Information", None))
        self.title.setText(QCoreApplication.translate("mainwindow", u"Title", None))
        self.doi_lab.setText(QCoreApplication.translate("mainwindow", u"DOI", None))
        self.doi_inp.setText(QCoreApplication.translate("mainwindow", u"__________", None))
        self.jour_lab.setText(QCoreApplication.translate("mainwindow", u"Journal", None))
        self.jour_inp.setText(QCoreApplication.translate("mainwindow", u"__________", None))
        self.date_lab.setText(QCoreApplication.translate("mainwindow", u"Date", None))
        self.date_inp.setText(QCoreApplication.translate("mainwindow", u"__________", None))
        self.yes_but.setText(QCoreApplication.translate("mainwindow", u"Yes", None))
        self.no_but.setText(QCoreApplication.translate("mainwindow", u"No", None))
        self.cancel_but.setText(QCoreApplication.translate("mainwindow", u"Cancel", None))
    # retranslateUi


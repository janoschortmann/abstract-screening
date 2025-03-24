# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLayout,
    QMainWindow, QPushButton, QSizePolicy, QSpacerItem,
    QSplitter, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(700, 500)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(mainwindow.sizePolicy().hasHeightForWidth())
        mainwindow.setSizePolicy(sizePolicy)
        mainwindow.setStyleSheet(u".QPushButton:hover {\n"
"    background-color: #64b5f6;\n"
"    color: #fff;\n"
"}\n"
"\n"
".QPushButton:pressed {\n"
"    background-color: #bbdefb;\n"
"}\n"
"\n"
".QMainWindow {\n"
"    background-color: black;\n"
"}\n"
"\n"
".QSplitter::handle::vertical {\n"
"    background-color: #19232d;\n"
"    border-radius 4px;\n"
"    padding: 1px 1px 1px 1px;\n"
"}\n"
"\n"
".QSplitter::handle::horizonal {\n"
"    background-color: #19232d;\n"
"    border-radius: 4px;\n"
"    padding: 1px 1px 1px 1px;\n"
"}\n"
"\n"
".QEditBox {\n"
"    background-color: grey;\n"
"}")
        self.centralwidget = QWidget(mainwindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.info = QWidget(self.centralwidget)
        self.info.setObjectName(u"info")
        self.info.setMaximumSize(QSize(16777215, 16777215))
        self.info.setStyleSheet(u"")
        self.top = QHBoxLayout(self.info)
        self.top.setSpacing(5)
        self.top.setObjectName(u"top")
        self.top.setContentsMargins(0, 0, 0, 0)
        self.params = QPushButton(self.info)
        self.params.setObjectName(u"params")
        self.params.setMinimumSize(QSize(80, 35))
        self.params.setStyleSheet(u"border: 1px solid grey;")
        self.params.setFlat(True)

        self.top.addWidget(self.params)

        self.data = QPushButton(self.info)
        self.data.setObjectName(u"data")
        self.data.setMinimumSize(QSize(80, 35))
        self.data.setStyleSheet(u"border: 1px solid grey;")
        self.data.setFlat(True)

        self.top.addWidget(self.data)

        self.about = QPushButton(self.info)
        self.about.setObjectName(u"about")
        self.about.setMinimumSize(QSize(80, 35))
        self.about.setStyleSheet(u"border: 1px solid grey;")
        self.about.setAutoDefault(False)
        self.about.setFlat(True)

        self.top.addWidget(self.about)

        self.filler = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.top.addItem(self.filler)

        self.other_but = QWidget(self.info)
        self.other_but.setObjectName(u"other_but")
        self.horizontalLayout = QHBoxLayout(self.other_but)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, -1, 0, -1)

        self.top.addWidget(self.other_but)


        self.verticalLayout.addWidget(self.info)

        self.body = QSplitter(self.centralwidget)
        self.body.setObjectName(u"body")
        sizePolicy.setHeightForWidth(self.body.sizePolicy().hasHeightForWidth())
        self.body.setSizePolicy(sizePolicy)
        self.body.setOrientation(Qt.Orientation.Vertical)
        self.bottom = QSplitter(self.body)
        self.bottom.setObjectName(u"bottom")
        sizePolicy.setHeightForWidth(self.bottom.sizePolicy().hasHeightForWidth())
        self.bottom.setSizePolicy(sizePolicy)
        self.bottom.setMaximumSize(QSize(16777215, 16777215))
        self.bottom.setOrientation(Qt.Orientation.Horizontal)
        self.steps = QWidget(self.bottom)
        self.steps.setObjectName(u"steps")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.steps.sizePolicy().hasHeightForWidth())
        self.steps.setSizePolicy(sizePolicy1)
        self.steps.setMinimumSize(QSize(120, 120))
        self.steps.setMaximumSize(QSize(16777215, 16777215))
        self.steps.setAutoFillBackground(False)
        self.steps.setStyleSheet(u"border: 1px solid grey;")
        self.verticalLayout_3 = QVBoxLayout(self.steps)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.verticalLayout_3.setContentsMargins(9, 9, -1, -1)
        self.step_lab = QLabel(self.steps)
        self.step_lab.setObjectName(u"step_lab")
        self.step_lab.setMaximumSize(QSize(16777215, 50))
        font = QFont()
        font.setFamilies([u"OCR A"])
        font.setPointSize(14)
        self.step_lab.setFont(font)
        self.step_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_3.addWidget(self.step_lab)

        self.next = QPushButton(self.steps)
        self.next.setObjectName(u"next")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.next.sizePolicy().hasHeightForWidth())
        self.next.setSizePolicy(sizePolicy2)
        self.next.setMinimumSize(QSize(100, 65))
        self.next.setMaximumSize(QSize(16777215, 16777215))
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(12)
        self.next.setFont(font1)
        self.next.setStyleSheet(u".QPushButton:hover {\n"
"    background-color: rgb(100, 100, 100);\n"
"    color: #fff;\n"
"}\n"
"\n"
".QPushButton:pressed {\n"
"	background-color: rgb(35, 35, 35);\n"
"}")

        self.verticalLayout_3.addWidget(self.next)

        self.bottom.addWidget(self.steps)
        self.body.addWidget(self.bottom)

        self.verticalLayout.addWidget(self.body)

        mainwindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainwindow)

        self.about.setDefault(False)


        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Comprehensive Abstsract Screening", None))
        self.params.setText(QCoreApplication.translate("mainwindow", u"Params", None))
        self.data.setText(QCoreApplication.translate("mainwindow", u"Data", None))
        self.about.setText(QCoreApplication.translate("mainwindow", u"About", None))
        self.step_lab.setText(QCoreApplication.translate("mainwindow", u"Step #1", None))
        self.next.setText(QCoreApplication.translate("mainwindow", u"Next", None))
    # retranslateUi


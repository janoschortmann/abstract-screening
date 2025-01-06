# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(700, 500)
        self.centralwidget = QWidget(mainwindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.info = QWidget(self.centralwidget)
        self.info.setObjectName(u"info")
        self.info.setMaximumSize(QSize(16777215, 16777215))
        self.info.setStyleSheet(u"border: 1px solid grey;")
        self.top = QHBoxLayout(self.info)
        self.top.setSpacing(0)
        self.top.setObjectName(u"top")
        self.top.setContentsMargins(0, 0, 0, 0)
        self.params = QPushButton(self.info)
        self.params.setObjectName(u"params")
        self.params.setMinimumSize(QSize(80, 35))
        self.params.setFlat(True)

        self.top.addWidget(self.params)

        self.data = QPushButton(self.info)
        self.data.setObjectName(u"data")
        self.data.setMinimumSize(QSize(80, 35))
        self.data.setFlat(True)

        self.top.addWidget(self.data)

        self.find = QPushButton(self.info)
        self.find.setObjectName(u"find")
        self.find.setMinimumSize(QSize(80, 35))
        self.find.setStyleSheet(u"border: 0px solid grey;")

        self.top.addWidget(self.find)

        self.about = QPushButton(self.info)
        self.about.setObjectName(u"about")
        self.about.setMinimumSize(QSize(80, 35))
        self.about.setAutoDefault(False)
        self.about.setFlat(True)

        self.top.addWidget(self.about)

        self.filler = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.top.addItem(self.filler)


        self.verticalLayout.addWidget(self.info)

        self.body = QSplitter(self.centralwidget)
        self.body.setObjectName(u"body")
        self.body.setOrientation(Qt.Orientation.Vertical)
        self.display = QWidget(self.body)
        self.display.setObjectName(u"display")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.display.sizePolicy().hasHeightForWidth())
        self.display.setSizePolicy(sizePolicy)
        self.body.addWidget(self.display)
        self.bottom = QSplitter(self.body)
        self.bottom.setObjectName(u"bottom")
        sizePolicy.setHeightForWidth(self.bottom.sizePolicy().hasHeightForWidth())
        self.bottom.setSizePolicy(sizePolicy)
        self.bottom.setMaximumSize(QSize(16777215, 111))
        self.bottom.setOrientation(Qt.Orientation.Horizontal)
        self.options = QWidget(self.bottom)
        self.options.setObjectName(u"options")
        self.bottom.addWidget(self.options)
        self.steps = QWidget(self.bottom)
        self.steps.setObjectName(u"steps")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.steps.sizePolicy().hasHeightForWidth())
        self.steps.setSizePolicy(sizePolicy1)
        self.steps.setAutoFillBackground(False)
        self.steps.setStyleSheet(u"border: 1px solid grey;")
        self.verticalLayout_3 = QVBoxLayout(self.steps)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
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
        self.find.setText(QCoreApplication.translate("mainwindow", u"Find", None))
        self.about.setText(QCoreApplication.translate("mainwindow", u"About", None))
        self.step_lab.setText(QCoreApplication.translate("mainwindow", u"Step #1", None))
        self.next.setText(QCoreApplication.translate("mainwindow", u"Next", None))
    # retranslateUi


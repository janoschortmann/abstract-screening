# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'params.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
    QLayout, QLineEdit, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(350, 230)
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.personal_lab = QLabel(mainwindow)
        self.personal_lab.setObjectName(u"personal_lab")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.personal_lab.sizePolicy().hasHeightForWidth())
        self.personal_lab.setSizePolicy(sizePolicy)
        self.personal_lab.setMinimumSize(QSize(300, 50))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.personal_lab.setFont(font)
        self.personal_lab.setStyleSheet(u"border: 1px solid grey;")
        self.personal_lab.setFrameShape(QFrame.Shape.NoFrame)
        self.personal_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.personal_lab)

        self.grid = QWidget(mainwindow)
        self.grid.setObjectName(u"grid")
        self.grid.setStyleSheet(u"border: 1px solid grey;")
        self.gridLayout = QGridLayout(self.grid)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.gridLayout.setHorizontalSpacing(10)
        self.token = QLabel(self.grid)
        self.token.setObjectName(u"token")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.token.setFont(font1)
        self.token.setStyleSheet(u"border: 0px solid black;")

        self.gridLayout.addWidget(self.token, 2, 0, 1, 1)

        self.query_limit = QLabel(self.grid)
        self.query_limit.setObjectName(u"query_limit")
        self.query_limit.setFont(font1)
        self.query_limit.setStyleSheet(u"border: 0px solid black;")

        self.gridLayout.addWidget(self.query_limit, 0, 0, 1, 1)

        self.col1 = QLabel(self.grid)
        self.col1.setObjectName(u"col1")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.col1.sizePolicy().hasHeightForWidth())
        self.col1.setSizePolicy(sizePolicy1)
        self.col1.setStyleSheet(u"border: 0px solid black;")
        self.col1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.col1, 2, 1, 1, 1)

        self.api_key = QLabel(self.grid)
        self.api_key.setObjectName(u"api_key")
        self.api_key.setFont(font1)
        self.api_key.setStyleSheet(u"border: 0px solid black;")

        self.gridLayout.addWidget(self.api_key, 1, 0, 1, 1)

        self.col2 = QLabel(self.grid)
        self.col2.setObjectName(u"col2")
        sizePolicy1.setHeightForWidth(self.col2.sizePolicy().hasHeightForWidth())
        self.col2.setSizePolicy(sizePolicy1)
        self.col2.setStyleSheet(u"border: 0px solid black;")
        self.col2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.col2, 1, 1, 1, 1)

        self.query_input = QLineEdit(self.grid)
        self.query_input.setObjectName(u"query_input")

        self.gridLayout.addWidget(self.query_input, 0, 2, 1, 1)

        self.api_input = QLineEdit(self.grid)
        self.api_input.setObjectName(u"api_input")

        self.gridLayout.addWidget(self.api_input, 1, 2, 1, 1)

        self.token_input = QLineEdit(self.grid)
        self.token_input.setObjectName(u"token_input")

        self.gridLayout.addWidget(self.token_input, 2, 2, 1, 1)

        self.col3 = QLabel(self.grid)
        self.col3.setObjectName(u"col3")
        sizePolicy1.setHeightForWidth(self.col3.sizePolicy().hasHeightForWidth())
        self.col3.setSizePolicy(sizePolicy1)
        self.col3.setStyleSheet(u"border: 0px solid black;")
        self.col3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.col3, 0, 1, 1, 1)

        self.gridLayout.setRowMinimumHeight(0, 30)
        self.gridLayout.setRowMinimumHeight(1, 30)
        self.gridLayout.setRowMinimumHeight(2, 30)

        self.verticalLayout.addWidget(self.grid)

        self.save_info = QPushButton(mainwindow)
        self.save_info.setObjectName(u"save_info")
        self.save_info.setFont(font)
        self.save_info.setStyleSheet(u"background-color: rgb(115, 255, 115)")

        self.verticalLayout.addWidget(self.save_info)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Query Info Window", None))
        self.personal_lab.setText(QCoreApplication.translate("mainwindow", u"Personal API Information", None))
        self.token.setText(QCoreApplication.translate("mainwindow", u"Token", None))
        self.query_limit.setText(QCoreApplication.translate("mainwindow", u"Query Limit", None))
        self.col1.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.api_key.setText(QCoreApplication.translate("mainwindow", u"API Key", None))
        self.col2.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.col3.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.save_info.setText(QCoreApplication.translate("mainwindow", u"Save", None))
    # retranslateUi


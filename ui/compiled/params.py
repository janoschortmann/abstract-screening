# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'params.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
    QLayout, QLineEdit, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(400, 310)
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
        self.col_5 = QLabel(self.grid)
        self.col_5.setObjectName(u"col_5")
        self.col_5.setStyleSheet(u"border: 0px solid grey;")

        self.gridLayout.addWidget(self.col_5, 4, 1, 1, 1)

        self.col_3 = QLabel(self.grid)
        self.col_3.setObjectName(u"col_3")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.col_3.sizePolicy().hasHeightForWidth())
        self.col_3.setSizePolicy(sizePolicy1)
        self.col_3.setStyleSheet(u"border: 0px solid black;")
        self.col_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.col_3, 0, 1, 1, 1)

        self.api_lab = QLabel(self.grid)
        self.api_lab.setObjectName(u"api_lab")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.api_lab.setFont(font1)
        self.api_lab.setStyleSheet(u"border: 0px solid black;")

        self.gridLayout.addWidget(self.api_lab, 1, 0, 1, 1)

        self.token_edit = QLineEdit(self.grid)
        self.token_edit.setObjectName(u"token_edit")

        self.gridLayout.addWidget(self.token_edit, 2, 2, 1, 1)

        self.thr_edit = QLineEdit(self.grid)
        self.thr_edit.setObjectName(u"thr_edit")

        self.gridLayout.addWidget(self.thr_edit, 3, 2, 1, 1)

        self.query_lab = QLabel(self.grid)
        self.query_lab.setObjectName(u"query_lab")
        self.query_lab.setFont(font1)
        self.query_lab.setStyleSheet(u"border: 0px solid black;")

        self.gridLayout.addWidget(self.query_lab, 0, 0, 1, 1)

        self.col_2 = QLabel(self.grid)
        self.col_2.setObjectName(u"col_2")
        sizePolicy1.setHeightForWidth(self.col_2.sizePolicy().hasHeightForWidth())
        self.col_2.setSizePolicy(sizePolicy1)
        self.col_2.setStyleSheet(u"border: 0px solid black;")
        self.col_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.col_2, 1, 1, 1, 1)

        self.query_edit = QLineEdit(self.grid)
        self.query_edit.setObjectName(u"query_edit")

        self.gridLayout.addWidget(self.query_edit, 0, 2, 1, 1)

        self.api_edit = QLineEdit(self.grid)
        self.api_edit.setObjectName(u"api_edit")

        self.gridLayout.addWidget(self.api_edit, 1, 2, 1, 1)

        self.thr_lab = QLabel(self.grid)
        self.thr_lab.setObjectName(u"thr_lab")
        self.thr_lab.setFont(font1)
        self.thr_lab.setStyleSheet(u"border: 0px solid grey;")

        self.gridLayout.addWidget(self.thr_lab, 3, 0, 1, 1)

        self.step_lab = QLabel(self.grid)
        self.step_lab.setObjectName(u"step_lab")
        self.step_lab.setFont(font1)
        self.step_lab.setStyleSheet(u"border: 0px solid grey;")

        self.gridLayout.addWidget(self.step_lab, 4, 0, 1, 1)

        self.col_4 = QLabel(self.grid)
        self.col_4.setObjectName(u"col_4")
        self.col_4.setStyleSheet(u"border: 0px solid grey;")

        self.gridLayout.addWidget(self.col_4, 3, 1, 1, 1)

        self.col_1 = QLabel(self.grid)
        self.col_1.setObjectName(u"col_1")
        sizePolicy1.setHeightForWidth(self.col_1.sizePolicy().hasHeightForWidth())
        self.col_1.setSizePolicy(sizePolicy1)
        self.col_1.setStyleSheet(u"border: 0px solid black;")
        self.col_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.col_1, 2, 1, 1, 1)

        self.token_lab = QLabel(self.grid)
        self.token_lab.setObjectName(u"token_lab")
        self.token_lab.setFont(font1)
        self.token_lab.setStyleSheet(u"border: 0px solid black;")

        self.gridLayout.addWidget(self.token_lab, 2, 0, 1, 1)

        self.step_edit = QLineEdit(self.grid)
        self.step_edit.setObjectName(u"step_edit")

        self.gridLayout.addWidget(self.step_edit, 4, 2, 1, 1)

        self.gridLayout.setRowMinimumHeight(0, 30)
        self.gridLayout.setRowMinimumHeight(1, 30)
        self.gridLayout.setRowMinimumHeight(2, 30)
        self.gridLayout.setRowMinimumHeight(3, 30)
        self.gridLayout.setRowMinimumHeight(4, 30)

        self.verticalLayout.addWidget(self.grid)

        self.save_info = QPushButton(mainwindow)
        self.save_info.setObjectName(u"save_info")
        self.save_info.setFont(font)
        self.save_info.setStyleSheet(u"background-color: rgb(115, 255, 115)")

        self.verticalLayout.addWidget(self.save_info)

        QWidget.setTabOrder(self.query_edit, self.api_edit)
        QWidget.setTabOrder(self.api_edit, self.token_edit)
        QWidget.setTabOrder(self.token_edit, self.thr_edit)
        QWidget.setTabOrder(self.thr_edit, self.step_edit)
        QWidget.setTabOrder(self.step_edit, self.save_info)

        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Query Info Window", None))
        self.personal_lab.setText(QCoreApplication.translate("mainwindow", u"Personal API Information", None))
        self.col_5.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.col_3.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.api_lab.setText(QCoreApplication.translate("mainwindow", u"API Key", None))
#if QT_CONFIG(tooltip)
        self.token_edit.setToolTip(QCoreApplication.translate("mainwindow", u"The token if multiple users use this computer for queries", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.thr_edit.setToolTip(QCoreApplication.translate("mainwindow", u"Exclude all papers with a probability below a threshold", None))
#endif // QT_CONFIG(tooltip)
        self.query_lab.setText(QCoreApplication.translate("mainwindow", u"Query Limit", None))
        self.col_2.setText(QCoreApplication.translate("mainwindow", u":", None))
#if QT_CONFIG(tooltip)
        self.query_edit.setToolTip(QCoreApplication.translate("mainwindow", u"The maximum number of papers queried in one search", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.api_edit.setToolTip(QCoreApplication.translate("mainwindow", u"The Scopus API key", None))
#endif // QT_CONFIG(tooltip)
        self.thr_lab.setText(QCoreApplication.translate("mainwindow", u"Threshold", None))
        self.step_lab.setText(QCoreApplication.translate("mainwindow", u"Step", None))
        self.col_4.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.col_1.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.token_lab.setText(QCoreApplication.translate("mainwindow", u"Token", None))
#if QT_CONFIG(tooltip)
        self.step_edit.setToolTip(QCoreApplication.translate("mainwindow", u"The probability step to list papers within a certain probability interval", None))
#endif // QT_CONFIG(tooltip)
        self.save_info.setText(QCoreApplication.translate("mainwindow", u"Save", None))
    # retranslateUi


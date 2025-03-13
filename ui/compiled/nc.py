# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'nc.ui'
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
    QGridLayout, QLabel, QSizePolicy, QVBoxLayout,
    QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(250, 200)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(mainwindow.sizePolicy().hasHeightForWidth())
        mainwindow.setSizePolicy(sizePolicy)
        mainwindow.setMinimumSize(QSize(250, 200))
        mainwindow.setMaximumSize(QSize(250, 200))
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.title = QLabel(mainwindow)
        self.title.setObjectName(u"title")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.title.sizePolicy().hasHeightForWidth())
        self.title.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.title.setFont(font)
        self.title.setStyleSheet(u"border: 1px solid grey;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setMargin(7)

        self.verticalLayout.addWidget(self.title)

        self.grid = QWidget(mainwindow)
        self.grid.setObjectName(u"grid")
        self.grid.setStyleSheet(u"border: 1px solid grey;")
        self.res_grid = QGridLayout(self.grid)
        self.res_grid.setObjectName(u"res_grid")
        self.sample_edit = QLabel(self.grid)
        self.sample_edit.setObjectName(u"sample_edit")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.sample_edit.setFont(font1)
        self.sample_edit.setTextFormat(Qt.TextFormat.PlainText)
        self.sample_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_edit.setWordWrap(False)

        self.res_grid.addWidget(self.sample_edit, 0, 1, 1, 1)

        self.sample_lab = QLabel(self.grid)
        self.sample_lab.setObjectName(u"sample_lab")
        self.sample_lab.setFont(font1)

        self.res_grid.addWidget(self.sample_lab, 0, 0, 1, 1)

        self.pos_edit = QLabel(self.grid)
        self.pos_edit.setObjectName(u"pos_edit")
        self.pos_edit.setFont(font1)
        self.pos_edit.setTextFormat(Qt.TextFormat.PlainText)
        self.pos_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pos_edit.setWordWrap(False)

        self.res_grid.addWidget(self.pos_edit, 1, 1, 1, 1)

        self.pos_lab = QLabel(self.grid)
        self.pos_lab.setObjectName(u"pos_lab")
        self.pos_lab.setFont(font1)

        self.res_grid.addWidget(self.pos_lab, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.grid)

        self.box = QDialogButtonBox(mainwindow)
        self.box.setObjectName(u"box")
        self.box.setOrientation(Qt.Orientation.Horizontal)
        self.box.setStandardButtons(QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout.addWidget(self.box)


        self.retranslateUi(mainwindow)
        self.box.accepted.connect(mainwindow.accept)
        self.box.rejected.connect(mainwindow.reject)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Operating Curve Results", None))
        self.title.setText(QCoreApplication.translate("mainwindow", u"Results of Operating Curve", None))
        self.sample_edit.setText(QCoreApplication.translate("mainwindow", u"_____", None))
        self.sample_lab.setText(QCoreApplication.translate("mainwindow", u"Sample Size:", None))
        self.pos_edit.setText(QCoreApplication.translate("mainwindow", u"_____", None))
        self.pos_lab.setText(QCoreApplication.translate("mainwindow", u"Positive Size:", None))
    # retranslateUi


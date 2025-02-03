# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interval.ui'
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
    QFrame, QHBoxLayout, QLabel, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(380, 320)
        mainwindow.setMinimumSize(QSize(380, 320))
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.title_lab = QLabel(mainwindow)
        self.title_lab.setObjectName(u"title_lab")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title_lab.sizePolicy().hasHeightForWidth())
        self.title_lab.setSizePolicy(sizePolicy)
        self.title_lab.setMinimumSize(QSize(0, 40))
        self.title_lab.setMaximumSize(QSize(16777215, 40))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.title_lab.setFont(font)
        self.title_lab.setStyleSheet(u"border: 1px solid grey;")
        self.title_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.title_lab)

        self.field = QWidget(mainwindow)
        self.field.setObjectName(u"field")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.field.sizePolicy().hasHeightForWidth())
        self.field.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.field)

        self.line = QFrame(mainwindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.interval = QHBoxLayout()
        self.interval.setObjectName(u"interval")
        self.spacer_3 = QSpacerItem(60, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.interval.addItem(self.spacer_3)

        self.from_lab = QLabel(mainwindow)
        self.from_lab.setObjectName(u"from_lab")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.from_lab.setFont(font1)

        self.interval.addWidget(self.from_lab)

        self.col_1 = QLabel(mainwindow)
        self.col_1.setObjectName(u"col_1")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.col_1.sizePolicy().hasHeightForWidth())
        self.col_1.setSizePolicy(sizePolicy2)

        self.interval.addWidget(self.col_1)

        self.from_edit = QLabel(mainwindow)
        self.from_edit.setObjectName(u"from_edit")
        self.from_edit.setFont(font1)

        self.interval.addWidget(self.from_edit)

        self.spacer_1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.interval.addItem(self.spacer_1)

        self.to_lab = QLabel(mainwindow)
        self.to_lab.setObjectName(u"to_lab")
        self.to_lab.setFont(font1)

        self.interval.addWidget(self.to_lab)

        self.col_2 = QLabel(mainwindow)
        self.col_2.setObjectName(u"col_2")
        sizePolicy2.setHeightForWidth(self.col_2.sizePolicy().hasHeightForWidth())
        self.col_2.setSizePolicy(sizePolicy2)

        self.interval.addWidget(self.col_2)

        self.to_edit = QLabel(mainwindow)
        self.to_edit.setObjectName(u"to_edit")
        self.to_edit.setFont(font1)

        self.interval.addWidget(self.to_edit)

        self.spacer_4 = QSpacerItem(60, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.interval.addItem(self.spacer_4)


        self.verticalLayout.addLayout(self.interval)

        self.line_2 = QFrame(mainwindow)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.box = QDialogButtonBox(mainwindow)
        self.box.setObjectName(u"box")
        self.box.setStandardButtons(QDialogButtonBox.StandardButton.No|QDialogButtonBox.StandardButton.Yes)

        self.verticalLayout.addWidget(self.box)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Acceptance Interval", None))
        self.title_lab.setText(QCoreApplication.translate("mainwindow", u"Papers in the interval", None))
        self.from_lab.setText(QCoreApplication.translate("mainwindow", u"From", None))
        self.col_1.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.from_edit.setText(QCoreApplication.translate("mainwindow", u"_____", None))
        self.to_lab.setText(QCoreApplication.translate("mainwindow", u"To", None))
        self.col_2.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.to_edit.setText(QCoreApplication.translate("mainwindow", u"_____", None))
    # retranslateUi


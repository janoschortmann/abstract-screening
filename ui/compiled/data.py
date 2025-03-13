# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'data.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QPushButton, QSizePolicy,
    QSplitter, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(450, 370)
        mainwindow.setStyleSheet(u".QPushButton:hover {\n"
"    background-color: #64b5f6;\n"
"    color: #fff;\n"
"}\n"
"\n"
".QPushButton:pressed {\n"
"    background-color: #bbdefb;\n"
"}")
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalWidget = QWidget(mainwindow)
        self.horizontalWidget.setObjectName(u"horizontalWidget")
        self.horizontalWidget.setStyleSheet(u"")
        self.top = QHBoxLayout(self.horizontalWidget)
        self.top.setSpacing(0)
        self.top.setObjectName(u"top")
        self.top.setContentsMargins(0, 0, 0, 0)
        self.find_but = QPushButton(self.horizontalWidget)
        self.find_but.setObjectName(u"find_but")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.find_but.sizePolicy().hasHeightForWidth())
        self.find_but.setSizePolicy(sizePolicy)
        self.find_but.setMinimumSize(QSize(80, 40))
        self.find_but.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.find_but.setAutoFillBackground(False)
        self.find_but.setStyleSheet(u"border: 1px solid grey;\n"
"margin-right: 5px;")

        self.top.addWidget(self.find_but)

        self.abs_path = QLabel(self.horizontalWidget)
        self.abs_path.setObjectName(u"abs_path")
        self.abs_path.setMinimumSize(QSize(0, 40))
        self.abs_path.setMaximumSize(QSize(16777215, 40))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.abs_path.setFont(font)
        self.abs_path.setStyleSheet(u"border: 1px solid grey;")
        self.abs_path.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.top.addWidget(self.abs_path)


        self.verticalLayout.addWidget(self.horizontalWidget)

        self.splitter = QSplitter(mainwindow)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        self.papers_view = QWidget(self.splitter)
        self.papers_view.setObjectName(u"papers_view")
        self.splitter.addWidget(self.papers_view)
        self.hbox_2 = QWidget(self.splitter)
        self.hbox_2.setObjectName(u"hbox_2")
        self.bottom = QHBoxLayout(self.hbox_2)
        self.bottom.setObjectName(u"bottom")
        self.bottom.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.bottom.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit(self.hbox_2)
        self.path.setObjectName(u"path")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.path.sizePolicy().hasHeightForWidth())
        self.path.setSizePolicy(sizePolicy1)
        self.path.setMinimumSize(QSize(100, 25))
        self.path.setMaximumSize(QSize(16777215, 16777215))
        self.path.setFrame(False)

        self.bottom.addWidget(self.path)

        self.specifier = QComboBox(self.hbox_2)
        self.specifier.setObjectName(u"specifier")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.specifier.sizePolicy().hasHeightForWidth())
        self.specifier.setSizePolicy(sizePolicy2)
        self.specifier.setMinimumSize(QSize(80, 0))

        self.bottom.addWidget(self.specifier)

        self.add_but = QPushButton(self.hbox_2)
        self.add_but.setObjectName(u"add_but")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.add_but.sizePolicy().hasHeightForWidth())
        self.add_but.setSizePolicy(sizePolicy3)
        self.add_but.setMinimumSize(QSize(0, 25))
        self.add_but.setMaximumSize(QSize(100, 16777215))
        self.add_but.setAutoFillBackground(False)
        self.add_but.setStyleSheet(u"background-color: rgb(115, 255, 115)")

        self.bottom.addWidget(self.add_but)

        self.remove_but = QPushButton(self.hbox_2)
        self.remove_but.setObjectName(u"remove_but")
        sizePolicy3.setHeightForWidth(self.remove_but.sizePolicy().hasHeightForWidth())
        self.remove_but.setSizePolicy(sizePolicy3)
        self.remove_but.setMinimumSize(QSize(0, 25))
        self.remove_but.setMaximumSize(QSize(100, 16777215))
        self.remove_but.setAutoFillBackground(False)
        self.remove_but.setStyleSheet(u"background-color: rgb(255, 90, 90)")

        self.bottom.addWidget(self.remove_but)

        self.splitter.addWidget(self.hbox_2)

        self.verticalLayout.addWidget(self.splitter)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Paper List Window", None))
        self.find_but.setText(QCoreApplication.translate("mainwindow", u"Find", None))
        self.abs_path.setText(QCoreApplication.translate("mainwindow", u"Complete List of Papers", None))
        self.path.setPlaceholderText(QCoreApplication.translate("mainwindow", u"File or Directory", None))
        self.add_but.setText(QCoreApplication.translate("mainwindow", u"Add", None))
        self.remove_but.setText(QCoreApplication.translate("mainwindow", u"Remove", None))
    # retranslateUi


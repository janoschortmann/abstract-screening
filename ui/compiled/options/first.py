# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'first.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QPushButton, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget)

class Ui_first_option(object):
    def setupUi(self, first_option):
        if not first_option.objectName():
            first_option.setObjectName(u"first_option")
        first_option.resize(650, 235)
        self.horizontalLayout = QHBoxLayout(first_option)
        self.horizontalLayout.setSpacing(15)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, -1)
        self.variables = QWidget(first_option)
        self.variables.setObjectName(u"variables")
        self.variables_layout = QVBoxLayout(self.variables)
        self.variables_layout.setObjectName(u"variables_layout")
        self.variables_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.edit_grid = QWidget(self.variables)
        self.edit_grid.setObjectName(u"edit_grid")
        self.edit_grid.setStyleSheet(u"border: 1px solid grey;")
        self.edits = QGridLayout(self.edit_grid)
        self.edits.setObjectName(u"edits")
        self.edits.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.dir_lab = QLabel(self.edit_grid)
        self.dir_lab.setObjectName(u"dir_lab")
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(10)
        font.setItalic(True)
        self.dir_lab.setFont(font)
        self.dir_lab.setStyleSheet(u"border: 0px solid black;")

        self.edits.addWidget(self.dir_lab, 1, 0, 1, 1)

        self.col_3 = QLabel(self.edit_grid)
        self.col_3.setObjectName(u"col_3")
        self.col_3.setStyleSheet(u"border: 0px solid black;")
        self.col_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.edits.addWidget(self.col_3, 1, 4, 1, 1)

        self.date_edit = QLineEdit(self.edit_grid)
        self.date_edit.setObjectName(u"date_edit")
        self.date_edit.setStyleSheet(u"border: 0px solid black;")

        self.edits.addWidget(self.date_edit, 0, 2, 1, 1)

        self.date_lab = QLabel(self.edit_grid)
        self.date_lab.setObjectName(u"date_lab")
        self.date_lab.setFont(font)
        self.date_lab.setStyleSheet(u"border: 0px solid black;")

        self.edits.addWidget(self.date_lab, 0, 0, 1, 1)

        self.sample_edit = QLineEdit(self.edit_grid)
        self.sample_edit.setObjectName(u"sample_edit")
        self.sample_edit.setStyleSheet(u"border: 0px solid black;")

        self.edits.addWidget(self.sample_edit, 0, 5, 1, 1)

        self.col_2 = QLabel(self.edit_grid)
        self.col_2.setObjectName(u"col_2")
        self.col_2.setStyleSheet(u"border: 0px solid black;")
        self.col_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.edits.addWidget(self.col_2, 1, 1, 1, 1)

        self.col_4 = QLabel(self.edit_grid)
        self.col_4.setObjectName(u"col_4")
        self.col_4.setStyleSheet(u"border: 0px solid black;")
        self.col_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.edits.addWidget(self.col_4, 0, 4, 1, 1)

        self.directory_edit = QLineEdit(self.edit_grid)
        self.directory_edit.setObjectName(u"directory_edit")
        self.directory_edit.setStyleSheet(u"border: 0px solid black;")

        self.edits.addWidget(self.directory_edit, 1, 2, 1, 1)

        self.validation_edit = QLineEdit(self.edit_grid)
        self.validation_edit.setObjectName(u"validation_edit")
        self.validation_edit.setStyleSheet(u"border: 0px solid black;")

        self.edits.addWidget(self.validation_edit, 1, 5, 1, 1)

        self.col_1 = QLabel(self.edit_grid)
        self.col_1.setObjectName(u"col_1")
        self.col_1.setStyleSheet(u"border: 0px solid black;")
        self.col_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.edits.addWidget(self.col_1, 0, 1, 1, 1)

        self.sample_lab = QLabel(self.edit_grid)
        self.sample_lab.setObjectName(u"sample_lab")
        self.sample_lab.setFont(font)
        self.sample_lab.setStyleSheet(u"border: 0px solid black;")
        self.sample_lab.setIndent(10)

        self.edits.addWidget(self.sample_lab, 0, 3, 1, 1)

        self.validation_lab = QLabel(self.edit_grid)
        self.validation_lab.setObjectName(u"validation_lab")
        self.validation_lab.setFont(font)
        self.validation_lab.setStyleSheet(u"border: 0px solid black;")
        self.validation_lab.setIndent(10)

        self.edits.addWidget(self.validation_lab, 1, 3, 1, 1)

        self.edits.setColumnMinimumWidth(1, 10)
        self.edits.setColumnMinimumWidth(4, 10)
        self.edits.setRowMinimumHeight(0, 50)
        self.edits.setRowMinimumHeight(1, 50)

        self.variables_layout.addWidget(self.edit_grid)

        self.query_buttons = QWidget(self.variables)
        self.query_buttons.setObjectName(u"query_buttons")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.query_buttons.sizePolicy().hasHeightForWidth())
        self.query_buttons.setSizePolicy(sizePolicy)
        self.query_buttons.setMinimumSize(QSize(0, 0))
        self.query_buttons.setMaximumSize(QSize(16777215, 70))
        self.buttons = QHBoxLayout(self.query_buttons)
        self.buttons.setSpacing(20)
        self.buttons.setObjectName(u"buttons")
        self.buttons.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.add = QPushButton(self.query_buttons)
        self.add.setObjectName(u"add")
        self.add.setMinimumSize(QSize(0, 30))
        self.add.setMaximumSize(QSize(16777215, 16777215))
        font1 = QFont()
        font1.setPointSize(10)
        self.add.setFont(font1)
        self.add.setStyleSheet(u"background-color: rgb(115, 255, 115);")

        self.buttons.addWidget(self.add)

        self.remove = QPushButton(self.query_buttons)
        self.remove.setObjectName(u"remove")
        self.remove.setMinimumSize(QSize(0, 30))
        self.remove.setMaximumSize(QSize(16777215, 16777215))
        self.remove.setFont(font1)
        self.remove.setStyleSheet(u"background-color: rgb(255, 90, 90);")

        self.buttons.addWidget(self.remove)


        self.variables_layout.addWidget(self.query_buttons)


        self.horizontalLayout.addWidget(self.variables)

        self.query_box = QTextEdit(first_option)
        self.query_box.setObjectName(u"query_box")

        self.horizontalLayout.addWidget(self.query_box)


        self.retranslateUi(first_option)

        QMetaObject.connectSlotsByName(first_option)
    # setupUi

    def retranslateUi(self, first_option):
        first_option.setWindowTitle(QCoreApplication.translate("first_option", u"Form", None))
        self.dir_lab.setText(QCoreApplication.translate("first_option", u"Directory", None))
        self.col_3.setText(QCoreApplication.translate("first_option", u":", None))
        self.date_lab.setText(QCoreApplication.translate("first_option", u"Date Range", None))
        self.col_2.setText(QCoreApplication.translate("first_option", u":", None))
        self.col_4.setText(QCoreApplication.translate("first_option", u":", None))
        self.col_1.setText(QCoreApplication.translate("first_option", u":", None))
        self.sample_lab.setText(QCoreApplication.translate("first_option", u"Sample Size", None))
        self.validation_lab.setText(QCoreApplication.translate("first_option", u"Validation Size", None))
        self.add.setText(QCoreApplication.translate("first_option", u"Add", None))
        self.remove.setText(QCoreApplication.translate("first_option", u"Remove", None))
    # retranslateUi


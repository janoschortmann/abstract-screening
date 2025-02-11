# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'second.ui'
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
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QWidget)

class Ui_second_option(object):
    def setupUi(self, second_option):
        if not second_option.objectName():
            second_option.setObjectName(u"second_option")
        second_option.resize(650, 235)
        self.horizontalLayout = QHBoxLayout(second_option)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, -1)
        self.main_grid = QGridLayout()
        self.main_grid.setObjectName(u"main_grid")
        self.main_grid.setVerticalSpacing(10)
        self.col_4 = QLabel(second_option)
        self.col_4.setObjectName(u"col_4")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.col_4.sizePolicy().hasHeightForWidth())
        self.col_4.setSizePolicy(sizePolicy)
        self.col_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.main_grid.addWidget(self.col_4, 1, 5, 1, 1)

        self.alpha_lab = QLabel(second_option)
        self.alpha_lab.setObjectName(u"alpha_lab")
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(10)
        font.setItalic(True)
        self.alpha_lab.setFont(font)

        self.main_grid.addWidget(self.alpha_lab, 0, 0, 1, 1)

        self.col_1 = QLabel(second_option)
        self.col_1.setObjectName(u"col_1")
        sizePolicy.setHeightForWidth(self.col_1.sizePolicy().hasHeightForWidth())
        self.col_1.setSizePolicy(sizePolicy)
        self.col_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.main_grid.addWidget(self.col_1, 0, 1, 1, 1)

        self.param_lab_2 = QLabel(second_option)
        self.param_lab_2.setObjectName(u"param_lab_2")
        self.param_lab_2.setFont(font)
        self.param_lab_2.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.main_grid.addWidget(self.param_lab_2, 1, 4, 1, 1)

        self.clear = QPushButton(second_option)
        self.clear.setObjectName(u"clear")
        self.clear.setMinimumSize(QSize(0, 0))
        self.clear.setStyleSheet(u"background-color: rgb(255, 90, 90);")

        self.main_grid.addWidget(self.clear, 1, 7, 1, 1)

        self.param_edit_1 = QLineEdit(second_option)
        self.param_edit_1.setObjectName(u"param_edit_1")
        self.param_edit_1.setMinimumSize(QSize(0, 0))

        self.main_grid.addWidget(self.param_edit_1, 0, 6, 1, 1)

        self.param_lab_1 = QLabel(second_option)
        self.param_lab_1.setObjectName(u"param_lab_1")
        self.param_lab_1.setFont(font)
        self.param_lab_1.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.main_grid.addWidget(self.param_lab_1, 0, 4, 1, 1)

        self.col_2 = QLabel(second_option)
        self.col_2.setObjectName(u"col_2")
        sizePolicy.setHeightForWidth(self.col_2.sizePolicy().hasHeightForWidth())
        self.col_2.setSizePolicy(sizePolicy)
        self.col_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.main_grid.addWidget(self.col_2, 1, 1, 1, 1)

        self.beta_edit = QLineEdit(second_option)
        self.beta_edit.setObjectName(u"beta_edit")
        self.beta_edit.setMinimumSize(QSize(0, 0))

        self.main_grid.addWidget(self.beta_edit, 1, 2, 1, 1)

        self.alpha_edit = QLineEdit(second_option)
        self.alpha_edit.setObjectName(u"alpha_edit")
        self.alpha_edit.setMinimumSize(QSize(0, 0))

        self.main_grid.addWidget(self.alpha_edit, 0, 2, 1, 1)

        self.beta_lab = QLabel(second_option)
        self.beta_lab.setObjectName(u"beta_lab")
        self.beta_lab.setFont(font)

        self.main_grid.addWidget(self.beta_lab, 1, 0, 1, 1)

        self.col_3 = QLabel(second_option)
        self.col_3.setObjectName(u"col_3")
        sizePolicy.setHeightForWidth(self.col_3.sizePolicy().hasHeightForWidth())
        self.col_3.setSizePolicy(sizePolicy)
        self.col_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.main_grid.addWidget(self.col_3, 0, 5, 1, 1)

        self.plot = QPushButton(second_option)
        self.plot.setObjectName(u"plot")
        self.plot.setMinimumSize(QSize(0, 0))
        self.plot.setStyleSheet(u"background-color: rgb(115, 255, 115);")

        self.main_grid.addWidget(self.plot, 0, 7, 1, 1)

        self.param_edit_2 = QLineEdit(second_option)
        self.param_edit_2.setObjectName(u"param_edit_2")
        self.param_edit_2.setMinimumSize(QSize(0, 0))

        self.main_grid.addWidget(self.param_edit_2, 1, 6, 1, 1)

        self.spacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.main_grid.addItem(self.spacer_2, 0, 3, 1, 1)

        self.spacer_1 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.main_grid.addItem(self.spacer_1, 1, 3, 1, 1)


        self.horizontalLayout.addLayout(self.main_grid)


        self.retranslateUi(second_option)

        QMetaObject.connectSlotsByName(second_option)
    # setupUi

    def retranslateUi(self, second_option):
        second_option.setWindowTitle(QCoreApplication.translate("second_option", u"Form", None))
        self.col_4.setText(QCoreApplication.translate("second_option", u":", None))
        self.alpha_lab.setText(QCoreApplication.translate("second_option", u"Alpha", None))
        self.col_1.setText(QCoreApplication.translate("second_option", u":", None))
        self.param_lab_2.setText(QCoreApplication.translate("second_option", u"Parameter 2", None))
        self.clear.setText(QCoreApplication.translate("second_option", u"Clear", None))
        self.param_lab_1.setText(QCoreApplication.translate("second_option", u"Parameter 1", None))
        self.col_2.setText(QCoreApplication.translate("second_option", u":", None))
        self.beta_lab.setText(QCoreApplication.translate("second_option", u"Beta", None))
        self.col_3.setText(QCoreApplication.translate("second_option", u":", None))
        self.plot.setText(QCoreApplication.translate("second_option", u"Plot", None))
    # retranslateUi


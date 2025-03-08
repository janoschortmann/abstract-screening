# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'third.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QLabel,
    QLineEdit, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_third_option(object):
    def setupUi(self, third_option):
        if not third_option.objectName():
            third_option.setObjectName(u"third_option")
        third_option.resize(650, 235)
        third_option.setStyleSheet(u"")
        self.verticalLayout = QVBoxLayout(third_option)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(-1, 0, -1, -1)
        self.vars = QWidget(third_option)
        self.vars.setObjectName(u"vars")
        self.vars.setStyleSheet(u"border: 1px solid grey;")
        self.variables = QGridLayout(self.vars)
        self.variables.setObjectName(u"variables")
        self.variables.setContentsMargins(-1, 0, -1, -1)
        self.col_1 = QLabel(self.vars)
        self.col_1.setObjectName(u"col_1")
        self.col_1.setStyleSheet(u"border: 0px solid grey;")
        self.col_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.variables.addWidget(self.col_1, 0, 1, 1, 1)

        self.col_2 = QLabel(self.vars)
        self.col_2.setObjectName(u"col_2")
        self.col_2.setStyleSheet(u"border: 0px solid grey;")
        self.col_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.variables.addWidget(self.col_2, 0, 5, 1, 1)

        self.col_3 = QLabel(self.vars)
        self.col_3.setObjectName(u"col_3")
        self.col_3.setStyleSheet(u"border: 0px solid grey;")
        self.col_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.variables.addWidget(self.col_3, 1, 1, 1, 1)

        self.pos_lab = QLabel(self.vars)
        self.pos_lab.setObjectName(u"pos_lab")
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(10)
        font.setItalic(True)
        self.pos_lab.setFont(font)
        self.pos_lab.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.pos_lab, 0, 4, 1, 1)

        self.splits_lab = QLabel(self.vars)
        self.splits_lab.setObjectName(u"splits_lab")
        self.splits_lab.setFont(font)
        self.splits_lab.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.splits_lab, 1, 4, 1, 1)

        self.size_lab = QLabel(self.vars)
        self.size_lab.setObjectName(u"size_lab")
        self.size_lab.setFont(font)
        self.size_lab.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.size_lab, 0, 0, 1, 1)

        self.splits_edit = QLineEdit(self.vars)
        self.splits_edit.setObjectName(u"splits_edit")
        self.splits_edit.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.splits_edit, 1, 6, 1, 1)

        self.seed_edit = QLineEdit(self.vars)
        self.seed_edit.setObjectName(u"seed_edit")
        self.seed_edit.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.seed_edit, 1, 2, 1, 1)

        self.seed_lab = QLabel(self.vars)
        self.seed_lab.setObjectName(u"seed_lab")
        self.seed_lab.setFont(font)
        self.seed_lab.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.seed_lab, 1, 0, 1, 1)

        self.size_edit = QLineEdit(self.vars)
        self.size_edit.setObjectName(u"size_edit")
        self.size_edit.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.size_edit, 0, 2, 1, 1)

        self.pos_edit = QLineEdit(self.vars)
        self.pos_edit.setObjectName(u"pos_edit")
        self.pos_edit.setStyleSheet(u"border: 0px solid grey;")

        self.variables.addWidget(self.pos_edit, 0, 6, 1, 1)

        self.col_4 = QLabel(self.vars)
        self.col_4.setObjectName(u"col_4")
        self.col_4.setStyleSheet(u"border: 0px solid grey;")
        self.col_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.variables.addWidget(self.col_4, 1, 5, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.variables.addItem(self.horizontalSpacer, 0, 3, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.variables.addItem(self.horizontalSpacer_2, 1, 3, 1, 1)

        self.variables.setRowMinimumHeight(0, 70)
        self.variables.setRowMinimumHeight(1, 70)

        self.verticalLayout.addWidget(self.vars)

        self.binary_vars = QWidget(third_option)
        self.binary_vars.setObjectName(u"binary_vars")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.binary_vars.sizePolicy().hasHeightForWidth())
        self.binary_vars.setSizePolicy(sizePolicy)
        self.binary_vars.setStyleSheet(u"")
        self.binaries = QGridLayout(self.binary_vars)
        self.binaries.setObjectName(u"binaries")
        self.sampling_lab = QLabel(self.binary_vars)
        self.sampling_lab.setObjectName(u"sampling_lab")
        self.sampling_lab.setFont(font)

        self.binaries.addWidget(self.sampling_lab, 0, 5, 1, 1)

        self.spacer_1 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.binaries.addItem(self.spacer_1, 0, 4, 1, 1)

        self.col_6 = QLabel(self.binary_vars)
        self.col_6.setObjectName(u"col_6")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.col_6.sizePolicy().hasHeightForWidth())
        self.col_6.setSizePolicy(sizePolicy1)
        self.col_6.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.binaries.addWidget(self.col_6, 0, 2, 1, 1)

        self.model_box = QComboBox(self.binary_vars)
        self.model_box.setObjectName(u"model_box")
        sizePolicy.setHeightForWidth(self.model_box.sizePolicy().hasHeightForWidth())
        self.model_box.setSizePolicy(sizePolicy)
        self.model_box.setMinimumSize(QSize(100, 0))

        self.binaries.addWidget(self.model_box, 0, 3, 1, 1)

        self.model_lab = QLabel(self.binary_vars)
        self.model_lab.setObjectName(u"model_lab")
        self.model_lab.setFont(font)

        self.binaries.addWidget(self.model_lab, 0, 1, 1, 1)

        self.sampling_box = QComboBox(self.binary_vars)
        self.sampling_box.setObjectName(u"sampling_box")
        sizePolicy.setHeightForWidth(self.sampling_box.sizePolicy().hasHeightForWidth())
        self.sampling_box.setSizePolicy(sizePolicy)
        self.sampling_box.setMinimumSize(QSize(100, 0))

        self.binaries.addWidget(self.sampling_box, 0, 7, 1, 1)

        self.spacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.binaries.addItem(self.spacer_2, 0, 0, 1, 1)

        self.col_5 = QLabel(self.binary_vars)
        self.col_5.setObjectName(u"col_5")
        sizePolicy1.setHeightForWidth(self.col_5.sizePolicy().hasHeightForWidth())
        self.col_5.setSizePolicy(sizePolicy1)
        self.col_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.binaries.addWidget(self.col_5, 0, 6, 1, 1)

        self.spacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.binaries.addItem(self.spacer_3, 0, 8, 1, 1)


        self.verticalLayout.addWidget(self.binary_vars)

        QWidget.setTabOrder(self.size_edit, self.pos_edit)
        QWidget.setTabOrder(self.pos_edit, self.seed_edit)
        QWidget.setTabOrder(self.seed_edit, self.splits_edit)
        QWidget.setTabOrder(self.splits_edit, self.model_box)
        QWidget.setTabOrder(self.model_box, self.sampling_box)

        self.retranslateUi(third_option)

        QMetaObject.connectSlotsByName(third_option)
    # setupUi

    def retranslateUi(self, third_option):
        third_option.setWindowTitle(QCoreApplication.translate("third_option", u"Form", None))
        self.col_1.setText(QCoreApplication.translate("third_option", u":", None))
        self.col_2.setText(QCoreApplication.translate("third_option", u":", None))
        self.col_3.setText(QCoreApplication.translate("third_option", u":", None))
        self.pos_lab.setText(QCoreApplication.translate("third_option", u"Positive Ratio", None))
        self.splits_lab.setText(QCoreApplication.translate("third_option", u"Splits", None))
        self.size_lab.setText(QCoreApplication.translate("third_option", u"Testing Size", None))
        self.splits_edit.setText(QCoreApplication.translate("third_option", u"2", None))
        self.seed_edit.setText(QCoreApplication.translate("third_option", u"0", None))
        self.seed_lab.setText(QCoreApplication.translate("third_option", u"Seed", None))
        self.size_edit.setText(QCoreApplication.translate("third_option", u"0.2", None))
        self.pos_edit.setText(QCoreApplication.translate("third_option", u"0.3", None))
        self.col_4.setText(QCoreApplication.translate("third_option", u":", None))
        self.sampling_lab.setText(QCoreApplication.translate("third_option", u"Sampling", None))
        self.col_6.setText(QCoreApplication.translate("third_option", u":", None))
        self.model_lab.setText(QCoreApplication.translate("third_option", u"Model", None))
        self.col_5.setText(QCoreApplication.translate("third_option", u":", None))
    # retranslateUi


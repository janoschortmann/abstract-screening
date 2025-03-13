# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'stats.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(600, 300)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(mainwindow.sizePolicy().hasHeightForWidth())
        mainwindow.setSizePolicy(sizePolicy)
        mainwindow.setMinimumSize(QSize(600, 300))
        mainwindow.setMaximumSize(QSize(600, 300))
        mainwindow.setSizeIncrement(QSize(0, 0))
        mainwindow.setStyleSheet(u"")
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.model_lab = QLabel(mainwindow)
        self.model_lab.setObjectName(u"model_lab")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.model_lab.sizePolicy().hasHeightForWidth())
        self.model_lab.setSizePolicy(sizePolicy1)
        self.model_lab.setMinimumSize(QSize(0, 40))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        self.model_lab.setFont(font)
        self.model_lab.setStyleSheet(u"border: 1px solid grey;")
        self.model_lab.setFrameShape(QFrame.Shape.NoFrame)
        self.model_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.model_lab)

        self.table = QWidget(mainwindow)
        self.table.setObjectName(u"table")
        self.table.setStyleSheet(u"border: 1px solid grey;")
        self.gridLayout = QGridLayout(self.table)
        self.gridLayout.setObjectName(u"gridLayout")
        self.pn_lab = QLabel(self.table)
        self.pn_lab.setObjectName(u"pn_lab")
        self.pn_lab.setMinimumSize(QSize(150, 0))
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.pn_lab.setFont(font1)
        self.pn_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.pn_lab, 0, 1, 1, 1)

        self.pp_lab = QLabel(self.table)
        self.pp_lab.setObjectName(u"pp_lab")
        self.pp_lab.setMinimumSize(QSize(150, 0))
        self.pp_lab.setFont(font1)
        self.pp_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.pp_lab, 0, 2, 1, 1)

        self.an_lab = QLabel(self.table)
        self.an_lab.setObjectName(u"an_lab")
        self.an_lab.setMinimumSize(QSize(150, 0))
        self.an_lab.setFont(font1)
        self.an_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.an_lab, 1, 0, 1, 1)

        self.ap_lab = QLabel(self.table)
        self.ap_lab.setObjectName(u"ap_lab")
        self.ap_lab.setMinimumSize(QSize(150, 0))
        self.ap_lab.setFont(font1)
        self.ap_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.ap_lab, 2, 0, 1, 1)

        self.an_stat = QLabel(self.table)
        self.an_stat.setObjectName(u"an_stat")
        self.an_stat.setStyleSheet(u"color: lightgreen;")
        self.an_stat.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.an_stat, 1, 1, 1, 1)

        self.fp_stat = QLabel(self.table)
        self.fp_stat.setObjectName(u"fp_stat")
        self.fp_stat.setStyleSheet(u"color: orange;")
        self.fp_stat.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.fp_stat, 1, 2, 1, 1)

        self.fn_stat = QLabel(self.table)
        self.fn_stat.setObjectName(u"fn_stat")
        self.fn_stat.setStyleSheet(u"color: orange;")
        self.fn_stat.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.fn_stat, 2, 1, 1, 1)

        self.ap_stat = QLabel(self.table)
        self.ap_stat.setObjectName(u"ap_stat")
        self.ap_stat.setStyleSheet(u"color: lightgreen;")
        self.ap_stat.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.ap_stat, 2, 2, 1, 1)

        self.filler = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.filler, 0, 0, 1, 1)


        self.verticalLayout.addWidget(self.table)

        self.precision_box = QWidget(mainwindow)
        self.precision_box.setObjectName(u"precision_box")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.precision_box.sizePolicy().hasHeightForWidth())
        self.precision_box.setSizePolicy(sizePolicy2)
        self.precision_box.setStyleSheet(u"")
        self.precision = QHBoxLayout(self.precision_box)
        self.precision.setSpacing(10)
        self.precision.setObjectName(u"precision")
        self.acc_lab = QLabel(self.precision_box)
        self.acc_lab.setObjectName(u"acc_lab")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.acc_lab.sizePolicy().hasHeightForWidth())
        self.acc_lab.setSizePolicy(sizePolicy3)
        self.acc_lab.setStyleSheet(u"")
        self.acc_lab.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.precision.addWidget(self.acc_lab)

        self.col_1 = QLabel(self.precision_box)
        self.col_1.setObjectName(u"col_1")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.col_1.sizePolicy().hasHeightForWidth())
        self.col_1.setSizePolicy(sizePolicy4)
        self.col_1.setStyleSheet(u"")
        self.col_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.precision.addWidget(self.col_1)

        self.acc_show = QLabel(self.precision_box)
        self.acc_show.setObjectName(u"acc_show")
        self.acc_show.setStyleSheet(u"")

        self.precision.addWidget(self.acc_show)

        self.recall_lab = QLabel(self.precision_box)
        self.recall_lab.setObjectName(u"recall_lab")
        self.recall_lab.setStyleSheet(u"")

        self.precision.addWidget(self.recall_lab)

        self.col_2 = QLabel(self.precision_box)
        self.col_2.setObjectName(u"col_2")
        sizePolicy4.setHeightForWidth(self.col_2.sizePolicy().hasHeightForWidth())
        self.col_2.setSizePolicy(sizePolicy4)
        self.col_2.setStyleSheet(u"")
        self.col_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.precision.addWidget(self.col_2)

        self.recall_show = QLabel(self.precision_box)
        self.recall_show.setObjectName(u"recall_show")
        self.recall_show.setStyleSheet(u"")

        self.precision.addWidget(self.recall_show)

        self.pre_lab = QLabel(self.precision_box)
        self.pre_lab.setObjectName(u"pre_lab")
        self.pre_lab.setStyleSheet(u"")

        self.precision.addWidget(self.pre_lab)

        self.col_3 = QLabel(self.precision_box)
        self.col_3.setObjectName(u"col_3")
        sizePolicy4.setHeightForWidth(self.col_3.sizePolicy().hasHeightForWidth())
        self.col_3.setSizePolicy(sizePolicy4)
        self.col_3.setStyleSheet(u"")
        self.col_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.precision.addWidget(self.col_3)

        self.prec_show = QLabel(self.precision_box)
        self.prec_show.setObjectName(u"prec_show")
        self.prec_show.setStyleSheet(u"")

        self.precision.addWidget(self.prec_show)

        self.f1_lab = QLabel(self.precision_box)
        self.f1_lab.setObjectName(u"f1_lab")

        self.precision.addWidget(self.f1_lab)

        self.col_4 = QLabel(self.precision_box)
        self.col_4.setObjectName(u"col_4")
        sizePolicy4.setHeightForWidth(self.col_4.sizePolicy().hasHeightForWidth())
        self.col_4.setSizePolicy(sizePolicy4)

        self.precision.addWidget(self.col_4)

        self.f1_show = QLabel(self.precision_box)
        self.f1_show.setObjectName(u"f1_show")

        self.precision.addWidget(self.f1_show)


        self.verticalLayout.addWidget(self.precision_box)

        self.line = QFrame(mainwindow)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.horizontalWidget = QWidget(mainwindow)
        self.horizontalWidget.setObjectName(u"horizontalWidget")
        sizePolicy2.setHeightForWidth(self.horizontalWidget.sizePolicy().hasHeightForWidth())
        self.horizontalWidget.setSizePolicy(sizePolicy2)
        self.horizontalLayout = QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.spacer)

        self.save = QPushButton(self.horizontalWidget)
        self.save.setObjectName(u"save")

        self.horizontalLayout.addWidget(self.save)

        self.cancel = QPushButton(self.horizontalWidget)
        self.cancel.setObjectName(u"cancel")

        self.horizontalLayout.addWidget(self.cancel)


        self.verticalLayout.addWidget(self.horizontalWidget)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"AI Statistics Window", None))
        self.model_lab.setText(QCoreApplication.translate("mainwindow", u"~~~  Model Statistics  ~~~", None))
        self.pn_lab.setText(QCoreApplication.translate("mainwindow", u"Predicted Negatives", None))
        self.pp_lab.setText(QCoreApplication.translate("mainwindow", u"Predicted Positives", None))
        self.an_lab.setText(QCoreApplication.translate("mainwindow", u"Actual Negatives", None))
        self.ap_lab.setText(QCoreApplication.translate("mainwindow", u"Actual Positives", None))
        self.an_stat.setText(QCoreApplication.translate("mainwindow", u"AN", None))
        self.fp_stat.setText(QCoreApplication.translate("mainwindow", u"FP", None))
        self.fn_stat.setText(QCoreApplication.translate("mainwindow", u"FN", None))
        self.ap_stat.setText(QCoreApplication.translate("mainwindow", u"AP", None))
        self.acc_lab.setText(QCoreApplication.translate("mainwindow", u"Accuracy", None))
        self.col_1.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.acc_show.setText(QCoreApplication.translate("mainwindow", u"__________", None))
        self.recall_lab.setText(QCoreApplication.translate("mainwindow", u"Recall", None))
        self.col_2.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.recall_show.setText(QCoreApplication.translate("mainwindow", u"__________", None))
        self.pre_lab.setText(QCoreApplication.translate("mainwindow", u"Precision", None))
        self.col_3.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.prec_show.setText(QCoreApplication.translate("mainwindow", u"__________", None))
        self.f1_lab.setText(QCoreApplication.translate("mainwindow", u"F1", None))
        self.col_4.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.f1_show.setText(QCoreApplication.translate("mainwindow", u"__________", None))
        self.save.setText(QCoreApplication.translate("mainwindow", u"Save", None))
        self.cancel.setText(QCoreApplication.translate("mainwindow", u"Cancel", None))
    # retranslateUi


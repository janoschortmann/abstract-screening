# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'first.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
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
    QLayout, QLineEdit, QPlainTextEdit, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

class Ui_first_option(object):
    def setupUi(self, first_option):
        if not first_option.objectName():
            first_option.setObjectName(u"first_option")
        first_option.resize(650, 240)
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
        self.edits.setContentsMargins(20, -1, 25, -1)
        self.from_lab = QLabel(self.edit_grid)
        self.from_lab.setObjectName(u"from_lab")
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(10)
        font.setItalic(True)
        self.from_lab.setFont(font)
        self.from_lab.setStyleSheet(u"border: 0px solig grey;")
        self.from_lab.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.edits.addWidget(self.from_lab, 0, 0, 1, 1)

        self.col_5 = QLabel(self.edit_grid)
        self.col_5.setObjectName(u"col_5")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(False)
        self.col_5.setFont(font1)
        self.col_5.setStyleSheet(u"border: 0px solig grey;")

        self.edits.addWidget(self.col_5, 0, 5, 1, 1)

        self.sample_edit = QLineEdit(self.edit_grid)
        self.sample_edit.setObjectName(u"sample_edit")
        self.sample_edit.setStyleSheet(u"")

        self.edits.addWidget(self.sample_edit, 0, 9, 1, 1)

        self.col_4 = QLabel(self.edit_grid)
        self.col_4.setObjectName(u"col_4")
        self.col_4.setStyleSheet(u"border: 0px solid black;")
        self.col_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.edits.addWidget(self.col_4, 0, 8, 1, 1)

        self.from_edit = QLineEdit(self.edit_grid)
        self.from_edit.setObjectName(u"from_edit")
        self.from_edit.setMaxLength(4)

        self.edits.addWidget(self.from_edit, 0, 3, 1, 1)

        self.dir_lab = QLabel(self.edit_grid)
        self.dir_lab.setObjectName(u"dir_lab")
        self.dir_lab.setFont(font)
        self.dir_lab.setStyleSheet(u"border: 0px solid black;")

        self.edits.addWidget(self.dir_lab, 1, 0, 1, 2)

        self.col_3 = QLabel(self.edit_grid)
        self.col_3.setObjectName(u"col_3")
        self.col_3.setStyleSheet(u"border: 0px solid black;")
        self.col_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.edits.addWidget(self.col_3, 1, 8, 1, 1)

        self.validation_lab = QLabel(self.edit_grid)
        self.validation_lab.setObjectName(u"validation_lab")
        self.validation_lab.setFont(font)
        self.validation_lab.setStyleSheet(u"border: 0px solid black;")
        self.validation_lab.setIndent(10)

        self.edits.addWidget(self.validation_lab, 1, 7, 1, 1)

        self.validation_edit = QLineEdit(self.edit_grid)
        self.validation_edit.setObjectName(u"validation_edit")
        self.validation_edit.setStyleSheet(u"")

        self.edits.addWidget(self.validation_edit, 1, 9, 1, 1)

        self.sample_lab = QLabel(self.edit_grid)
        self.sample_lab.setObjectName(u"sample_lab")
        self.sample_lab.setFont(font)
        self.sample_lab.setStyleSheet(u"border: 0px solid black;")
        self.sample_lab.setIndent(10)

        self.edits.addWidget(self.sample_lab, 0, 7, 1, 1)

        self.col_1 = QLabel(self.edit_grid)
        self.col_1.setObjectName(u"col_1")
        self.col_1.setFont(font1)
        self.col_1.setStyleSheet(u"border: 0px solig grey;")

        self.edits.addWidget(self.col_1, 0, 2, 1, 1)

        self.to_lab = QLabel(self.edit_grid)
        self.to_lab.setObjectName(u"to_lab")
        self.to_lab.setFont(font)
        self.to_lab.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.to_lab.setStyleSheet(u"border: 0px solig grey;")
        self.to_lab.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.edits.addWidget(self.to_lab, 0, 4, 1, 1)

        self.col_2 = QLabel(self.edit_grid)
        self.col_2.setObjectName(u"col_2")
        self.col_2.setStyleSheet(u"border: 0px solid black;")
        self.col_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.edits.addWidget(self.col_2, 1, 2, 1, 1)

        self.to_edit = QLineEdit(self.edit_grid)
        self.to_edit.setObjectName(u"to_edit")
        self.to_edit.setMaxLength(4)

        self.edits.addWidget(self.to_edit, 0, 6, 1, 1)

        self.directory_edit = QLineEdit(self.edit_grid)
        self.directory_edit.setObjectName(u"directory_edit")
        self.directory_edit.setStyleSheet(u"")

        self.edits.addWidget(self.directory_edit, 1, 3, 1, 4)

        self.edits.setColumnMinimumWidth(4, 30)

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
        font2 = QFont()
        font2.setPointSize(10)
        self.add.setFont(font2)
        self.add.setStyleSheet(u"background-color: rgb(115, 255, 115);")

        self.buttons.addWidget(self.add)

        self.remove = QPushButton(self.query_buttons)
        self.remove.setObjectName(u"remove")
        self.remove.setMinimumSize(QSize(0, 30))
        self.remove.setMaximumSize(QSize(16777215, 16777215))
        self.remove.setFont(font2)
        self.remove.setStyleSheet(u"background-color: rgb(255, 90, 90);")

        self.buttons.addWidget(self.remove)


        self.variables_layout.addWidget(self.query_buttons)


        self.horizontalLayout.addWidget(self.variables)

        self.query_boxes = QVBoxLayout()
        self.query_boxes.setObjectName(u"query_boxes")
        self.query_boxes.setContentsMargins(-1, 0, -1, 20)
        self.search_lab = QLabel(first_option)
        self.search_lab.setObjectName(u"search_lab")
        self.search_lab.setFont(font)
        self.search_lab.setTextFormat(Qt.TextFormat.AutoText)
        self.search_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.search_lab.setOpenExternalLinks(True)

        self.query_boxes.addWidget(self.search_lab)

        self.query_box = QPlainTextEdit(first_option)
        self.query_box.setObjectName(u"query_box")

        self.query_boxes.addWidget(self.query_box)

        self.space = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.query_boxes.addItem(self.space)

        self.params_lab = QLabel(first_option)
        self.params_lab.setObjectName(u"params_lab")
        self.params_lab.setFont(font)
        self.params_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.params_lab.setOpenExternalLinks(True)

        self.query_boxes.addWidget(self.params_lab)

        self.params_box = QPlainTextEdit(first_option)
        self.params_box.setObjectName(u"params_box")

        self.query_boxes.addWidget(self.params_box)


        self.horizontalLayout.addLayout(self.query_boxes)

        QWidget.setTabOrder(self.from_edit, self.to_edit)
        QWidget.setTabOrder(self.to_edit, self.directory_edit)
        QWidget.setTabOrder(self.directory_edit, self.sample_edit)
        QWidget.setTabOrder(self.sample_edit, self.validation_edit)
        QWidget.setTabOrder(self.validation_edit, self.query_box)
        QWidget.setTabOrder(self.query_box, self.params_box)
        QWidget.setTabOrder(self.params_box, self.add)
        QWidget.setTabOrder(self.add, self.remove)

        self.retranslateUi(first_option)

        QMetaObject.connectSlotsByName(first_option)
    # setupUi

    def retranslateUi(self, first_option):
        first_option.setWindowTitle(QCoreApplication.translate("first_option", u"Form", None))
#if QT_CONFIG(tooltip)
        self.edit_grid.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.edit_grid.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.from_lab.setText(QCoreApplication.translate("first_option", u"From", None))
        self.col_5.setText(QCoreApplication.translate("first_option", u":", None))
#if QT_CONFIG(tooltip)
        self.sample_edit.setToolTip(QCoreApplication.translate("first_option", u"Percentage of papers queried assigned to the AI (0.--)", None))
#endif // QT_CONFIG(tooltip)
        self.sample_edit.setText(QCoreApplication.translate("first_option", u"0.2", None))
        self.col_4.setText(QCoreApplication.translate("first_option", u":", None))
#if QT_CONFIG(tooltip)
        self.from_edit.setToolTip(QCoreApplication.translate("first_option", u"Starting date in years (Included)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.from_edit.setStatusTip("")
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        self.from_edit.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.from_edit.setInputMask("")
        self.from_edit.setText(QCoreApplication.translate("first_option", u"2020", None))
        self.from_edit.setPlaceholderText("")
        self.dir_lab.setText(QCoreApplication.translate("first_option", u"Directory", None))
        self.col_3.setText(QCoreApplication.translate("first_option", u":", None))
        self.validation_lab.setText(QCoreApplication.translate("first_option", u"Validation Size", None))
#if QT_CONFIG(tooltip)
        self.validation_edit.setToolTip(QCoreApplication.translate("first_option", u"Percentage of papers queried assigned for corroboration (0.--)", None))
#endif // QT_CONFIG(tooltip)
        self.validation_edit.setText(QCoreApplication.translate("first_option", u"0.1", None))
        self.validation_edit.setPlaceholderText("")
        self.sample_lab.setText(QCoreApplication.translate("first_option", u"Sample Size", None))
        self.col_1.setText(QCoreApplication.translate("first_option", u":", None))
        self.to_lab.setText(QCoreApplication.translate("first_option", u"To", None))
        self.col_2.setText(QCoreApplication.translate("first_option", u":", None))
#if QT_CONFIG(tooltip)
        self.to_edit.setToolTip(QCoreApplication.translate("first_option", u"Finishing date in years (Excluded)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.to_edit.setStatusTip("")
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        self.to_edit.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.to_edit.setInputMask("")
        self.to_edit.setText(QCoreApplication.translate("first_option", u"2021", None))
        self.to_edit.setPlaceholderText("")
#if QT_CONFIG(tooltip)
        self.directory_edit.setToolTip(QCoreApplication.translate("first_option", u"Target directory of query", None))
#endif // QT_CONFIG(tooltip)
        self.directory_edit.setPlaceholderText(QCoreApplication.translate("first_option", u"~/.acas/query", None))
        self.add.setText(QCoreApplication.translate("first_option", u"Add", None))
        self.remove.setText(QCoreApplication.translate("first_option", u"Remove", None))
        self.search_lab.setText(QCoreApplication.translate("first_option", u"<a href=\"https://dev.elsevier.com/sc_search_tips.html\">Query Search</a>", None))
#if QT_CONFIG(tooltip)
        self.query_box.setToolTip(QCoreApplication.translate("first_option", u"Querying parameters for Scopus", None))
#endif // QT_CONFIG(tooltip)
        self.query_box.setPlainText(QCoreApplication.translate("first_option", u"TITLE-ABS-KEY(\"\")", None))
        self.params_lab.setText(QCoreApplication.translate("first_option", u"<a href=\"https://dev.elsevier.com/documentation/ScopusSearchAPI.wadl\">Additional Parameters</a>", None))
#if QT_CONFIG(tooltip)
        self.params_box.setToolTip(QCoreApplication.translate("first_option", u"Additional Parameters (in HTTP request format)", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi


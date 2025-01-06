# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'find.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDateEdit, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_mainwindow(object):
    def setupUi(self, mainwindow):
        if not mainwindow.objectName():
            mainwindow.setObjectName(u"mainwindow")
        mainwindow.resize(470, 350)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(mainwindow.sizePolicy().hasHeightForWidth())
        mainwindow.setSizePolicy(sizePolicy)
        mainwindow.setMinimumSize(QSize(470, 350))
        mainwindow.setMaximumSize(QSize(470, 350))
        self.verticalLayout = QVBoxLayout(mainwindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.search_lab = QLabel(mainwindow)
        self.search_lab.setObjectName(u"search_lab")
        self.search_lab.setMinimumSize(QSize(0, 40))
        self.search_lab.setMaximumSize(QSize(16777215, 40))
        self.search_lab.setSizeIncrement(QSize(0, 0))
        font = QFont()
        font.setFamilies([u"Open Sans"])
        font.setPointSize(12)
        self.search_lab.setFont(font)
        self.search_lab.setStyleSheet(u"border: 1px solid grey;")
        self.search_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.search_lab)

        self.select_grid = QGridLayout()
        self.select_grid.setObjectName(u"select_grid")
        self.title_regex = QPushButton(mainwindow)
        self.title_regex.setObjectName(u"title_regex")
        self.title_regex.setCheckable(True)

        self.select_grid.addWidget(self.title_regex, 0, 3, 1, 1)

        self.journal_edit = QLineEdit(mainwindow)
        self.journal_edit.setObjectName(u"journal_edit")

        self.select_grid.addWidget(self.journal_edit, 1, 2, 1, 1)

        self.label = QLabel(mainwindow)
        self.label.setObjectName(u"label")
        font1 = QFont()
        font1.setFamilies([u"Open Sans"])
        font1.setPointSize(10)
        font1.setItalic(True)
        self.label.setFont(font1)

        self.select_grid.addWidget(self.label, 4, 0, 1, 1)

        self.doi = QLabel(mainwindow)
        self.doi.setObjectName(u"doi")
        self.doi.setFont(font1)

        self.select_grid.addWidget(self.doi, 3, 0, 1, 1)

        self.journal = QLabel(mainwindow)
        self.journal.setObjectName(u"journal")
        self.journal.setFont(font1)

        self.select_grid.addWidget(self.journal, 1, 0, 1, 1)

        self.title_edit = QLineEdit(mainwindow)
        self.title_edit.setObjectName(u"title_edit")

        self.select_grid.addWidget(self.title_edit, 0, 2, 1, 1)

        self.label_box = QComboBox(mainwindow)
        self.label_box.setObjectName(u"label_box")

        self.select_grid.addWidget(self.label_box, 4, 2, 1, 1)

        self.journal_regex = QPushButton(mainwindow)
        self.journal_regex.setObjectName(u"journal_regex")
        self.journal_regex.setCheckable(True)

        self.select_grid.addWidget(self.journal_regex, 1, 3, 1, 1)

        self.col_3 = QLabel(mainwindow)
        self.col_3.setObjectName(u"col_3")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.col_3.sizePolicy().hasHeightForWidth())
        self.col_3.setSizePolicy(sizePolicy1)
        self.col_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_grid.addWidget(self.col_3, 3, 1, 1, 1)

        self.doi_edit = QLineEdit(mainwindow)
        self.doi_edit.setObjectName(u"doi_edit")

        self.select_grid.addWidget(self.doi_edit, 3, 2, 1, 1)

        self.title = QLabel(mainwindow)
        self.title.setObjectName(u"title")
        self.title.setFont(font1)

        self.select_grid.addWidget(self.title, 0, 0, 1, 1)

        self.col_1 = QLabel(mainwindow)
        self.col_1.setObjectName(u"col_1")
        sizePolicy1.setHeightForWidth(self.col_1.sizePolicy().hasHeightForWidth())
        self.col_1.setSizePolicy(sizePolicy1)
        self.col_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_grid.addWidget(self.col_1, 0, 1, 1, 1)

        self.col_2 = QLabel(mainwindow)
        self.col_2.setObjectName(u"col_2")
        sizePolicy1.setHeightForWidth(self.col_2.sizePolicy().hasHeightForWidth())
        self.col_2.setSizePolicy(sizePolicy1)
        self.col_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_grid.addWidget(self.col_2, 1, 1, 1, 1)

        self.col_4 = QLabel(mainwindow)
        self.col_4.setObjectName(u"col_4")
        sizePolicy1.setHeightForWidth(self.col_4.sizePolicy().hasHeightForWidth())
        self.col_4.setSizePolicy(sizePolicy1)
        self.col_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_grid.addWidget(self.col_4, 4, 1, 1, 1)

        self.date_range = QHBoxLayout()
        self.date_range.setObjectName(u"date_range")
        self.date_range.setContentsMargins(10, -1, -1, -1)
        self.from_lab = QLabel(mainwindow)
        self.from_lab.setObjectName(u"from_lab")
        font2 = QFont()
        font2.setFamilies([u"OCR A"])
        font2.setItalic(False)
        font2.setUnderline(False)
        self.from_lab.setFont(font2)

        self.date_range.addWidget(self.from_lab)

        self.from_date = QDateEdit(mainwindow)
        self.from_date.setObjectName(u"from_date")

        self.date_range.addWidget(self.from_date)

        self.to_lab = QLabel(mainwindow)
        self.to_lab.setObjectName(u"to_lab")
        font3 = QFont()
        font3.setFamilies([u"OCR A"])
        self.to_lab.setFont(font3)

        self.date_range.addWidget(self.to_lab)

        self.to_date = QDateEdit(mainwindow)
        self.to_date.setObjectName(u"to_date")

        self.date_range.addWidget(self.to_date)


        self.select_grid.addLayout(self.date_range, 2, 2, 1, 1)

        self.date = QLabel(mainwindow)
        self.date.setObjectName(u"date")
        self.date.setFont(font1)

        self.select_grid.addWidget(self.date, 2, 0, 1, 1)

        self.col_5 = QLabel(mainwindow)
        self.col_5.setObjectName(u"col_5")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.col_5.sizePolicy().hasHeightForWidth())
        self.col_5.setSizePolicy(sizePolicy2)
        self.col_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_grid.addWidget(self.col_5, 2, 1, 1, 1)

        self.find = QPushButton(mainwindow)
        self.find.setObjectName(u"find")
        self.find.setStyleSheet(u"background-color: rgb(115, 255, 115);")

        self.select_grid.addWidget(self.find, 2, 3, 1, 1)

        self.clear_edits = QPushButton(mainwindow)
        self.clear_edits.setObjectName(u"clear_edits")
        self.clear_edits.setStyleSheet(u"background-color: rgb(255, 95, 95);")

        self.select_grid.addWidget(self.clear_edits, 3, 3, 1, 1)

        self.select_grid.setRowMinimumHeight(0, 40)
        self.select_grid.setRowMinimumHeight(1, 40)
        self.select_grid.setRowMinimumHeight(2, 40)
        self.select_grid.setRowMinimumHeight(3, 40)
        self.select_grid.setRowMinimumHeight(4, 40)

        self.verticalLayout.addLayout(self.select_grid)

        self.clear_search = QPushButton(mainwindow)
        self.clear_search.setObjectName(u"clear_search")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.clear_search.sizePolicy().hasHeightForWidth())
        self.clear_search.setSizePolicy(sizePolicy3)
        self.clear_search.setMinimumSize(QSize(0, 40))
        font4 = QFont()
        font4.setFamilies([u"Open Sans"])
        font4.setPointSize(11)
        self.clear_search.setFont(font4)
        self.clear_search.setAutoFillBackground(False)
        self.clear_search.setStyleSheet(u"background-color: rgb(255, 90, 90);")

        self.verticalLayout.addWidget(self.clear_search)


        self.retranslateUi(mainwindow)

        QMetaObject.connectSlotsByName(mainwindow)
    # setupUi

    def retranslateUi(self, mainwindow):
        mainwindow.setWindowTitle(QCoreApplication.translate("mainwindow", u"Searching Window", None))
        self.search_lab.setText(QCoreApplication.translate("mainwindow", u"Search Documents", None))
        self.title_regex.setText(QCoreApplication.translate("mainwindow", u"Regex", None))
        self.label.setText(QCoreApplication.translate("mainwindow", u"Label", None))
        self.doi.setText(QCoreApplication.translate("mainwindow", u"DOI", None))
        self.journal.setText(QCoreApplication.translate("mainwindow", u"Journal", None))
        self.journal_regex.setText(QCoreApplication.translate("mainwindow", u"Regex", None))
        self.col_3.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.title.setText(QCoreApplication.translate("mainwindow", u"Title", None))
        self.col_1.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.col_2.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.col_4.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.from_lab.setText(QCoreApplication.translate("mainwindow", u"From:", None))
        self.to_lab.setText(QCoreApplication.translate("mainwindow", u"To:", None))
        self.date.setText(QCoreApplication.translate("mainwindow", u"Date", None))
        self.col_5.setText(QCoreApplication.translate("mainwindow", u":", None))
        self.find.setText(QCoreApplication.translate("mainwindow", u"Find", None))
        self.clear_edits.setText(QCoreApplication.translate("mainwindow", u"Clear All", None))
        self.clear_search.setText(QCoreApplication.translate("mainwindow", u"Remove from search", None))
    # retranslateUi


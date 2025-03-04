#!/usr/bin/env python3
"""
File that is used to test whether the data manipulation between classes
actually works as intended.

@author  Thomas Gauthier
@version 0.0
"""
# Error with Pixmap. Must initialize a QMainApplication before creating instances
# Of the QWidget and QDialog classes.
from PySide6.QtWidgets import QApplication
ignored: QApplication = QApplication()

from typing import Self

from python.src.utils.files import Paper
from ui.windows             import Data, Loading

import datetime as dt
import unittest as uni
import os.path  as osp

import os

"""
Test class for stemming words.

@author  Thomas Gauthier
@version 0.0
"""
class Stemming(uni.TestCase):
    def test_stems(self: Self) -> None:
        samples: list[Paper] = [
            Paper(
                "Title",
                "",
                "Journal",
                None,
                "",
                None,
                None,
            ),
            Paper(
                "Title, Title Title.",
                "Abstract! Abstract?, Eating",
                "Don't, Do 42 not, Do 42",
                None,
                "",
                None,
                None
            )
        ]

        load: Loading = Loading(samples, min_words=2)
        # Note that this function will write files onto your computer
        load.getStems()

        # Given that there may be some stem lists that the user wants to keep, this test would only work if the directory did not previously exist
        self.assertTrue(osp.exists(Loading.DEFAULT_WRITE) and osp.isdir(Loading.DEFAULT_WRITE))

        """
        To be quite honest, there isn't anything more we can do
        Other than adding a buch of print statements and see the results.
        """

"""
Class checking basic data manipulation.

@author  Thomas Gauthier
@version 0.0
"""
class Manipulation(uni.TestCase):
    def setUp(self: Self) -> None:
        self.data:   Data = Data()
        self.papers: list[Paper] = [
            Paper(
                "First Title TXT",
                "Abstract TXT",
                "Journal TXT",
                dt.date(2000, 1, 1),
                "Private",
                "DOI TXT"
            ),
            Paper(
                "First Title CSV",
                "First Abstract CSV",
                "First Journal CSV",
                dt.date(1900, 1, 1),
                "Private",
            ),
            Paper(
                "Second,\\n Title \"CSV",
                "Second Abstract CSV",
                "Second Journal CSV",
                dt.date(1900, 1, 2),
                "Private",
                "DOI CSV"
            )
        ]

    def test_add(self: Self) -> None:
        training: bool = False
        main:     bool = False

        def call() -> None:
            nonlocal training
            training = True

        def callM() -> None:
            nonlocal main
            main = True

        self.data.dataset[0].connect(callM)
        self.data.training[0].connect(call)

        self.data.add(text="Training", path=(osp.join("tests", "datasets", "valid")))

        self.assertTrue(main)
        self.assertTrue(training)

        for paper in self.data.dataset[1]:
            self.assertEqual(paper[0], self.data.training)
            init: bool = False
            for has in self.papers:
                init |= has == paper[1]
            self.assertTrue(init)

        for paper in self.data.training[1]:
            init: bool = False
            for has in self.papers:
                init |= has == paper
            self.assertTrue(init)

    def test_remove(self: Self) -> None:
        self.data.dataset[1].append(None, self.papers[1])
        self.data.dataset[1].append(None, self.papers[0])
        self.data.dataset[1].append(self.data.validation, self.papers[-1])
        self.data.validation[1].append(self.papers[-1])

        validation: bool = False
        main:       bool = False

        def call() -> None:
            nonlocal validation
            validation = True

        def callM() -> None:
            nonlocal main
            main = True

        self.data.dataset[0].connect(callM)
        self.data.validation[0].connect(call)

        self.data.remove(path=osp.join("tests", "datasets", "valid"))

        self.assertTrue(main)
        self.assertTrue(validation)

        self.assertFalse(self.data.dataset[1])
        self.assertFalse(self.data.validation[1])

if __name__ == "__main__":
    uni.main()
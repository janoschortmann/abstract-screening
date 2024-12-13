#!/usr/bin/env python3
"""
Main file setting up the UI.

@author  Thomas Gauthier
@version 0.0
"""
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, uic
from typing import Self

from python.src.utils.files import File

# Done to access all newer features of the typing library such as generics
if sys.version_info > (3, 12):
    sys.stderr.write("Python version inferior to 3.12+. Please make sure to update to a newer version.")
    sys.exit()

"""
--------------------------------------------------
                Global Constants
--------------------------------------------------
"""
APPLICATION_NAME: str = "Abstract Screening"
APPLICATION_UI:   str = "./ui/mainwindow.ui"

VIEW_OPTIONS: list[str] = [
    "Marked",
    "Unmarked",
    "Accepted",
    "Rejected"
]

"""
Main window containing the UI and setting up the signals.

@author  Thomas Gauthier
@version 0.0
"""
class MainWindow(QMainWindow):
    # The initializer of the window.
    # Will search for the XML file from the constant APPLICATION_UI
    def __init__(self: Self, *argv: list[str]) -> None:
        # Setting up the UI
        super().__init__()

        # Todo : Connect the main window with step 1
        # Todo : Connect the signals/callbacks

        self.show()

if __name__ == "__main__":
    app  = QApplication([])
    main = MainWindow()
    app.exec()

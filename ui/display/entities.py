#!/usr/bin/env python3
"""
The main document regrouping the elements which were impossible
to fully code in the Qt Designer.

This include, for example, the table of step one (or from the data button)
which shows each paper. Another example would be the Operating Characteristic Curve.

@author  Thomas Gauthier
@version 0.1
"""
from multiprocessing   import Pool
from pyqtgraph         import PlotWidget, mkPen, QtGui
from PySide6.QtCore    import (
                                QAbstractTableModel,
                                QModelIndex,
                                QModelRoleData,
                                QObject,
                                Qt
                              )
from PySide6.QtWidgets import QMessageBox, QWidget, QTableView, QVBoxLayout
from scipy.stats       import binom
from threading         import Lock
from typing            import Any, Callable, Final, Self, override, final

from python.src.utils.files import Paper
from python.src.utils.functions import clear_empty, transform_as_dict, unique
from ui.resources_loader import *

import datetime as dt
import numpy as np
import re

# Aliases
data_role   = Qt.ItemDataRole
item_flags  = Qt.ItemFlag
align_flags = Qt.AlignmentFlag

"""
Basic class that represent the skeleton of all model table model used.

T must be an instance of the Paper class.

@author  Thomas Gauthier
@version 0.1
"""
class PaperTableModel[T](QAbstractTableModel):
    # Default initializer that assigns the data received and the columns shown
    # Please note that the data is not copied but referenced. Thus, it is assumed
    # That the original instance won't change. This is done to be more efficient,
    # But puts the responsability on the user to not fuck this up.
    def __init__(self: Self, data: list[T] | None, columns: list[str] | tuple[str], headers: list[str] | tuple[str]) -> None:
        super().__init__()

        if not isinstance(T, Paper):
            raise TypeError("T is not from the Paper class")

        if len(headers) != len(columns):
            raise Exception("List of different sizes")

        self.__columns: Final[list[str]] | tuple[str] = columns.copy()
        self.__headers: Final[list[str]] | tuple[str] = headers.copy()
        self.__data: list[T] = data if data is not None else []

    def rowCount(self: Self) -> int:
        return len(self.__data)

    def columnCount(self: Self) -> int:
        return len(self.__columns)

    # All data will be aligned in the center
    def headerData(self: Self, index: int, orientation: Qt.Orientation, role: data_role) -> Any:
        if role == data_role.TextAlignmentRole:
            return align_flags.AlignCenter

        if role == data_role.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.__headers[index]

        return None

    # All data will be aligned in the center
    def data(self: Self, index: QModelIndex, role: data_role) -> Any:
        if role == data_role.TextAlignmentRole:
            return align_flags.AlignCenter

        if role == data_role.DisplayRole:
            # Introspection moment
            return str(vars(self.__data[index.column()])[self.__columns[index.row()]])

"""
Default class implementing the PaperTableModel.

@author  Thomas Gauthier
@version 0.0
"""
class PaperTableView[T](QWidget):
    # Default initializer
    def __init__[**P](self: Self, data: list[T] | None, columns: list[str] | tuple[str], headers: list[str] | tuple[str], *args: P.args) -> None:
        # Variables
        super().__init__(args)
        self.model: PaperTableModel = PaperTableModel(data, columns, headers)
        self.view:       QTableView = QTableView()
        self.view.setModel(self.model)

        # Layout
        layout: QWidget = QVBoxLayout()
        layout.addLayout(self.view)
        self.setLayout(layout)

"""
Implements the PaperTableModel with barebone data manipulation.

@author  Thomas Gauthier
@version 0.0
"""
class MutableTableModel[T](PaperTableModel):
    # Default initializer
    # Here again, the data is passed by reference
    def __init__(self: Self, data: list[T] | None, columns: list[str] | tuple[str], headers: list[str] | tuple[str]) -> None:
        super().__init__(data, columns, headers)

    # Appends the new data and removes doubles
    # Note that this could also be done outside this method since the data
    # Is passed by reference. This also accepts doubles of papers
    def append(self: Self, data: list[T]) -> None:
        self.__data.append(data)
        self.dataChanged.emit()

    # Removes the targets and sends back the failures
    def remove(self: Self, removal: list[T]) -> list[T] | None:
        failures: list[T] = []

        for element in removal:
            try: self.__data.remove(element)
            except: failures.append(element)

        self.dataChanged.emit()
        return failures or None

    # Clearing
    def clear(self: Self) -> None:
        self.__data.clear()
        self.dataChanged.emit()

"""
A TableModel that implements a basic finding feature.

Since it implements the PaperTableModel, the data is passed by reference to be faster.
This implies that there may be problems with synchronisation and that is why
anyone that uses this must not manipulate the data outside this class.

More information is found in the initializer of PaperTableModel.

@author  Thomas Gauthier
@version 0.0
"""
class FindingModel[T](PaperTableModel):
    # Default initializer
    def __init__(self: Self, data: list[T] | None, columns: list[str] | tuple[str], headers: list[str] | tuple[str]) -> None:
        # Additional variables
        self.__hidden: list[T] = data
        self.__lock:      Lock = Lock()
        self.__criterias: dict[int, Callable[[T], bool]] = {}

        # Synchronizing the data
        self.hidden_changed: QObject = QObject()

        super().__init__(self.__hidden.copy(), columns, headers)

    # Updates the view
    @staticmethod
    def updates[R, **P](func: Callable[P, R]) -> Callable[P, R]:
        # Get the "self" instance
        self: FindingModel = P.args[0]
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            self.__lock.acquire()
            argument: R = func(args, kwargs)
            self.update_showing()
            self.__lock.release()
            return argument
        return inner

    # Mutates the hidden data
    @staticmethod
    def mutated[R, **P](func: Callable[P, R]) -> Callable[P, R]:
        self: FindingModel = P.args[0]
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            var: R = func(args, kwargs)
            self.hidden_changed.emit()
            return var
        return FindingModel.updates(inner)

    # Will update the showing data. Used when the real data has changed
    def update_showing(self: Self) -> None:
        self.__data: list[T] = []

        for paper in self.__hidden:
            # Could be boxed in another function, depending on future requirements
            respects_all: bool = True
            for criteria in self.__criterias.values():
                if not criteria(paper):
                    respects_all = False
                    break

            if respects_all:
                self.__data.append(paper)

        self.dataChanged.emit()


    # Function that will add the criterias for the view
    @updates
    def add_criterias(self: Self, criterias: dict[int, Callable[[T], bool]] | list[Callable[[T], bool]]) -> None:
        self.__criterias.update(transform_as_dict(criterias, mapper=id))

    @updates
    def remove_criterias(self: Self, criteria: dict[int, Callable[[T], bool]] | list[Callable[[T], bool]]) -> list[int] | None:
        keys: list[int] = []

        if isinstance(criteria, list): keys = map(id, criteria)
        else: keys = criteria.keys()

        failures: list[int] = []
        for key in keys:
            try: self.__criterias.pop(key)
            except: failures.append(key)

        return failures or None

    @updates
    def clear_criterias(self: Self) -> None:
        self.__criterias.clear()

    # Appends new data and shows it if it respects the current criterias
    # Note that this could also be done outside this method since the data
    # Is passed by reference. This also accepts doubles
    @mutated
    def append(self: Self, data: list[T]) -> None:
        self.__hidden.extend(data)

    # Removes the elements that are the same as in the removal list
    @mutated
    def remove(self: Self, removal: list[T]) -> list[T] | None:
        failures: list[T] = []

        for element in removal:
            try: self.__hidden.remove(element)
            except: failures.append(element)

        return failures or None

    # Clear alls the data
    @mutated
    def clear(self: Self) -> None:
        self.__hidden.clear()

"""
Implements the PaperTableModel with barebone unique data manipulation.

@author  Thomas Gauthier
@version 0.0
"""
class UniqueTableModel[T](MutableTableModel):
    # Default initializer
    def __init__(self: Self, data: list[T] | None, columns: list[str] | tuple[str], headers: list[str] | tuple[str]) -> None:
        super().__init__(data, columns, headers)

    # Appends the new data and removes doubles
    @override
    def append(self: Self, data: list[T]) -> None:
        self.__data.append(data)
        unique(self.__data)
        self.dataChanged.emit()

"""
Basic class that represents a table showing the title,
journal, date, doi and the current label of some paper.

This is used for example on step one and on the data label.

When an element is clicked, it will cycle through the possible labels.

@author  Thomas Gauthier
@version 0.1
"""
@final
class SelectionModel[T](FindingModel):
    # An ordered tuple of the data shown
    columns: tuple[str] = ("title", "date", "jour", "doi", "label")
    headers: tuple[str] = ("Title", "Date", "Journal", "DOI", "Label")
    dict_images: Final[dict[int, QtGui.QIcon]] = {
        0: QtGui.QIcon(":/resources/grey_bar.png"),
        1: QtGui.QIcon(":/resources/green_checkbar.png"),
        2: QtGui.QIcon(":/resources/delete.png")
    }

    """
    Basic constructor that receives a list of the elements containing papers.
    Note that if T is not an instance of Paper, this will throw an error.
    """
    def __init__(self: Self, data: list[T]) -> None:
        super(self, data, SelectionModel.columns, SelectionModel.headers)

    # Return the flags for an item in the table
    def flags(self: Self) -> item_flags:
        return item_flags.ItemIsEditable

    # Basic function used to return layout information
    @override
    def data(self: Self, index: QModelIndex, role: QModelRoleData) -> Any:
        if index.column() != self.__columns.index("label"):
            return super().data(self, index, role)

        if role == data_role.DecorationRole:
            return SelectionModel.dict_images[self.__data[index.row()].label]

        return None

    # Default behaviour for clicking on an item
    def on_clicked(self: Self, index: QModelIndex) -> None:
        selected: T = self.__viewing[index.row()]
        selected.index = (selected.index + 1) % len(SelectionModel.dict_images)
        self.layoutChanged.emit()

"""
Basic implementation of the FindingModel as a TableView.

The argument T must be an instance of the Paper class.

@author  Thomas Gauthier
@version 0.1
"""
class FindingView[T](QWidget):
    # Basic constructor receiving data and arguments for initialization
    def __init__[**P](self: Self, data: list[T], model: FindingModel = FindingModel, *args: P.args) -> None:
        # Default initialization
        super().__init__(self, *args)
        self.table_view: QTableView = QTableView()
        self.model: SelectionModel  = model(data)
        self.table_view.setModel(self.model)

        # Connecting
        layout: QWidget = QVBoxLayout(self)
        layout.addLayout(self.table_view)
        self.setLayout(layout)

    """
    Basic function that will change the data being dispayed
    by the hits it will find given the arguments.

    Note that, by default, it will search for a string
    that is defined by the regex *arg* (verbose), but, if the boolean in the tuple
    is set to True, then it will consider the string as a regex.
    """
    def find(
             self: Self,
             title: tuple[str, bool] | None,
             journal: tuple[str, bool] | None,
             date: tuple[dt.date, dt.date] | None,
             doi: str | None,
             label: str | None
            ) -> None:
        if label not in Paper.POSSIBILITIES: raise Exception("Not in possibilities")

        # Initialization
        self.model.clear_criterias()
        shortcut: Callable[[T], re.Pattern] = lambda arg: re.compile(arg[0] if arg[1] else re.escape(arg[0]))
        every: list[Callable[[T], bool] | None] = [
            (lambda val: (lambda paper: val.match(paper.title)))(shortcut(title)) if not isinstance(title, None) else None,
            (lambda val: (lambda paper: val.match(paper.journal)))(shortcut(journal)) if not isinstance(journal, None) else None,
            (lambda paper: (date[0] <= paper.date and paper.date <= date[1])) if not isinstance(date, None) else None,
            (lambda paper: (paper.doi == doi)) if not isinstance(doi, None) else None,
        ]
        if label is not None and label == Paper.POSSIBILITIES[0]: every.append(lambda paper: paper.labeled())
        else: every.append((lambda paper: (paper.label == label)) if not isinstance(label, None) else None)

        clear_empty(every)
        self.model.add_criterias(every)

    """
    Basic function that will remove all the files that respect the findings.

    For more information on how it finds these papers, please go check
    the "find" method.
    """
    def remove_found(
                     self: Self,
                     title: tuple[str, bool] | None,
                     journal: tuple[str, bool] | None,
                     date: tuple[dt.date, dt.date] | None,
                     doi: str | None,
                     label: int | None
                    ) -> None:
        self.find(title, journal, date, doi, label)
        # HACK : THIS IS UNSAFE, BUT NECESSARY
        self.model.remove(self.model.__data.copy())

"""
Basic graph using pyqtgraph as instructed in the pdf.

It draws the operating characteristic curve based on the parameters
and a few lines to represent them.

@author  Thomas Gauthier
@version 0.0
"""
@final
class OperatingCurve(QWidget):
    # Number of points shown
    """
    Todo : If necessary, change this to be a constant based on the maximal width of the screens
    Such as, for example (Java PseudoCode):
    final int points = Math.round(screens.stream().map(screen::getWidth).max() * CONSTANT);
    Or something like that...
    """
    POINTS:     Final[int]   = 500
    # Threshold to stop the search for the nc parameters. Will throw an exception pass that point
    THRESHOLD:  Final[int]   = 1000
    # Number of processes in the pool
    POOL_COUNT: Final[int]   = 4

    # Default initializer that only set the basic themes
    def __init__(self: Self) -> None:
        super().__init__()
        # Plot theme
        self.plot: PlotWidget = PlotWidget()
        self.pen:  QtGui.QPen = mkPen(color = 'b', width = 5, style = Qt.PenStyle.SolidLine)

        self.plot.setBackground('w')
        self.plot.plotItem.setLabel("left", "Probability of acceptance")
        self.plot.plotItem.setLabel("bottom", "Fraction defective")

        # Layout
        layout = QVBoxLayout(self)
        layout.addLayout(self.plot)
        self.setLayout(layout)

    # Setter for the attributes of the characteristic function
    def set_attributes(
                        self: Self,
                        alpha:  float,
                        beta:   float,
                        param1: float,
                        param2: float
                      ) -> None:
        self.alpha:  float = alpha
        self.beta:   float = beta
        self.param1: float = param1
        self.param2: float = param2

    """
    Default method used to plot the Operating curve.

    Note that there is a constant number of points, since recalculating
    some lot of given points each time the window is resized gets expensive pretty quickly.

    Note that the amount of points plotted follows the resolution of the biggest screen.
    """
    def plot(self: Self) -> None:
        # Assuming that the self.nc variables is not None
        if self.nc is None:
            raise TypeError("The N and C parameters do not exist")

        p_values: np.ndarray[np.floating[Any]] = np.linspace(0, 0.5, OperatingCurve.POINTS)
        probabilities: list[float] = [OperatingCurve.probability_of_acceptance(self.nc[0], self.nc[1], p) for p in p_values]

        # Plotting the function and the lines
        self.plot.plotItem.plot(p_values, probabilities, pen=self.pen)
        self.plot.plotItem.addLegend()

        # Lines representing the current values used
        producer: float = 1 - self.alpha
        self.plot.plotItem.addLine(
                                    name = "Producer: 1 - alpha (" + str(producer) + ')',
                                    y = producer,
                                    pen = mkPen(hsv = (20, 85, 95), width = 0.5, style = Qt.PenStyle.DashLine)
                                  )

        self.plot.plotItem.addLine(
                                    name = "Consumer: beta (" + str(self.beta) + ')',
                                    y = self.beta,
                                    pen = mkPen(color = 'b', width = 0.5, style = Qt.PenStyle.DashLine)
                                  )

        self.plot.plotItem.addLine(
                                    name = "Parameter 1: " + str(self.param1),
                                    x = self.param1,
                                    pen = mkPen(color = 'g', width = 0.5, style = Qt.PenStyle.DashLine)
                                  )

        self.plot.plotItem.addLine(
                                    name = "Parameter 2: " + str(self.param2),
                                    x = self.param2,
                                    pen = mkPen(color = 'r', width = 0.5, style = Qt.PenStyle.DashLine)
                                  )

        # Other formatting
        self.plot.plotItem.showGrid(x = True, y = True)
        self.plot.plotItem.legend.addItem(None, "Number of samples: " + str(self.nc[0]))
        self.plot.plotItem.legend.addItem(None, "Minimal number of acceptance: " + str(self.nc[1]))

    """
    Method used to find the number of sample required and the
    minimum count so that the batch will be accepted.

    Note that there does not exist an explicit (as far as I know) formula
    for the nc parameters since they are distributed in a
    [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution).
    Hence, why the brute force approach.

    For more information on this method, please consult the requirements file.
    """
    def find_nc(self: Self) -> None:
        lock: Lock = Lock()

        # Inner function used by the instances of the pool
        # In most cases, it will find the smallest result
        # And the trade for speed is worth the while
        def inner(pool: Any, initial: int) -> None:
            counter: int = initial

            while counter <= OperatingCurve.THRESHOLD:
                for c in range(counter):
                    # Calculate producer's risk
                    producer: float = sum([binom.pmf(k, counter, self.param1) for k in range(c + 1)])
                    # Calculate consumer's risk
                    consumer: float = sum([binom.pmf(k, counter, self.param2) for k in range(c + 1)])

                    if producer >= 1 - self.alpha and consumer <= self.beta:
                        # If two results are possible, lock one for the pool to terminate
                        lock.acquire()
                        self.nc = [counter, c]
                        pool.terminate()

                    counter += OperatingCurve.POOL_COUNT

        with Pool(OperatingCurve.POOL_COUNT) as pool:
            for i in range(1, OperatingCurve.POOL_COUNT):
                pool.apply(inner, args = (pool, i))

        lock.release()

    """
    Linear method of find_nc. For more information, go see find_nc.

    This method is deprecated.
    """
    def linear_find_nc(self: Self) -> None:
        for n in range(1, OperatingCurve.THRESHOLD):
            for c in range(n):
                # Calculate producer's risk
                producer: float = sum([binom.pmf(k, n, self.param1) for k in range(c + 1)])
                # Calculate consumer's risk
                consumer: float = sum([binom.pmf(k, n, self.param2) for k in range(c + 1)])

                if producer >= 1 - self.alpha and consumer <= self.beta:
                    self.nc = [n, c]
                    return

    # Method used to get the points
    @staticmethod
    def probability_of_acceptance(n: int, c: int, p: float) -> float:
        return sum([binom.pmf(k, n, p) for k in range(c + 1)])

"""
A factory method that will return a QMessageBox
for some error based on client input.

@author  Thomas Gauthier
@version 0.0
"""
def error_factory(name: str, message: str, parent: QWidget | None = None) -> QMessageBox:
    return mb_factory(name, message, QMessageBox.Icon.Critical, QMessageBox.StandardButton.Ok, parent)

"""
A factory method that will return a QMessageBox
with the given parameters.

@author  Thomas Gauthier
@version 0.0
"""
def mb_factory(
                name: str,
                message: str,
                icon: QMessageBox.Icon,
                buttons: QMessageBox.StandardButton,
                parent: QWidget | None = None
              ) -> QMessageBox:
    mb: QMessageBox = QMessageBox(parent)
    mb.setWindowTitle(name)
    mb.setText(message)
    mb.setStandardButtons(buttons)
    mb.setIcon(icon)
    return mb
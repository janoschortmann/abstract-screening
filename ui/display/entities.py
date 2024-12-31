#!/usr/bin/env python3
"""
The main document regrouping the elements which were impossible
to fully code in the Qt Designer.

This include, for example, the table of step one (or from the data button)
which shows each paper. Another example would be the Operating Characteristic Curve.

@author  Thomas Gauthier
@version 0.0
"""
from multiprocessing   import Pool, Lock
from pyqtgraph         import PlotWidget, mkPen, QtGui
from PySide6.QtCore    import (
                                QAbstractTableModel,
                                QIcon,
                                QModelIndex,
                                QModelRoleData,
                                Qt
                              )
from PySide6.QtWidgets import QWidget, QTableView, QVBoxLayout
from scipy.stats       import binom
from typing            import Any, Callable, Final, Self, final

from ...python.src.utils.files import Paper
from ...python.src.utils.functions import clear_empty, transform_as_dict, unique
from ..resources_loader import *

import datetime as dt
import numpy as np
import re

# Aliases
data_role   = Qt.ItemDataRole
item_flags  = Qt.ItemFlag
align_flags = Qt.AlignmentFlag

"""
Basic class that represents a table showing the title,
journal, date, doi and the current label of some paper.

This is used for example on step one and on the data label.

When an element is clicked, it will cycle through the possible labels.

@author  Thomas Gauthier
@version 0.0
"""
# Fixme : If needed, change the data to a numpy array if it's too slow
@final
class PaperModel[T](QAbstractTableModel):
    # An ordered tuple of the data shown
    column_order: tuple[str] = ("Title", "Journal", "Date", "DOI", "Label")
    column_size: Final[int] = len(column_order)

    dict_images: Final[dict[int, QIcon]] = {
        0: QIcon(":/resources/grey_bar.png"),
        1: QIcon(":/resources/green_checkbar.png"),
        2: QIcon(":/resources/delete.png")
    }

    """
    Basic constructor that receives a list of the elements containing papers.
    Note that if T is not an instance of Paper, this will throw an error.

    Please be aware that the data will be copied and, therefore, may have
    a significant runtime cost depending on its application.
    """
    def __init__(self: Self, data: list[T]) -> None:
        if not isinstance(T, Paper):
            raise TypeError("T is not from the Paper class")

        # Initializers
        super().__init__()
        self.__criterias: dict[int, Callable[[T], bool]] = {}
        self.__data: list[T] = []
        # Will initially have the same pointer as the data for optimization
        # Purposes, but, when using the "find" functionality, it will create another
        # Reference to not modity the data behind the table
        self.__viewing: list[T] = self.__data

        # Is used instead of direct assignment to also clear doubles
        self.append(data)

    # Returns four each time since this table only shows the title, journal, doi and label
    def rowCount(self: Self, index: QModelIndex = None) -> int:
        return PaperModel.column_size

    # Returns the current number of papers that are designed to be shown
    def columnCount(self: Self, index: QModelIndex = None) -> int:
        return len(self.__viewing[0])

    # Shows the column names/row names
    def headerData(self: Self, section: int, orientation: Qt.Orientation, role: data_role) -> None:
        if role == data_role.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return PaperModel.column_order[section]

    # Return the flags for an item in the table
    def flags(self: Self) -> item_flags:
        return item_flags.ItemIsEditable

    # Basic function used to return layout information
    def data(self: Self, index: QModelIndex, role: QModelRoleData) -> Any:
        # Centers the data by default
        if role == data_role.TextAlignmentRole:
            return align_flags.AlignCenter

        label_column: Final[bool] = index.column() == PaperModel.column_order.index("Label")
        selected_paper: Final[Paper] = self.__viewing[index.row()]

        # More modulable if the tuple changes
        if role == data_role.DisplayRole and not label_column:
            column_name: str = PaperModel.column_order[index.column()]

            # Fixme : If it takes too much time to load, change this to not have introspection
            # This was done to be more modulable, but isn't strictly necessary
            for name in PaperModel.column_order:
                if column_name == name:
                    return vars(selected_paper)[name]


        if role == data_role.DecorationRole and label_column:
            return PaperModel.dict_images[selected_paper.label]

        return None

    # Function that declares that a method should update the data that must be shown
    def updates[R, **P](self: Self, func: Callable[P, R]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            argument: R = func(args, kwargs)
            self.update_showing()
            return argument
        return inner

    # Will update the showing data. Used when the real data has changed
    # Todo : Add asynchronous task to this
    def update_showing(self: Self) -> None:
        self.__viewing: list[T] = []

        for paper in self.__data:
            # Could be boxed in another function, depending on future requirements
            respects_all: bool = True
            for criteria in self.__criterias.values():
                if not criteria(paper):
                    respects_all = False
                    break

            if respects_all:
                self.__viewing.append(paper)

        self.dataChanged.emit()


    # Function that will add the criterias for the view
    @updates
    def add_criterias(self: Self, criterias: dict[int, Callable[[T], bool]] | list[Callable[[T], bool]]) -> None:
        self.__criterias.update(transform_as_dict(criterias, mapper=id))

    @updates
    def remove_criterias(self: Self, criteria: dict[int, Callable[[T], bool]] | list[Callable[[T], bool]]) -> list[int] | None:
        keys: list[int] = []

        if isinstance(criteria, list):
            keys = map(id, criteria)
        else:
            keys = criteria.keys()

        success: list[int] = []
        for key in keys:
            try:
                self.__criterias.pop(key)
            except:
                success.append(key)

        return success or None

    @updates
    def clear_criterias(self: Self) -> None:
        self.__criterias.clear()

    # Appends new data and shows it if it respects the current criterias
    @updates
    def append(self: Self, data: list[T]) -> None:
        self.__data.extend(data)
        unique(self.__data)

    # Removes the elements that are the same as in the removal list
    @updates
    def remove(self: Self, removal: list[T]) -> list[T] | None:
        success: list[T] = []

        for element in removal:
            try:
                self.__data.remove(element)
            except:
                success.append(element)

        return success or None

    # Clear alls the data
    @updates
    def clear(self: Self) -> None:
        self.__data.clear()

    # Default behaviour for clicking on an item
    def on_clicked(self: Self, index: QModelIndex) -> None:
        selected: T = self.__viewing[index.row()]
        selected.index = (selected.index + 1) % len(PaperModel.dict_images)
        self.layoutChanged.emit()

"""
Basic implementation of the PaperModel as a TableView.

@author  Thomas Gauthier
@version 0.0
"""
class PaperView[T](QWidget):
    # Basic constructor receiving data and arguments for initialization
    def __init__[**P](self: Self, data: list[T], *args: P.args) -> None:
        # Default initialization
        super().__init__(self, *args)
        self.table_view: QTableView     = QTableView()
        self.model: PaperModel          = PaperModel(data)

        # Callbacks
        self.table_view.setModel(self.model)
        self.table_view.clicked.connect(self.model.on_clicked)

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

    @author  Thomas Gauthier
    @version 0.0
    """
    def find(
            self: Self,
            title: tuple[str, bool] | None,
            journal: tuple[str, bool] | None,
            date: tuple[dt.date, dt.date] | None,
            doi: str | None,
            label: int | None
            ) -> None:
        # Initialization
        self.model.clear_criterias()
        shortcut: Callable[[T], re.Pattern] = lambda arg: re.compile(arg[0] if arg[1] else re.escape(arg[0]))
        every: list[Callable[[T], bool] | None] = [
            (lambda val: (lambda paper: val.match(paper.title)))(shortcut(title)) if not isinstance(title, None) else None,
            (lambda val: (lambda paper: val.match(paper.journal)))(shortcut(journal)) if not isinstance(journal, None) else None,
            (lambda paper: (date[0] <= paper.date and paper.date <= date[1])) if not isinstance(date, None) else None,
            (lambda paper: (paper.doi == doi)) if not isinstance(doi, None) else None,
            (lambda paper: (paper.label == label)) if not isinstance(label, None) else None
        ]
        clear_empty(every)
        self.model.add_criterias(every)

"""
Basic graph using pyqtgraph as instructed in the manual.

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
    Or something as such
    """
    POINTS:     Final[int]   = 500
    # Threshold to stop the search for the nc parameters. Will throw an exception pass that point
    THRESHOLD:  Final[int]   = 1000
    # Number of processes in the pool
    POOL_COUNT: Final[int]   = 4

    # Default initializer that only set the basic themes
    def __init__(self: Self) -> None:
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
        lock: Any = Lock()

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
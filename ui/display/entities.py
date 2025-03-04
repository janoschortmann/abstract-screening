#!/usr/bin/env python3
"""
The main document regrouping the elements which were impossible
to fully code in the Qt Designer.

This include, for example, the table from the first step (or from the data button)
which shows each paper. Another example would be the Operating Characteristic Curve.

@author  Thomas Gauthier
@version 0.3
"""
from multiprocessing      import Array, Lock, RLock, Pool
from pyqtgraph            import PlotWidget, mkPen, QtGui
from PySide6.QtCore       import (
                                   QAbstractTableModel,
                                   QEvent,
                                   QModelIndex,
                                   QModelRoleData,
                                   QTimer,
                                   Slot,
                                   Signal,
                                   Qt
                                 )
from PySide6.QtGui        import QFont
from PySide6.QtWidgets    import QMessageBox, QWidget, QTableView, QVBoxLayout
from scipy.stats          import binom
from typing               import Any, Callable, Final, Iterable, Self, override, final

from python.src.utils.files     import Paper
from python.src.utils.functions import clearEmpty, unique
from ui.resources_loader        import *

import datetime as dt
import numpy as np
import re

# Aliases
data_role   = Qt.ItemDataRole
item_flags  = Qt.ItemFlag
align_flags = Qt.AlignmentFlag

"""
Basic class that represent the skeleton of all model table model used.

@author  Thomas Gauthier
@version 0.2
"""
class PaperTableModel(QAbstractTableModel):
    # Default initializer that assigns the data received and the columns shown
    # Please note that the data is not copied but referenced. Thus, it is assumed
    # That the original instance won't change. This is done to be more efficient,
    # But puts the responsability on the programmer to not do undefined behaviour.
    def __init__(
                  self:    Self,
                  data:    list | None,
                  columns: list[str],
                  headers: list[str],
                  parent:  QWidget | None = None
                ) -> None:
        super().__init__(parent)

        if len(headers) != len(columns):
            raise Exception("List of different sizes")

        self.__columns: Final[list[str]] = columns.copy()
        self.__headers: Final[list[str]] = headers.copy()
        self.__data: list[Paper] = data if data is not None else []

    @override
    def rowCount(self: Self, index: QModelIndex) -> int:
        return len(self.__data)

    @override
    def columnCount(self: Self, index: QModelIndex) -> int:
        return len(self.__columns)

    # All data will be aligned in the center
    @override
    def headerData(self: Self, index: int, orientation: Qt.Orientation, role: data_role) -> Any:
        if role == data_role.TextAlignmentRole:
            return align_flags.AlignCenter

        if role == data_role.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.__headers[index]

        return None

    # All data will be aligned in the center
    @override
    def data(self: Self, index: QModelIndex, role: data_role) -> Any:
        if role == data_role.TextAlignmentRole:
            return align_flags.AlignCenter

        if role == data_role.DisplayRole:
            # Introspection moment
            return str(vars(self.__data[index.column()])[self.__columns[index.row()]])

"""
Default class implementing the PaperTableModel.

@author  Thomas Gauthier
@version 0.1
"""
class PaperTableView(QWidget):
    # Default initializer
    def __init__(
                  self: Self,
                  data: list[Paper] | None,
                  columns: list[str] | tuple[str],
                  headers: list[str] | tuple[str],
                  parent: QWidget | None = None
                ) -> None:
        # Variables
        super().__init__(parent)
        self.model: PaperTableModel = PaperTableModel(data, columns, headers)
        self.view: QTableView = QTableView()
        self.view.setModel(self.model)

        # Layout
        layout: QWidget = QVBoxLayout()
        layout.addLayout(self.view)
        self.setLayout(layout)

"""
Implements the PaperTableModel with barebone data manipulation.

@author  Thomas Gauthier
@version 0.1
"""
class MutableTableModel(PaperTableModel):
    # Default initializer
    # Here again, the data is passed by reference
    def __init__(
                  self: Self,
                  data: list[Paper] | None,
                  columns: list[str] | tuple[str],
                  headers: list[str] | tuple[str],
                  parent:    QWidget | None = None
                ) -> None:
        super().__init__(data, columns, headers, parent)

    # Appends the new data and removes doubles
    # Note that this could also be done outside this method since the data
    # Is passed by reference. This also accepts doubles of papers
    def append(self: Self, data: list) -> None:
        self.__data.append(data)
        self.layoutChanged.emit()

    # Removes the targets and sends back the failures
    def remove(self: Self, removal: list) -> list | None:
        failures: list = []

        for element in removal:
            try: self.__data.remove(element)
            except: failures.append(element)

        self.layoutChanged.emit()
        return failures or None

    # Clearing
    def clear(self: Self) -> None:
        self.__data.clear()
        self.layoutChanged.emit()

"""
A TableModel that implements a basic finding feature.

Since it implements the PaperTableModel, the data is passed by reference to be faster.
This implies that there may be problems with synchronisation and that is why
anyone that uses this must not manipulate the data outside this class.

More information is found in the initializer of PaperTableModel.

@author  Thomas Gauthier
@version 0.2
"""
class FindingModel(PaperTableModel):
    hidden_changed: Signal = Signal()

    # Default initializer
    def __init__(
                  self: Self,
                  data: list[Paper] | None,
                  columns: list[str] | tuple[str],
                  headers: list[str] | tuple[str],
                  parent:    QWidget | None = None
                ) -> None:
        # Additional variables
        self.__lock: Any = Lock()
        self.__hidden: list[Paper] = data
        self.__criterias: list[Callable[..., bool]] = []

        # Hidden  copied since it represents the data shown
        super().__init__(self.__hidden.copy(), columns, headers, parent)

    # Updates the view
    @staticmethod
    def updates[R, **P](func: Callable[P, R]) -> Callable[P, R]:
        # Get the "self" instance
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            self.__lock.acquire()
            argument: R = func(args, kwargs)
            self.updateShowing()
            self.__lock.release()
            return argument
        return inner

    # Mutates the hidden data
    @classmethod
    def mutated[R, **P](FindingModel, func: Callable[P, R]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            var: R = func(args, kwargs)
            self.hidden_changed.emit()
            return var
        return FindingModel.updates(inner)

    # Will update the showing data. Used when the real data has changed
    @Slot()
    def updateShowing(self: Self) -> None:
        self.__data = self.getRespecting()
        self.layoutChanged.emit()

    # Method used to return the papers that are respecting the criterias
    def getRespecting(self: Self) -> list:
        respecting: list[Paper] = []

        for paper in self.__hidden:
            # Could be boxed in another function, depending on future requirements
            respects_all: bool = True
            for criteria in self.__criterias:
                if not criteria(paper):
                    respects_all = False
                    break

            if respects_all: respecting.append(paper)

        return respecting

    # Function that will add the criterias for the view
    def addCriterias(self: Self, criterias: Iterable[Callable[..., bool]]) -> None:
        self.__criterias.extend(criterias)

    def removeCriterias(self: Self, criteria: dict[int, Callable[..., bool]] | list[Callable[..., bool]]) -> list[int] | None:
        keys: Iterable[int] = []

        if isinstance(criteria, list): keys = map(id, criteria)
        else: keys = criteria.keys()

        failures: list[int] = []
        for key in keys:
            try: self.__criterias.pop(key)
            except: failures.append(key)

        return failures or None

    def clearCriterias(self: Self) -> None:
        self.__criterias.clear()

    # Appends new data and shows it if it respects the current criterias
    # Note that this could also be done outside this method since the data
    # Is passed by reference. This also accepts doubles
    def append(self: Self, data: list[Paper]) -> None:
        self.__hidden.extend(data)

    # Removes the elements that are the same as in the removal list
    def remove(self: Self, removal: list[Paper]) -> list | None:
        failures: list = []

        for element in removal:
            try: self.__hidden.remove(element)
            except: failures.append(element)

        return failures or None

    # Clear alls the data
    def clear(self: Self) -> None:
        self.__hidden.clear()

# Manually decorating methods
for method in (
    FindingModel.clear,
    FindingModel.remove,
    FindingModel.append,
):
    method = FindingModel.mutated(method)

for method in (
    FindingModel.clearCriterias,
    FindingModel.addCriterias,
    FindingModel.removeCriterias
):
    method = FindingModel.updates(method)

"""
Implements the PaperTableModel with barebone unique data manipulation.

@author  Thomas Gauthier
@version 0.1
"""
class UniqueTableModel(MutableTableModel):
    # Default initializer
    def __init__(
                  self: Self, data: list | None,
                  columns: list[str] | tuple[str],
                  headers: list[str] | tuple[str],
                  parent:    QWidget | None = None
                ) -> None:
        super().__init__(data, columns, headers, parent)

    # Appends the new data and removes doubles
    @override
    def append(self: Self, data: list) -> None:
        self.__data.append(data)
        unique(self.__data)
        self.layoutChanged.emit()

"""
Basic class that represents a table showing the title,
journal, date, doi and the current label of some paper.

This is used for example on step one and on the data label.

When an element is clicked, it will cycle through the possible labels.

@author  Thomas Gauthier
@version 0.2
"""
@final
class SelectionModel(FindingModel):
    # An ordered list of the data shown
    columns: Final[list[str]] = ["title", "date", "jour", "doi", "label"]
    headers: Final[list[str]] = ["Title", "Date", "Journal", "DOI", "Label"]
    dict_images: Final[dict[int, QtGui.QIcon]] = {
        0: QtGui.QIcon(":/resources/grey_bar.png"),
        1: QtGui.QIcon(":/resources/green_checkbar.png"),
        2: QtGui.QIcon(":/resources/delete.png")
    }

    # Basic constructor that receives a list of the elements containing papers.
    def __init__(self: Self, data: list, parent: QWidget | None = None) -> None:
        super().__init__(data, SelectionModel.columns, SelectionModel.headers, parent)

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
    def clicked(self: Self, index: QModelIndex) -> None:
        selected: Paper = self.__viewing[index.row()]
        selected.index = (selected.index + 1) % len(SelectionModel.dict_images)
        self.layoutChanged.emit()

    # Default behaviour for double clicking on an item
    def doubleClicked(self: Self, index: QModelIndex) -> None:
        from ui.windows import PaperView
        paper_view: PaperView = PaperView(self.__viewing[index.row()])
        paper_view.changed.connect(self.layoutChanged.emit)
        paper_view.show()

"""
Basic view for selecting papers that reimplement the clicked and
double clicked callbacks.

@author  Thomas Gauthier
@version 0.1
"""
class SelectionView(QTableView):
    # Basic Constructor. Only calls the super constructor
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    # Overrides it with the one from the model
    @override
    def clicked(self: Self, index: QModelIndex) -> None:
        self.model.clicked(index)

    # Overrides it with the one from the model
    @override
    def doubleClicked(self: Self, index: QModelIndex) -> None:
        self.model.doubleClicked(index)

"""
Basic implementation of the FindingModel as a TableView.

@author  Thomas Gauthier
@version 0.3
"""
class FindingView(QWidget):
    """
    The basic constructor for a FindingView. It receives a model that inherits from the finding view
    and a view that inherits from QTableView so that multiple TableViews can have a "find" functionality.

    Note that if the model and/or the view is not provided, this will automatically assume that they
    are standard QTableView and FindingModel (respectively).

    Please be sure to pass an instance of the view and an instance of the model as rvalues, since,
    by keeping an exterior reference, this may lead to bugs.
    """
    def __init__[**P](
                  self:   Self,
                  model:  FindingModel,
                  view:   QTableView,
                  parent: QWidget | None = None,
                ) -> None:
        # Default initialization
        super().__init__(parent)
        view.setModel(model)

        # Saving the values for another time
        self.view:    QTableView = view
        self.model: FindingModel = model

        # Connecting both layouts together
        layout: QWidget = QVBoxLayout(self)
        layout.children().append(view)
        self.setLayout(layout)

    # A useful method for getting the criterias based on standard input
    # Note that this method is temporary and is only used to get the lambdas
    def __getCriterias(
                        self:    Self,
                        title:   tuple[str, bool] | None,
                        journal: tuple[str, bool] | None,
                        date:    tuple[dt.date, dt.date] | None,
                        doi:     str | None,
                        label:   str | None
                       ) -> list[Callable[..., bool]]:
        # Shortcuts for not typing it out everytime
        shortcut: Callable[..., re.Pattern] = lambda arg: re.compile(arg[0] if arg[1] else re.escape(arg[0]))
        trying: Callable[..., Any] = lambda val, func: func if not isinstance(val, None) else None

        every: list[Callable[..., bool] | None] = [
            trying(journal, (lambda val:   (lambda paper: val.match(paper.journal)))(shortcut(journal))),
            trying(title,   (lambda val:   (lambda paper: val.match(paper.title)))(shortcut(title))),
            trying(date,    (lambda paper: (date[0] <= paper.date and paper.date <= date[1]))),
            trying(doi,     (lambda paper: (paper.doi == doi)))
        ]
        if label is not None and label == Paper.POSSIBILITIES[0]: every.append(lambda paper: paper.labeled())
        else: every.append(trying(label, (lambda paper: (paper.label == label))))

        clearEmpty(every)
        return every

    """
    Basic function that will change the data being dispayed
    by the hits it will find given the arguments.

    Note that, by default, it will search for a string
    that is defined by the regex *arg* (verbose), but, if the boolean in the tuple
    is set to True, then it will consider the string as a regex.
    """
    def find(
              self:    Self,
              title:   tuple[str, bool] | None,
              journal: tuple[str, bool] | None,
              date:    tuple[dt.date, dt.date] | None,
              doi:     str | None,
              label:   str | None
            ) -> None:
        if label not in Paper.POSSIBILITIES: raise Exception("Not in possibilities")

        """
        HACK : Access to private variables.

        Note that this is generally unsafe, but since the "addCriterias" method is invoked,
        This is used to save a bit of runtime execution, since, if the "clearCriterias" method
        Was invoked, that would update the view twice, which is suboptimal

        The lock must be acquired to be thread safe.
        """
        self.model.__lock.acquire()
        self.model.__criterias.clear()
        self.model.__lock.release()

        self.model.addCriterias(self.__getCriterias(title, journal, date, doi, label))

    def found(
               self:    Self,
               title:   tuple[str, bool] | None,
               journal: tuple[str, bool] | None,
               date:    tuple[dt.date, dt.date] | None,
               doi:     str | None,
               label:   str | None
             ) -> list:
        """
        HACK : Access to private variables.

        This is generally unsafe, but necessary in this context, since you
        don't want to modify the view, but only access the papers to see which ones
        respect the theoretical criteria emitted.

        The lock must be acquired to be thread safe.
        """
        self.model.__lock.acquire()

        momento: list[Callable[..., Any]] = self.model.__criterias.copy()
        self.model.__criterias = self.__getCriterias(title, journal, date, doi, label)

        temp: list = self.model.getRespecting()
        self.model.__criterias = momento

        self.model.__lock.release()
        return temp

    """
    Basic function that will remove all the files that respect the findings.

    For more information on how it finds these papers, please go check
    the "find" method.
    """
    def removeFound(
                      self: Self,
                      title: tuple[str, bool] | None,
                      journal: tuple[str, bool] | None,
                      date: tuple[dt.date, dt.date] | None,
                      doi: str | None,
                      label: int | None
                    ) -> None:
        self.model.remove(self.found(title, journal, date, doi, label))

"""
Basic graph using pyqtgraph as instructed in the pdf.

It draws the operating characteristic curve based on the parameters
and a few lines to represent them.

@author  Thomas Gauthier
@version 0.1
"""
@final
class OperatingCurve(QWidget):
    # Number of points shown
    """
    FIXME : If necessary, change this to be a constant based on the maximal width of the screens
    Such as, for example (Java PseudoCode):
    final int points = Math.round(screens.stream().map(screen::getWidth).max() * CONSTANT);
    Or something like that...
    """
    POINTS:     Final[int]   = 500
    # Threshold to stop the search for the nc parameters. Will throw an exception pass that point
    THRESHOLD:  Final[int]   = 1000
    # Number of processes in the pool
    POOL_COUNT: Final[int]   = 4

    modifying: Signal = Signal(bool)

    # Default initializer that only set the basic themes
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Plot theme
        self.plot: PlotWidget = PlotWidget()
        self.pen:  QtGui.QPen = mkPen(color='b', width=5, style=Qt.PenStyle.SolidLine)

        self.plot.setBackground('w')
        self.plot.plotItem.setLabel("left", "Probability of acceptance")
        self.plot.plotItem.setLabel("bottom", "Fraction defective")

        # Layout
        layout = QVBoxLayout(self)
        layout.addLayout(self.plot)
        self.setLayout(layout)

        # Multithreading
        self.__lock: Any = RLock()

    @override
    def setDisabled(self: Self, state: bool) -> None:
        self.plot.setDisabled(state)
        self.clear.setDisabled(state)

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.setDisabled(not state)

    # Static method that modifies some function to require the lock
    @staticmethod
    def modifies[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            self.__lock.acquire()

            self.modifying.emit(True)
            ret: R = func(args, kwargs)
            self.modifying.emit(False)

            self.__lock.release()
            return ret
        return inner

    # Setter for the attributes of the characteristic function
    def setAttributes(
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

    Note that this is using the global delegator, for it contains signals
    to the UI thread. Note that, given that if there is another Delegator from
    the UI, both will compete over the same resources, it is unnecessary and
    unwise to create a parameter that represents a specific delegator when the global
    one suffices for this specific scenario.

    Note that there is a constant number of points, since recalculating
    some lot of given points each time the window is resized gets expensive pretty quickly.

    Note that the amount of points plotted follows the resolution of the biggest screen.
    """
    def plotf(self: Self, delegate: bool = True) -> None:
        from ui.windows import GLO_DEL

        for item in (self.alpha, self.beta, self.param1, self.param2):
            if item is None: raise Exception("Cannot have null parameters")
        if self.nc[0] == -1: self.findNC()

        p_values: np.ndarray[np.floating[Any]] = np.linspace(0, 0.5, OperatingCurve.POINTS)
        probabilities: list[float] = [OperatingCurve.probAcceptance(self.nc[0], self.nc[1], p) for p in p_values]

        def do() -> None:
            # Plotting the function and the lines
            self.plot.plotItem.plot(p_values, probabilities, pen=self.pen)
            self.plot.plotItem.addLegend()

            # Lines representing the current values used
            producer: float = 1 - self.alpha
            self.plot.plotItem.addLine(
                                        name = "Producer: 1 - alpha (" + str(producer) + ')',
                                        y    = producer,
                                        pen  = mkPen(hsv = (20, 85, 95), width = 0.5, style = Qt.PenStyle.DashLine)
                                      )

            self.plot.plotItem.addLine(
                                        name = "Consumer: beta (" + str(self.beta) + ')',
                                        y    = self.beta,
                                        pen  = mkPen(color = 'b', width = 0.5, style = Qt.PenStyle.DashLine)
                                      )

            self.plot.plotItem.addLine(
                                        name = "Parameter 1: " + str(self.param1),
                                        x    = self.param1,
                                        pen  = mkPen(color = 'g', width = 0.5, style = Qt.PenStyle.DashLine)
                                      )

            self.plot.plotItem.addLine(
                                        name = "Parameter 2: " + str(self.param2),
                                        x    = self.param2,
                                        pen  = mkPen(color = 'r', width = 0.5, style = Qt.PenStyle.DashLine)
                                      )

            # Other formatting
            self.plot.plotItem.showGrid(x = True, y = True)
            self.plot.plotItem.legend.addItem(None, "Number of samples: " + str(self.nc[0]))
            self.plot.plotItem.legend.addItem(None, "Minimal number of acceptance: " + str(self.nc[1]))

        if delegate: do()
        else: GLO_DEL.call(do)


    """
    Clears the mask of all data put inside
    """
    def clear(self: Self) -> None:
        self.plot.plotItem.clear()

    """
    Method used to find the number of sample required and the
    minimum count so that the batch will be accepted.

    Note that there does not exist an explicit (as far as I know) formula
    for the nc parameters since they are distributed in a
    [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution).
    Hence, why the brute force approach.

    For more information on this method, please consult the requirements file.
    """
    def findNC(self: Self) -> tuple[int, int]:
        lock:    Any = Lock()
        self.nc: Any = Array('i', 2)

        # Inner function used by the instances of the pool
        # In most cases, it will find the smallest result
        # And the trade for speed is worth the while
        def inner(pool: Any, initial: int) -> tuple[int, int]:
            num: int = initial

            while num <= OperatingCurve.THRESHOLD:
                for cnt in range(num):
                    # Calculate producer's risk
                    producer: float = sum([binom.pmf(k, num, self.param1) for k in range(cnt + 1)])
                    # Calculate consumer's risk
                    consumer: float = sum([binom.pmf(k, num, self.param2) for k in range(cnt + 1)])

                    if producer >= 1 - self.alpha and consumer <= self.beta:
                        nonlocal lock
                        # If two results are possible, lock one for the pool to terminate
                        lock.acquire()
                        self.nc[0] = num
                        self.nc[1] = cnt
                        pool.terminate()

                    num += OperatingCurve.POOL_COUNT
            self.nc[0] = -1
            pool.terminate()

        with Pool(OperatingCurve.POOL_COUNT) as pool:
            for i in range(OperatingCurve.POOL_COUNT):
                pool.apply(inner, args=(pool, i + 1))

            pool.join()

            if self.nc[0] == -1:
                raise Exception("Could not find a nc that satisfies the values")

        lock.release()

    """
    Linear method of findNC. For more information, go see findNC for more information.

    This method is deprecated.
    """
    def linearFindNC(self: Self) -> None:
        for num in range(1, OperatingCurve.THRESHOLD):
            for cnt in range(num):
                # Calculate producer's risk
                producer: float = sum([binom.pmf(k, num, self.param1) for k in range(cnt + 1)])
                # Calculate consumer's risk
                consumer: float = sum([binom.pmf(k, num, self.param2) for k in range(cnt + 1)])

                if producer >= 1 - self.alpha and consumer <= self.beta:
                    self.nc = (num, cnt)
                    return

    # Method used to get the points
    @staticmethod
    def probAcceptance(n: int, c: int, p: float) -> float:
        return sum([binom.pmf(k, n, p) for k in range(c + 1)])

for method in (
    OperatingCurve.setAttributes,
    OperatingCurve.plotf,
    OperatingCurve.findNC,
    OperatingCurve.linearFindNC
):
    method = OperatingCurve.modifies(method)

"""
A factory method that will return a QMessageBox
for some error based on client input.

@author  Thomas Gauthier
@version 0.0
"""
def errorFactory(name: str, message: str, parent: QWidget | None = None) -> QMessageBox:
    return mbFactory(name, message, QMessageBox.Icon.Critical, QMessageBox.StandardButton.Ok, parent)

"""
A factory method that will return a QMessageBox with the given parameters.

@author  Thomas Gauthier
@version 0.0
"""
def mbFactory(
               name:    str,
               message: str,
               icon:    QMessageBox.Icon | None,
               buttons: QMessageBox.StandardButton | None,
               parent:  QWidget | None = None
             ) -> QMessageBox:
    mb: QMessageBox = QMessageBox(parent)
    mb.setWindowTitle(name)
    mb.setText(message)

    if buttons is not None: mb.setStandardButtons(buttons)
    if icon is not None: mb.setIcon(icon)

    mb.setFont(QFont("Open Sans", 14, 10))
    return mb

def waitFactory(
                 name:     str,
                 message:  str,
                 interval: int,
                 maxd:     int = 3,
                 parent:   QWidget | None = None
               ) -> QMessageBox:
    dots: int = 0
    mb: QMessageBox = mbFactory(
        name,
        message,
        QMessageBox.Icon.Information,
        None,
        parent
    )

    def increase() -> None:
        nonlocal dots
        dots = dots % maxd + 1

    timer: QTimer = QTimer(parent)
    timer.setInterval(interval)
    timer.timeout.connect(lambda : (increase(), mb.setText(message + "." * dots)))

    mapper: Callable[[QEvent], None] = mb.closeEvent
    mb.closeEvent = lambda event: (mapper(event), timer.stop(), timer.deleteLater())

    mb.show()
    timer.start()\

    return mb
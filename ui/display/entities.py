#!/usr/bin/env python3
"""
The main document regrouping the elements which were impossible
to fully code in the Qt Designer.

This include, for example, the table from the first step (or from the data button)
which shows each paper. Another example would be the Operating Characteristic Curve.

@author  Thomas Gauthier
@version 0.4
"""
from multiprocessing      import Lock, RLock
from PySide6              import QtGui
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
from python.src.utils.functions import unique
from ui.resources_loader        import *

import datetime  as dt
import numpy     as np
import pyqtgraph as pg

import re

# Aliases
data_role   = Qt.ItemDataRole
item_flags  = Qt.ItemFlag
align_flags = Qt.AlignmentFlag

"""
Basic class that represent the skeleton of all model table model used.

@author  Thomas Gauthier
@version 0.3
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

        self._columns: Final[list[str]] = columns.copy()
        self._headers: Final[list[str]] = headers.copy()
        self._data:    list[Paper] = data if data else []

    @override
    def rowCount(self: Self, index: QModelIndex) -> int:
        return len(self._data)

    @override
    def columnCount(self: Self, index: QModelIndex) -> int:
        return len(self._columns)

    # All data will be aligned in the center
    @override
    def headerData(self: Self, index: int, orientation: Qt.Orientation, role: data_role) -> Any:
        if role == data_role.TextAlignmentRole:
            return align_flags.AlignCenter

        if role == data_role.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[index]

        return None

    # All data will be aligned in the center
    @override
    def data(self: Self, index: QModelIndex, role: data_role) -> Any:
        if role == data_role.TextAlignmentRole:
            return align_flags.AlignCenter

        if role == data_role.DisplayRole:
            # Introspection moment
            string: str = str(vars(self._data[index.row()])[self._columns[index.column()]])
            return string if string else "N/A"

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
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.view)

"""
Implements the PaperTableModel with barebone data manipulation.

@author  Thomas Gauthier
@version 0.2
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
        self._data.append(data)
        self.layoutChanged.emit()

    # Removes the targets and sends back the failures
    def remove(self: Self, removal: list) -> list | None:
        failures: list = []

        for element in removal:
            try: self._data.remove(element)
            except: failures.append(element)

        self.layoutChanged.emit()
        return failures or None

    # Clearing
    def clear(self: Self) -> None:
        self._data.clear()
        self.layoutChanged.emit()

"""
A TableModel that implements a basic finding feature.

Since it implements the PaperTableModel, the data is passed by reference to be faster.
This implies that there may be problems with synchronisation and that is why
anyone that uses this must not manipulate the data outside this class.

More information is found in the initializer of PaperTableModel.

@author  Thomas Gauthier
@version 0.3
"""
class FindingModel(PaperTableModel):
    # Note that this variable is disjoint of the dataChanged variable.
    # Even though hidden_changed should be connected to the dataChanged, because
    # Of the updateShowing method, here, we save a lot of computer cycles
    # If we do not connect this signal to the other. That implies that dataChanged must
    # Be manually emitted once this is.
    hidden_changed: Signal = Signal()

    # Default initializer
    def __init__(
                  self:    Self,
                  data:    list[Paper] | None,
                  columns: list[str] | tuple[str],
                  headers: list[str] | tuple[str],
                  parent:  QWidget | None = None
                ) -> None:
        # Additional variables
        self.__lock:      Any = Lock()
        self._hidden:    list[Paper] = data
        self._criterias: list[Callable[..., bool]] = []

        # Hidden  copied since it represents the data shown
        super().__init__(self._hidden.copy(), columns, headers, parent)

    # Updates the view
    @staticmethod
    def updates[R, **P](func: Callable[P, R]) -> Callable[P, R]:
        # Get the "self" instance
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            self.__lock.acquire()
            argument: R = func(*args, **kwargs)
            self.updateShowing()
            self.__lock.release()
            return argument
        return inner

    # Mutates the hidden data
    @classmethod
    def mutated[R, **P](FindingModel, func: Callable[P, R]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            var: R = func(*args, **kwargs)
            self.hidden_changed.emit()
            return var
        return FindingModel.updates(inner)

    # Will update the showing data. Used when the real data has changed
    @Slot()
    def updateShowing(self: Self) -> None:
        self._data = self.getRespecting()
        self.layoutChanged.emit()

    # Method used to return the papers that are respecting the criterias
    def getRespecting(self: Self) -> list:
        respecting: list[Paper] = []

        for paper in self._hidden:
            # Could be boxed in another function, depending on future requirements
            respects_all: bool = True
            for criteria in self._criterias:
                if not criteria(paper):
                    respects_all = False
                    break
            if respects_all: respecting.append(paper)

        return respecting

    # Function that will add the criterias for the view
    def addCriterias(self: Self, criterias: Iterable[Callable[..., bool]]) -> None:
        self._criterias.extend(criterias)

    def removeCriterias(self: Self, criterias: dict[int, Callable[..., bool]] | list[Callable[..., bool]]) -> list[int] | None:
        keys: Iterable[int] = []

        if isinstance(criterias, list): keys = map(id, criterias)
        else: keys = criterias.keys()

        failures: list[int] = []
        for key in keys:
            try: self._criterias.pop(key)
            except: failures.append(key)

        return failures or None

    def clearCriterias(self: Self) -> None:
        self._criterias.clear()

    # Appends new data and shows it if it respects the current criterias
    # Note that this could also be done outside this method since the data
    # Is passed by reference. This also accepts doubles
    def append(self: Self, data: list[Paper]) -> None:
        self._hidden.extend(data)

    # Removes the elements that are the same as in the removal list
    def remove(self: Self, removal: list[Paper]) -> list | None:
        failures: list = []

        for element in removal:
            try: self._hidden.remove(element)
            except: failures.append(element)

        return failures or None

    # Clear alls the data
    def clear(self: Self) -> None:
        self._hidden.clear()

FindingModel.clear  = FindingModel.mutated(FindingModel.clear)
FindingModel.remove = FindingModel.mutated(FindingModel.remove)
FindingModel.append = FindingModel.mutated(FindingModel.append)

FindingModel.addCriterias    = FindingModel.updates(FindingModel.addCriterias)
FindingModel.clearCriterias  = FindingModel.updates(FindingModel.clearCriterias)
FindingModel.removeCriterias = FindingModel.updates(FindingModel.removeCriterias)

"""
Implements the PaperTableModel with barebone unique data manipulation.

@author  Thomas Gauthier
@version 0.2
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
        self._data.append(data)
        unique(self._data)
        self.layoutChanged.emit()

"""
Basic class that represents a table showing the title,
journal, date, doi and the current label of some paper.

This is used for example on step one and on the data label.

When an element is clicked, it will cycle through the possible labels.

@author  Thomas Gauthier
@version 0.3
"""
@final
class SelectionModel(FindingModel):
    # An ordered list of the data shown
    columns: Final[list[str]] = ["title", "date", "jour", "doi", "label"]
    headers: Final[list[str]] = ["Title", "Date", "Journal", "DOI", "Label"]
    dict_images: Final[dict[int, QtGui.QIcon]] = {
        "Unlabeled": QtGui.QIcon(":/resources/grey_bar.png"),
        "Accepted":  QtGui.QIcon(":/resources/green_checkmark.png"),
        "Rejected":  QtGui.QIcon(":/resources/delete.png")
    }

    # Basic constructor that receives a list of the elements containing papers.
    def __init__(self: Self, data: list, parent: QWidget | None = None) -> None:
        super().__init__(data, SelectionModel.columns, SelectionModel.headers, parent)

    # Return the flags for an item in the table
    def flags(self: Self, index: Any) -> item_flags:
        return item_flags.ItemIsEditable

    # Basic function used to return layout information
    @override
    def data(self: Self, index: QModelIndex, role: QModelRoleData) -> Any:
        if index.column() != self._columns.index("label"):
            return super().data(index, role)
        elif role == data_role.DecorationRole:
            return SelectionModel.dict_images[self._data[index.row()].label]
        elif role == data_role.DisplayRole:
            return self._data[index.row()].label

        return None

    # Default behaviour for clicking on an item
    def clicked(self: Self, index: QModelIndex) -> None:
        selected: Paper    = self._data[index.row()]
        selected.label     = Paper.LABELS[(Paper.LABELS.index(selected.label) + 1) % len(SelectionModel.dict_images)]
        field: QModelIndex = self.createIndex(index.row(), len(self._headers) - 1)
        self.dataChanged.emit(field, field, [])

    # Default behaviour for double clicking on an item
    def doubleClicked(self: Self, index: QModelIndex) -> None:
        from ui.windows import PaperView
        paper_view: PaperView = PaperView(self._data[index.row()])
        def _connection() -> None:
            nonlocal index, self
            self.dataChanged.emit(index, index, [])
        paper_view.changed.connect(_connection)
        paper_view.show()

"""
Basic view for selecting papers that reimplement the clicked and
double clicked callbacks.

@author  Thomas Gauthier
@version 0.2
"""
class SelectionView(QTableView):
    # Basic Constructor. Only calls the super constructor
    def __init__(self: Self, data: list, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.__model = SelectionModel(data)
        self.setModel(self.__model)

        self.clicked.connect(self.__model.clicked)
        self.doubleClicked.connect(self.__model.doubleClicked)

"""
Basic implementation of the FindingModel as a TableView.

@author  Thomas Gauthier
@version 0.4
"""
class FindingView(QWidget):
    """
    The basic constructor for a FindingView. It receives a view that inherits from QTableView
    so that multiple TableViews can have a "find" functionality.

    Note that the view must already have a model set internally.

    Please be sure to pass an instance of the view and an instance of the model as rvalues, since,
    by keeping an exterior reference, this may lead to bugs.
    """
    def __init__(
                  self:   Self,
                  view:   QTableView,
                  parent: QWidget | None = None,
                ) -> None:
        # Default initialization
        super().__init__(parent)

        # Saving the values for another time
        self.view:    QTableView = view
        # Shortcut instead of always calling self.view.model()
        self.model: FindingModel = view.model()

        # Connecting both layouts together
        layout: QWidget = QVBoxLayout(self)
        layout.layout().addWidget(view)
        self.setLayout(layout)

    # A useful method for getting the criterias based on standard input
    # Note that this method is temporary and is only used to get the lambdas
    def _getCriterias(
                       self:    Self,
                       title:   tuple[bool, str] | None,
                       journal: tuple[bool, str] | None,
                       date:    tuple[dt.date, dt.date] | None,
                       doi:     str | None,
                       label:   str | None
                     ) -> list[Callable[..., bool]]:
        # Shortcuts for not typing it out everytime
        shortcut: Callable[..., re.Pattern] = lambda arg: (re.compile(arg[1] if arg[0] else re.escape(arg[1])))

        every: list[Callable[[Paper], bool] | None] = []
        if journal is not None: every.append((lambda val: (lambda paper: val.match(paper.journal)))(shortcut(journal)))
        if title is not None: every.append((lambda val: (lambda paper: val.match(paper.title)))(shortcut(title)))
        if date is not None: every.append(lambda paper: (date[0] <= paper.date and paper.date <= date[1]))
        if doi is not None: every.append(lambda paper: (paper.doi == doi))

        if label is not None and label == Paper.POSSIBILITIES[0]: every.append(lambda paper: paper.labeled())
        elif label is not None: every.append(lambda paper: (paper.label == label))

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
        self.model._FindingModel__lock.acquire()
        self.model._criterias.clear()
        self.model._FindingModel__lock.release()

        self.model.addCriterias(self._getCriterias(title, journal, date, doi, label))

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
        self.model._FindingModel__lock.acquire()

        momento: list[Callable[..., Any]] = self.model._criterias.copy()
        self.model._criterias = self._getCriterias(title, journal, date, doi, label)

        temp: list = self.model.getRespecting()
        self.model._criterias = momento

        self.model._FindingModel__lock.release()
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
    POINTS:     Final[int]   = 100
    # Threshold to stop the search for the nc parameters. Will throw an exception pass that point
    THRESHOLD:  Final[int]   = 1000
    # Number of processes in the pool
    POOL_COUNT: Final[int]   = 4

    modifying: Signal = Signal(bool)

    # Default initializer that only set the basic themes
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Plot theme
        self.plot: pg.PlotWidget = pg.PlotWidget()
        self.pen:  QtGui.QPen = pg.mkPen(color='k', width=2, style=Qt.PenStyle.SolidLine)

        self.plot.plotItem.setLabel("left", "Probability of acceptance")
        self.plot.plotItem.setLabel("bottom", "Fraction defective")
        self.plot.plotItem.setTitle("Operating Curve")
        self.plot.setBackgroundBrush(QtGui.QColor.fromRgb(2960685))
        # Doesn't do anything because all the objects added don't have the attribute 'plotData'/'implements
        # Making the internal name be none thus making the label not append any InfiniteLine that will
        # Be instantiated later on. Leave this as be.
        self.plot.plotItem.addLegend()  # FIXME : Make a legend that shows the InfiniteLines

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # Multithreading
        self.__lock:         Any = RLock()
        self.nc: tuple[int, int] = (-1, -1)

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
            try: ret: R = func(*args, **kwargs)
            finally:
                self.modifying.emit(False)
                self.__lock.release()

            return ret
        return inner

    # Setter for the attributes of the characteristic function
    def setAttributes(
                       self:   Self,
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
        from ui.windows import GLO_DEL, NC
        for item in (self.alpha, self.beta, self.param1, self.param2):
            if item is None: raise Exception("Cannot have null parameters")
        self.findNC()

        p_values: np.ndarray[np.floating[Any]] = np.linspace(0, 0.5, OperatingCurve.POINTS)
        probabilities: list[float] = [OperatingCurve.probAcceptance(self.nc[0], self.nc[1], p) for p in p_values]

        def _do() -> None:
            nonlocal self, p_values, probabilities
            # Plotting the function and the lines
            self.plot.plotItem.plot(p_values, probabilities, pen=self.pen)

            # Lines representing the current values used
            producer: float = 1 - self.alpha
            self.plot.plotItem.addLine(
                                        label     = "Producer: 1 - alpha (" + str(producer) + ')',
                                        labelOpts = {'movable': True},
                                        y         = producer,
                                        pen       = pg.mkPen(color="#f26824", width=2, style=Qt.PenStyle.DashLine)
                                      )

            self.plot.plotItem.addLine(
                                        label     = "Consumer: beta (" + str(self.beta) + ')',
                                        labelOpts = {'movable': True},
                                        y         = self.beta,
                                        pen       = pg.mkPen(color='w', width=2, style=Qt.PenStyle.DashLine)
                                      )

            self.plot.plotItem.addLine(
                                        label     = "Parameter 1: " + str(self.param1),
                                        labelOpts = {'movable': True},
                                        x         = self.param1,
                                        pen       = pg.mkPen(color='g', width=2, style=Qt.PenStyle.DashLine)
                                      )

            self.plot.plotItem.addLine(
                                        label     = "Parameter 2: " + str(self.param2),
                                        labelOpts = {'movable': True},
                                        x         = self.param2,
                                        pen       = pg.mkPen(color='r', width=2, style=Qt.PenStyle.DashLine)
                                      )

            # Other formatting
            self.plot.plotItem.showGrid(x=True, y=True)
            self.nc_win = NC(self.nc[0], self.nc[1])
            self.nc_win.show()

        if not delegate: _do()
        else: GLO_DEL.call(_do)

    # Clears the mask of all data put inside
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
    def findNC(self: Self) -> None:
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

OperatingCurve.setAttributes = OperatingCurve.modifies(OperatingCurve.setAttributes)
OperatingCurve.plotf         = OperatingCurve.modifies(OperatingCurve.plotf)
OperatingCurve.findNC        = OperatingCurve.modifies(OperatingCurve.findNC)

"""
A factory method that will return a QMessageBox
for some error based on client input.

@author  Thomas Gauthier
@version 0.0
"""
def errorFactory(name: str, message: str, parent: QWidget | None = None) -> QMessageBox:
    return mbFactory(
        name,
        message,
        QMessageBox.Icon.Critical,
        QMessageBox.StandardButton.Ok,
        parent
    )

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
        dots = (dots % maxd) + 1

    timer: QTimer = QTimer(parent)
    timer.setInterval(interval)
    timer.timeout.connect(lambda : (increase(), mb.setText(message + "." * dots)))

    mapper: Callable[[QEvent], None] = mb.closeEvent
    mb.closeEvent = lambda event: (mapper(event), timer.stop(), timer.deleteLater())

    mb.show()
    timer.start()

    return mb
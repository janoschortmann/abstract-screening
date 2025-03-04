#!/usr/bin/env python3
"""
File containing all instances of the windows defined with the xml files.
It also connects the signals/callbacks of those windows.

@author  Thomas Gauthier
@version 0.3
"""
from datetime          import datetime
from pathlib           import Path
from threading         import Condition, Thread, Lock
from typing            import *
from PySide6.QtWidgets import (
                                QDialog,
                                QWidget,
                                QMessageBox,
                                QProgressBar,
                                QSizePolicy,
                                QTableView
                              )
from PySide6.QtCore    import QObject, QThread, Qt, Signal, SignalInstance, Slot
from PySide6.QtGui     import QStandardItemModel

from python.src.utils.files          import Paper, CENTRAL
from ui.compiled                     import (
                                              data,
                                              find,
                                              frequency,
                                              interval,
                                              loading,
                                              paper,
                                              params,
                                              stats,
                                              stem
                                            )
from ui.compiled.options             import first, second, third
from ui.display.entities             import (
                                              FindingView,
                                              FindingModel,
                                              PaperTableView,
                                              mbFactory,
                                              errorFactory
                                            )
from nltk.corpus                     import stopwords
from nltk.stem                       import PorterStemmer
from nltk.tokenize                   import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Web and async
from urllib.parse import quote
import aiohttp
import asyncio

# Relative libs
import python.src.utils.functions as funcs

# Internal libs
import csv
import json
import math
import random
import string

# Renamming
import os.path as osp
import numpy   as np
import pandas  as pd

import re

"""
Class that can should only be called from the ui therad so that other threads
may manipulate the UI without any problems. This is a patch to the current threading environment
which forces the ui the worker threads to be separated.

In a perfect world, this wouldn't be necessary, but it is a hack that works.

@author  Thomas Gauthier
@version 0.0
"""
@final
class Delegator(QObject):
    # This actually doesn't represent a signal that is own by the ui, but, since,
    # In theory, this object should've been created by the ui thread (which is now it's owner),
    # This signal is also own by the ui thread, making connections with it possible without any
    # Structural modification. Therefore, with this signal, any thread can modify the signal by
    # Calling the "Call" function in the ui object's Delegator
    ui_signal:   Signal = Signal()

    def __init__(self: Self) -> None:
        super().__init__()
        self.__lock: Lock = Lock()

    def call[**P](
              self:     Self,
              func:     Callable[[None], None],
              *args:    P.args,
              raised:   Callable[[Exception], None] | None = None,
              final:    Callable[[bool], None] | None = None,
              **kwargs: P.kwargs
            ) -> None:
        self.__lock.acquire()

        def _call() -> None:
            thrown: bool = False
            try: func(*args, **kwargs)
            except Exception as ex:
                thrown = True
                if raised is not None: raised(ex)
                else: raise ex
            finally:
                if final is not None: final(thrown)
                self.ui_signal.disconnect(_call)
                self.__lock.release()

        self.ui_signal.connect(_call, type=Qt.ConnectionType.QueuedConnection)
        self.ui_signal.emit()

    def wait[**P](
              self:     Self,
              func:     Callable[[None], None],
              *args:    P.args,
              raised:   Callable[[Exception], None] | None = None,
              final:    Callable[[bool], None] | None = None,
              **kwargs: P.kwargs
            ) -> None:
        waiter: Condition = Condition()
        self.__lock.acquire()

        def _call() -> None:
            thrown: bool = False
            try: func(args, kwargs)
            except Exception as ex:
                thrown = True
                if raised is not None: raised(ex)
                else: raise ex
            finally:
                if final is not None: final(thrown)
                self.ui_signal.disconnect(_call)
                with waiter: waiter.notify()
                self.__lock.release()

        self.ui_signal.connect(_call, type=Qt.ConnectionType.QueuedConnection)
        self.ui_signal.emit()

        with waiter: waiter.wait()

    # Function that allocates the given factory produced by the object to the desired thread
    def allocate(self: Self, func: Callable[[QObject | None], QObject], thread: QThread) -> QObject:
        res: QObject = func(self.ui_object)
        res.moveToThread(thread)
        return res

GLO_DEL: Final[Delegator] = Delegator()

"""
A window for showing all the papers in a given probability interval.

@author  Thomas Gauthier
@version 0.0
"""
@final
class Interval(QDialog, interval.Ui_mainwindow):
    # Default initialization
    def __init__(self: Self, papers: list[Paper], fr: float, to: float) -> None:
        # Basic contruction
        super().__init__()
        self.setupUi(self)

        # The main window
        self.field = PaperTableView(
            papers,
            ["title", "date", "jour", "prob"],
            ["Title", "Date", "Journal", "Probability"],
            self
        )

        self.from_edit.setText(str(fr))
        self.to_edit.setText(str(to))

"""
The main window after the stem has been called.

This is truly a barebone implementation and does nothing special on its own.

It will only get the minimial number of stems required for the stem to be accepted.

@author  Thomas Gauthier
@version 0.0
"""
@final
class Frequency(QDialog, frequency.Ui_mainwindow):
    # Default initializer
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        # Normal initializing
        super().__init__(parent)
        self.setupUi(self)

"""
A basic class that represents the list of the stem
given with the previous "frequency.py" script.

Note that it is up to the caller to synchronize
with the given list of stems afterwards (note that
if the instance of data is passed as a reference and
not a copy, this will automatically clone it).

Originally, it was figured that the user could add their own stems.
Unfortunately, I see two problems with that.

The first one is the optimization.
Indeed, if the user adds their own stems, that would mean to recalculate whether a given
stem is in the database, which would be quite expensive.

The second one is of utility. If the user must add some keyword relative to their
field, that would imply that, for all the given papers, not a single one included
the one which interested them, and, in most likelyhood, would not influence the AI,
which bases itself on the most predominant words already found.

In summary, even though this feature was originally planned, based on
the usefulness and the changes required to make this feature, it will be cut
from this program. For anyone who tries to add it back, go ahead, but I seriously
doubt about its efficacy.

@author  Thomas Gauthier
@version 0.1
"""
@final
class Stem(QDialog, stem.Ui_mainwindow):
    # Default initializer
    def __init__(self: Self, data: np.ndarray, parent: QWidget | None = None) -> None:
        # Basic initialization
        super().__init__(parent)
        self.setupUi(self)

        # ListView manipulation
        self.__model: QStandardItemModel = QStandardItemModel()
        self.__model.appendRow(data)
        self.stem_list.setModel(self.__model)

"""
Default implementation of the paper class.

This will only show, based on the xml file, the relevant information
for a given paper and will, based on the response, accept or reject the given paper.

@author  Thomas Gauthier
@version 0.1
"""
@final
class PaperView(QWidget, paper.Ui_mainwindow):
    changed: Signal = Signal()

    # The default constructor which receives a given paper to show
    def __init__(self: Self, paper: Paper, parent: QWidget | None = None) -> None:
        # Default initialization
        super().__init__(parent)
        self.setupUi(self)

        # Shortcut
        short: Callable[[Any], None] = lambda var: str(var) if var else "Unknown"

        # Sync UI with the given paper data
        self.title.setText(paper.title)
        self.abstract.setText(paper.abstr)
        self.doi_inp.setText(short(paper.doi))
        self.jour_inp.setText(short(paper.jour))
        self.date_inp.setText(short(paper.date))

        # Callbacks
        self.yes_but.clicked.connect(
            lambda ignored: (
                ((paper.give("Accepted"), self.changed.emit()) if paper.label != "Accepted" else None),
                self.close()
            )
        )
        self.no_but.clicked.connect(
            lambda ignored: (
                ((paper.give("Rejected"), self.changed.emit()) if paper.label != "Rejected" else None),
                self.close()
            )
        )
        self.cancel_but.clicked.connect(lambda ignored: self.close())

"""
Default implementation of the parameters window.

@author  Thomas Gauthier
@version 0.2
"""
@final
class Parameters(QWidget, params.Ui_mainwindow):
    # Tuple representing the labels associated with the line edits
    # Dictionary between their name in code and the ones printed out
    LABELS: Final[dict[str, str]] = {
        "api":   "API Key",
        "limit":   "Limit",
        "thr": "Threshold",
        "token":   "Token",
        "step":     "Step"
    }
    # Directory for printing parameters
    DIR:    Final[str] = osp.join(CENTRAL, "files")
    # File for printing parameters
    FILE:   Final[str] = osp.join(DIR, "params.json")

    writing: Signal = Signal(bool)

    # Default initializer
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        # Setting up the ui
        super().__init__(parent)
        self.setupUi(self)

        # Events
        self.writing.connect(self.setDisabled)

        # Callback
        self.save_info.clicked.connect(self.printInfo)

        if osp.exists(Parameters.FILE) and osp.isfile(Parameters.FILE):
            with open(Parameters.FILE) as file:
                values: Any = json.loads(file.read())

                # Shortcut to catch errors
                def get(key: str) -> str:
                    nonlocal values
                    return values.get(key, "")

                self.query_edit.setText(get(Parameters.LABELS["limit"]))
                self.api_edit.setText(get(Parameters.LABELS["api"]))
                self.token_edit.setText(get(Parameters.LABELS["token"]))
                self.thr_edit.setText(get(Parameters.LABELS["thr"]))
                self.step_edit.setText(get(Parameters.LABELS["step"]))

    @Slot(bool)
    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.save_info.setEnabled(state)

    @Slot(bool)
    @override
    def setDisabled(self: Self, state: bool) -> None:
        self.setEnabled(not state)

    # Prints the data in the line edits to the json file
    @Slot()
    def printInfo(self: Self) -> None:
        self.writing.emit(True)
        funcs.mkabsent(Parameters.DIR)
        with open(Parameters.FILE, mode="w") as file:
            file.writelines(
                json.dumps(
                    {
                        Parameters.LABELS["limit"]: self.query_edit.text(),
                        Parameters.LABELS["api"]:   self.api_edit.text(),
                        Parameters.LABELS["token"]: self.token_edit.text(),
                        Parameters.LABELS["thr"]:   self.thr_edit.text(),
                        Parameters.LABELS["step"]:  self.step_edit.text()
                    },
                    indent=4
                ).__str__()
            )
        self.save_info.setText("Saved!")
        self.writing.emit(False)

"""
Dialog window representing the statistics of the trained AI.

The arguments passed are:
"an" - Actual Negatives
"fn" - False Negatives
"ap" - Actual Positives
"fp" - False Positives

"acc" - Accuracy
"rec" - Recall
"pre" - Precision
"f1"  - F1 Stats

When pressed "save", will emit a signal with object "True"
When pressed "cancel", will emit a signal with object "False"

@author  Thomas Gauthier
@version 0.1
"""
@final
class Statistics(QDialog, stats.Ui_mainwindow):
    result: Signal = Signal(bool)

    # Initializer
    def __init__[**P](self: Self, parent: QWidget | None = None, **args: P.kwargs) -> None:
        super().__init__(parent)
        self.setupUi(self)

        shortcut: Callable[[str], str] = lambda string: args.get(string, "N/A")

        # Congruent window
        self.an_stat.setText(shortcut("an"))
        self.fn_stat.setText(shortcut("fn"))
        self.ap_stat.setText(shortcut("ap"))
        self.fp_stat.setText(shortcut("fp"))

        # Bottom
        self.f1_show.setText(shortcut("f1"))
        self.acc_show.setText(shortcut("acc"))
        self.prec_show.setText(shortcut("pre"))
        self.recall_show.setText(shortcut("rec"))

        self.decision_box.accepted.connect(lambda ignored: self.result.emit(True))
        self.decision_box.rejected.connect(lambda ignored: self.result.emit(False))

"""
A window for finding documents and removing them. Is used, for example,
in the FindTableView.

Note that it does not directly access the data but only emits signals
for the table to connect to.

@author  Thomas Gauthier
@version 0.2
"""
@final
class Find(QWidget, find.Ui_mainwindow):
    remove_signal: Signal = Signal()
    find_signal:   Signal = Signal()

    # Default initializer
    def __init__(self: Self, title: str | None = None, parent: QWidget | None = None) -> None:
        #Initializing
        super().__init__(parent)
        self.setupUi(self)
        self.label_box.addItems(Paper.POSSIBILITIES)

        # Custom title to differentiate different find windowr
        if title is not None: self.setWindowTitle(title)

        # Callbacks
        self.find_but.clicked.connect(self.find_signal.emit)
        self.clear_search.clicked.connect(self.remove_signal.emit)
        self.clear_edits.clicked.connect(self.clear)

    @Slot(bool)
    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.clear_search.setEnabled(state)
        self.clear_edits.setEnabled(state)
        self.find_but.setEnabled(state)

    @Slot(bool)
    @override
    def setDisabled(self: Self, state: bool) -> None:
        return self.setEnabled(not state)

    # Clear all edits
    @Slot()
    def clear(self: Self) -> None:
        for edit in (self.title_edit, self.journal_edit, self.doi_edit, self.label): edit.clear()

    # Sends a report of the given values that the user assigned
    # A quick shortcut with kwargs manipulation
    def sendReport(self: Self) -> dict[str, Any]:
        arguments: dict[str, Any] = {}

        # Shortcuts
        def shortcut[Q](name: str, arg: Q | None) -> None:
            nonlocal arguments
            if not arg: arguments[name] = arg

        # Note that these names correspond to the args in the "find" function
        # In the FindingView so that when calling "find" with "send_report",
        # You can assign the parameters directly with **report
        shortcut("date",    (self.from_date.date().toPython(), self.to_date.date().toPython()))
        shortcut("journal", (self.journal_regex.isChecked(), self.journal_edit.text()))
        shortcut("title",   (self.title_regex.isChecked(), self.title_edit.text()))
        shortcut("label",   self.label_box.currentText())
        shortcut("doi",     self.doi_edit.text())

        return arguments


"""
The class representing the complete dataset.
That imply that this manages all the data by itself.

In addition, since it manages the data and the main winow cannot
act without data. It is understood that there is no need for any
internal attribute to represent some specific delegator (there is
only one main delegator for the UI).

It also supports multithreading. See method signature and definition
for a specific method.

@author  Thomas Gauthier
@version 0.4
"""
@final
class Data(QWidget, data.Ui_mainwindow):
    # Main file where the dumps are..... well..... dumped
    CORE_DUMP: Final[str] = osp.join(CENTRAL, "dumps")

    """
    An inner class used to link up the different kinds of
    datasets.

    This is the easiest option since the model from the TableView
    can modify the data without this class being notified.
    """
    class JoinedList(list):
        def __init__(self: Self) -> None:
            super().__init__()
            self.leasers:  list[tuple[SignalInstance, list] | None] = list()
            self.emitters: set[SignalInstance] = set()

        @override
        def copy(self: Self) -> list:
            clone: Data.JoinedList = []
            for item in range(len(self)): clone.append(self[item])
            return clone

        @override
        def append(self: Self, item: Any) -> None:
            raise Exception("Not implemented")

        def append(self: Self, lease: tuple[SignalInstance, list] | None, instance: Paper, /) -> None:
            super().append(instance)
            self.leasers.append(lease)

            if lease is not None:
                lease[1].append(instance)
                self.emitters.add(lease[0])

        @override
        def extend(self: Self, iterable: Iterable[tuple[tuple[SignalInstance, list] | None, Paper]], /) -> None:
            for item in iterable: self.append(*item)

        @override
        def pop(self: Self, index: Any = -1, /) -> tuple[tuple[SignalInstance, list] | None, Paper]:
            val: tuple[SignalInstance, list] | None = self.leasers.pop(index)
            paper: Paper = super().pop(index)

            if val is not None:
                self.emitters.add(val[0])

            if paper in val[1]:
                val[1].remove(paper)

            return (val, paper)

        @override
        def insert(self: Self, index: Any,  lease: tuple[SignalInstance, list] | None, obj: Paper, /) -> Never:
            super().insert(index, obj)
            self.leasers.insert(index, lease)

            if lease is not None:
                self.emitters.add(lease[0])

            if obj not in lease[1]:
                lease[1].append(obj)

        @override
        def remove(self: Self, value: Paper) -> None:
            index: int = super().index(value)
            super().pop(index)

            obj: tuple[SignalInstance, list] | None = self.leasers.pop(index)
            if obj is not None:
                obj[1].remove(value)
                self.emitters.add(obj[0])  # Faster than calling each time the emit()

        def removeAll(self: Self, other: Paper) -> None:
            for item in other: self.remove(item)

        @override
        def sort(self: Self, *, key: Callable[..., Any], reverse: bool = False) -> Never:
            raise Exception("Not implemented")

        @override
        def __iter__(self: Self) -> Iterator[tuple[tuple[SignalInstance, list] | None, Paper]]:
            self.counter: int = 0
            return self

        @override
        def __next__(self: Self) -> tuple[tuple[SignalInstance, list] | None, Paper]:
            if self.counter == len(self):
                del self.counter
                raise StopIteration()
            item: tuple[tuple[SignalInstance, list] | None, Paper]  = self[self.counter]
            self.counter += 1
            return item

        @override
        def __getitem__(self: Self, index: Any, /) -> tuple[tuple[SignalInstance, list] | None, Paper]:
            return (self.leasers[index], super().__getitem__(index))

        # The rest of the methods will be the same
        def emit(self: Self) -> None:
            for emitter in self.emitters:
                if emitter is not None: emitter.emit()

        def switch(self: Self, paper: Paper, new: tuple[SignalInstance, list] | None) -> None:
            index: int = super().index(paper)
            temp: tuple[SignalInstance, list] | None = self.leasers[index]

            if temp == new: return
            self.leasers[index] = new

            if temp is not None: temp[1].remove(paper)
            if new is not None: new[1].append(paper)

            self.emitters.update(new[0], temp[0])

    being_modified:    Signal = Signal(bool)
    validation_signal: Signal = Signal()
    training_signal:   Signal = Signal()
    dataset_signal:    Signal = Signal()

    # Default constructor
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        # Default initializer
        super().__init__(parent)
        self.setupUi(self)
        self.find_window: Find = Find("Finder for Data")

        # Multithreading and parallelism
        self.__lock: Lock = Lock()

        # Callbacks
        self.add_but.clicked.connect(lambda ignored: Thread(target=self.add).start())
        self.remove_but.clicked.connect(lambda ignored: Thread(target=self.remove).start())
        self.find_but.clicked.connect(toggler(self.find_window))

        # Data attributes which are represented as a tuple of a Signal and list
        # This was done for the signals of the Qt API
        self.validation: tuple[SignalInstance, list] = (self.validation_signal, [])
        self.training:   tuple[SignalInstance, list] = (self.training_signal,   [])
        self.dataset:    tuple[SignalInstance, Data.JoinedList] = (self.dataset_signal, Data.JoinedList())

        # Widgets
        self.__papers: FindingView = FindingView(
            FindingModel(
                self.dataset[1],
                headers=["Title", "Journal", "Date"],
                columns=["title", "jour", "date"]
            ),
            QTableView(),
            self
        )
        self.specifier.addItems(["Training", "Validation", "None"])

        # Connect callbacks in the find window
        # In theory, this should not have any problems with the ui objects being in another thread since
        # It only emits signals and, thus, should not create any QObjects
        self.find_window.find_signal.connect(
            lambda : self.accessPapers(
                lambda papers: Thread(
                    target=papers.find,
                    kwargs=self.find_window.sendReport()
                ).start()
            )
        )
        self.find_window.remove_signal.connect(
            lambda : self.accessPapers(
                lambda papers: Thread(
                    target=papers.removeFound,
                    kwargs=self.find_window.sendReport()
                ).start()
            )
        )

        # Connecting with the finding view for updating
        self.dataset[0].connect(self.__papers.model.updateShowing)
        self.being_modified.connect(self.setDisabled)
        self.being_modified.connect(self.find_window.setDisabled)

    @Slot(bool)
    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.remove_but.setEnabled(state)
        self.find_window.setEnabled(state)
        self.add_but.setEnabled(state)

    @Slot(bool)
    @override
    def setDisabled(self: Self, state: bool) -> None:
        return self.setEnabled(not state)

    """
    Function used to access the papers and call a function upon them.

    Will automatically set this in the disabled state until
    the callable received finishes.

    This is used since this data window will be multithreaded.

    Note that this calls, automatically, the functions self.dataset[0].emit() and self.dataset[1].emit()
    """
    @Slot(object)
    def accessPapers(self: Self, function: Callable[[FindingView], None], wait: bool = True) -> bool:
        # Note that, in theory, the result shouldn't be necessary, since you could
        # Very well just get the state from the Signal of this widget, but this could still be useful in the future
        res: bool = self.__lock.acquire(wait)
        if not res: return res

        self.being_modified.emit(True)
        function(self.__papers)

        self.dataset[0].emit()
        self.dataset[1].emit()

        self.being_modified.emit(False)
        self.__lock.release()

        return res

    """
    A static useful method used for dumping the failures based on the constant CORE_DUMP in Data.

    @author  Thomas Gauthier
    @version 0.0
    """
    @staticmethod
    def dump[Q](failures: list[Q], parent: QWidget | None = None) -> None:
        if not failures:
            funcs.mkabsent(Data.CORE_DUMP)
            with open(osp.join(Data.CORE_DUMP, datetime.now().strftime("%Y-%m-%d-%Hh%Mm") + ".txt"), mode="w") as file:
                file.writelines(map(str, failures))

            mbFactory(
                "Core Dumped",
                f"Parsing of lines failed. Core dumped in the {Data.CORE_DUMP} directory",
                QMessageBox.Icon.Warning,
                QMessageBox.StandardButton.Ok,
                parent
            ).show()

    # Default callback for the adder
    def add(self: Self, text: str | None = None,  path: str | None = None) -> None:
        global GLO_DEL
        text:   str = self.specifier.itemText() if text is None else text
        _path: Path = Path(self.path.text()     if path is None else path)

        # Could also be done with a dictionary
        # But I don't feel like it
        dataset: tuple[SignalInstance, list] | None = None
        match text:
            case "Validation": dataset = self.validation
            case "Training":   dataset = self.training
            case _: dataset = None

        failures: list[tuple[Path, int]] = []
        try: self.accessPapers(
                lambda papers: (
                    failures.extend(self._recursiveAdd(dataset, _path)),
                    papers.model.layoutChanged.emit()
                )
            )
        except Exception as ex:
            GLO_DEL.call(
                lambda path, _self: errorFactory(
                    "Error in adding",
                    "Error in " + str(path.absolute()),
                    _self
                ).show(),
                path=_path,
                _self=self
            )

        if failures: Data.dump(failures, self)

    """
    A method used for recursive adding.
    Returns a list of lines which could not be parsed.

    This function is not blocking since it would add to the overall cost when
    there are lots of tiny files (espectially with RLock), thus it is the
    responsability of the inheriter to lock this function (if required).

    Note that this method is not *required* (since you could only call the parse method
    and then remove all hits), but this is faster since you don't need to append to a list
    and then remove it, which would bring the runtime at twice the time.
    """
    def _recursiveAdd(self: Self, dataset: tuple[SignalInstance, list] | None, path: Path) -> list[tuple[Path, int]]:
        failures: list[tuple[Path, int]] = []
        if path.is_dir():
            for other in path.iterdir(): failures.extend(self._recursiveAdd(dataset, other.absolute()))
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path, encoding="utf8") as file:
                parsed: Paper | None = None
                count:  int = -1

                for line in file.readlines():
                    count += 1

                    try: parsed = Paper.parseLine(line, str(path))
                    except:
                        failures.append((path, count))
                        continue


                    # Parsed will not be None
                    # Must be done manually to find the lines
                    # If this is too long, either change the dataset to a set and not a list or use numpy
                    index: int = -1
                    try: index = self.dataset[1].index(paper)
                    except:
                        self.dataset[1].append(dataset, parsed)
                        continue

                    # The if is not necessary in this context, but is better understood and less bug prone
                    if self.dataset[1].leasers[index] != dataset:
                        self.dataset[1].switch(paper, dataset)
        else: raise Exception("Bad file type")  # In the case of a link or othrer file format

        return failures

    # Default callback for the remover
    # Will show a QMessageBox based on the removal process
    def remove(self: Self, path: str | None = None) -> None:
        global GLO_DEL
        _path: Path = Path(self.path.text() if path is None else path)

        failures: list[tuple[Path, str, int]] = []
        try: self.accessPapers(
            lambda papers: (
                    failures.extend(self._recursiveRemove(_path)),
                    papers.model.layoutChanged.emit()
                )
            )
        except Exception as ex:
            GLO_DEL.call(
                lambda ex, path, _self: errorFactory(
                    "Error in removing",
                    ex.message + " in " + str(path.absolute()),
                    _self
                ).show(),
                ex=ex,
                path=_path,
                _self=self
            )

        if failures: Data.dump(failures, self)

    """
    A method used for recursive adding.
    Returns a list of lines which could not be pared.

    This function is not blocking since it would add to the overall cost when
    there are lots of tiny files (espectially with RLock), thus it is the
    responsability of the inheriter to lock this function (if required).

    Note that this method is not *required* (since you could only call the parse method
    and then remove all hits), but this is faster since you don't need to append to a list
    and then remove it, which would bring the runtime at twice the time.
    """
    def _recursiveRemove(self: Self, path: Path) -> list[tuple[Path, int, int]]:
        failures: list[tuple[Path, int, int]] = []

        if path.is_dir():
            for other in path.iterdir(): failures.extend(self._recursiveRemove(other.absolute()))
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path, encoding="utf8") as file:
                parsed: Paper | None = None
                count:  int = -1
                for line in file.readlines():
                    count += 1

                    # Must do this manually since the remove function will
                    # Search for a tuple and not an element
                    try: parsed = Paper.parseLine(line, str(path))
                    except:
                        failures.append((path, 0, count))
                        continue

                    try: self.dataset[1].remove(parsed)
                    except: failures.append((path, 1, count))
        else: raise Exception("Bad file type")

        return failures

    """
    Default function that will parse the given file/directory
    and return the elements found inside.

    Note that the recursive_add ant recursive_remove do the same job,
    except it doesn't need to append the results first, which decreases the
    time spent on the function by a factor of 2.
    """
    def parse(self: Self, path: Path) -> tuple[list[tuple[Path, int]], list]:
        failures: list[tuple[Path, int]] = []
        results:  list[Paper] = []

        if path.is_dir():
            for other in path.iterdir():
                data: tuple[list[tuple[Path, int]], list] = self.parse(other.absolute())
                failures.extend(data[0])
                results.extend(data[1])
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path, encoding="utf8") as file:
                parsed: Paper | None = None
                count:   int = -1

                for line in file.readlines():
                    count += 1

                    try: parsed = Paper.parseLine(line, str(path))
                    except:
                        failures.append((path, count))
                        continue

                    results.append(parsed)
        else: raise Exception("Bad file type")
        return (failures, results)

"""
This class encapsulates the stemming of the papers.

Note that this is not synchronized since there is nothing else
for the user to do or to add while this is stemming.

It will be called from the "MainWindow" class
for it to be __executed and not just shown__.

When the process is done stemming, this class
will emit a signal (from itself) with the reference of the list of the
stems and will also print out, at the default location,
a list of the stems in case of a crash.

@author  Thomas Gauthier, Janosch Ortmann
@version 0.2
"""
@final
class Loading(QDialog, loading.Ui_mainwindow):
    # Default write location of the stems
    DEFAULT_WRITE: Final[str] = osp.join(CENTRAL, "results", "")

    done: Signal = Signal()

    # Default initialization
    def __init__(
                 self:      Self,
                 samples:   list,
                 min_words: int = 3,
                 parent:    QWidget | None = None
                ) -> None:
        # Regular initialize
        super().__init__(parent)
        self.setupUi(self)

        self.__min_words:   int = min_words
        self.__samples:    list = samples  # Not copied here since it will be afterwards

    # Automatically allocates to the delegator
    def getStems(self: Self) -> None:
        global GLO_DEL
        # This is why this is not synchronized
        # Introspection for callable. Required later for the function "fit_transform"
        length_samples: int = len(self.__samples)
        length:         int  = length_samples  # Current Length for the progress bar
        GLO_DEL.call(lambda _self: _self.process.setText("Tokenizing papers by their representation..."), _self=self)

        tokens:  np.ndarray = np.empty((length_samples, 1), dtype=np.object_)
        lowered: np.ndarray = np.empty(length_samples,      dtype=np.object_)
        for count in range(length):
            lowered[count] = str(self.__samples[count]).lower()
            # Faster than to always append
            tokens[count][0] = wordpunct_tokenize(lowered[count])
            GLO_DEL.call(lambda _self: _self.bar.setValue(round(25 * count / length)), _self=self)
        tokens = np.concatenate(np.ravel(tokens))

        GLO_DEL.call(lambda _self: _self.process.setText("Removing invalid words..."), _self=self)
        sw = stopwords.words('english')
        punct = list(string.punctuation)

        length = len(sw)
        for count in range(length):
            tokens = np.delete(tokens, tokens == sw[count])
            GLO_DEL.call(
                lambda _self, count, length: _self.bar.setValue(25 + round(10 * count / length)),
                _self=self,
                count=count,
                length=length
            )

        length = len(punct)
        for count in range(length):
            tokens = np.delete(tokens, tokens == punct[count])
            GLO_DEL.call(
                lambda _self, count, length: _self.bar.setValue(35 + round(10 * count / length)),
                _self=self,
                count=count,
                length=length
            )

        numeric_mask: np.ndarray = np.asarray([word.isnumeric() for word in tokens], dtype = bool)
        tokens = np.delete(tokens, np.where(numeric_mask)[0])

        """
        For those that are confused here (as I was at first), we are only stemming single
        grams since, if we were stemming double grams, we would obtain some
        weird stuff that does not correspond to reality.

        Also, the way grammar works, it is rare to see two matching stems
        with different suffixes. For example:
        >   It is truly a piece of art.
        Here, "truly" is followed with the determiner "a". If we only
        took the stems, we would get "tru" and "a". But, as you may understand,
        there does not exist a way to connect both "tru" and "a" that doesn't
        require the stem "tru" to be an adverb.

        In general, the grammatical structure will require us to take a single option,
        hence why stemming both words is unnecessary.

        Thus, for convenience and to not reimplement the vectorizing
        library, only single grams are stemmed whilst the rest is just parsed.
        """

        GLO_DEL.call(lambda _self: _self.process.setText("Stemming words..."), _self=self)
        stemmer: PorterStemmer = PorterStemmer()
        for count in range(len(tokens)):
            tokens[count] = stemmer.stem(tokens[count])
            GLO_DEL.call(
                lambda _self, count, length: _self.bar.setValue(45 + round(10 * count / length)),
                _self=self,
                count=count,
                length=length
            )
        del stemmer, numeric_mask

        GLO_DEL.call(
            lambda _self: _self.process.setText("Counting uni grams..."),
            _self=self
        )
        count:     np.ndarray =  np.asarray(list(Counter(tokens).items()))

        if count.size == 0: return  # Nothing is worth doing if no single stem
        uni_found: np.ndarray = count[count[:, 1].astype(np.int_) >= self.__min_words]
        del count

        # Removing common words
        # Note to self: You cannot have a tuple of booleans, it must be a list of booleans
        uni_found = uni_found[[len(string) >= 4 for string in uni_found[:, 0]], :]

        # Counting each valid gram in the texts. Note that this is required since
        # We want the stems and not the words, thus a CountVectorizer cannot be used
        uni_sample_found: np.ndarray = np.empty((length_samples, uni_found.shape[0]))
        for sample_index in range(uni_sample_found.shape[0]):
            for stem_index in range(uni_sample_found.shape[1]):
                uni_sample_found[sample_index, stem_index] = lowered[sample_index].count(uni_found[stem_index, 0])
        GLO_DEL.call(lambda _self: _self.bar.setValue(65), _self=self)

        del lowered

        # Initializing for bi grams
        self.process.setText("Counting bi grams...")
        vectorizer: CountVectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=sw)
        bi_found:        np.ndarray = vectorizer.fit_transform(map(str, self.__samples)).toarray()
        bi_words:        np.ndarray = vectorizer.get_feature_names_out()

        def update_bi(respecting: np.ndarray | list[bool]) -> None:
            nonlocal bi_words, bi_found
            bi_words = bi_words[respecting]
            bi_found = bi_found[:, respecting]
        update_bi(np.sum(bi_found, axis=0) >= self.__min_words)                         # Removing grams with not enough words
        update_bi(~np.isin(bi_words, punct))                                            # Removing grams with punctuation
        update_bi([not any(char.isdigit() for char in string) for string in bi_words])  # Or containing digits
        GLO_DEL.call(lambda _self: _self.bar.setValue(75), _self=self)

        del update_bi

        # Initializing for tri grams
        GLO_DEL.call(lambda _self: _self.process.setText("Counting tri grams..."), _self=self)
        vectorizer: CountVectorizer = CountVectorizer(ngram_range=(3, 3), stop_words=sw)
        tri_found:       np.ndarray = vectorizer.fit_transform(map(str, self.__samples)).toarray()
        tri_words:       np.ndarray = vectorizer.get_feature_names_out()

        def update_tri(respecting: np.ndarray | list[bool]) -> None:
            nonlocal tri_words, tri_found
            tri_words = tri_words[respecting]
            tri_found = tri_found[:, respecting]

        # Removing unacceptable tri grams
        update_tri(np.sum(tri_found, axis=0) >= self.__min_words)                         # Removing grams with not enough words
        update_tri(~np.isin(tri_words, punct))                                            # Removing grams with punctuation
        update_tri([not any(char.isdigit() for char in string) for string in tri_words])  # Or containing digits
        GLO_DEL.call(lambda _self: _self.bar.setValue(90), _self=self)

        del update_tri

        GLO_DEL.call(lambda _self: _self.process.setText("Printing grams..."), _self=self)  # Done for safekeeping
        self.grams: np.ndarray = np.concatenate((uni_found[:, 0], bi_words, tri_words))

        # Shortcut so that you won't have to do this in the MainWindow
        self.grams_found: np.ndarray = np.concatenate((uni_sample_found, bi_found, tri_found), axis=1)
        funcs.mkabsent(Loading.DEFAULT_WRITE)
        pd.DataFrame(self.grams).to_csv(
            osp.join(Loading.DEFAULT_WRITE, datetime.now().strftime("%Y-%m-%d-%Hh%Mm") + ".txt"),
            sep=',',
            header=False,
            index=False
        )

        GLO_DEL.call(lambda _self: (_self.process.setText("Finished"), _self.bar.setValue(100)), _self=self)

        self.done.emit()

"""
Widget that represent the first parameters.

This class can also query from the Scopus database, but be advised
that since this does not control the data (that is the role of the Data class),
that means that, after querying, it will emit information about its current state
so that the main window can latch on it widhout doing any modifications to the stems.

The querying starts a new process

@author  Thomas Gautier, Janosch Ortmann
@version 0.2
"""
@final
class First(QWidget, first.Ui_first_option):
    # The default setting for when the directory isn't specified
    DEFAULT_QUERY: Final[str] = osp.join(CENTRAL, "query")
    # Number of queries per iteration
    COUNT:         Final[int] = 100
    # The name of the files that will be printed
    FILES: tuple[str] = ("validation.csv", "citations.csv", "sample.csv")

    querying:      Signal = Signal(bool)
    remove_signal: Signal = Signal()
    add_signal:    Signal = Signal()

    # Default initializer
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUi(self)

        self.__lock: Lock = Lock()
        def _process(signal: SignalInstance) -> None:
            try:
                # In the case that we want, in the future, the thread to do something else while waiting
                asyncio.run(self.query())
                signal.emit()
            except: pass
            finally:
                global GLO_DEL
                def do() -> None:
                    nonlocal self
                    self.progress.setParent(None)
                    del self.progress
                    self.querying.emit(False)
                GLO_DEL.call(do)

                self.querying.emit(False)
                self.__lock.release()

        shortcut: Callable[[SignalInstance], None] = lambda signal: Thread(target=_process, args=(signal,)).start()

        # Asynchronously querying
        self.remove.clicked.connect(lambda ignored: shortcut(self.remove_signal))
        self.add.clicked.connect(lambda ignored: shortcut(self.add_signal))

        # Deactivate multiple querying at the same time
        self.querying.connect(self.setDisabled)

    @Slot(bool)
    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.add.setEnabled(state)
        self.remove.setEnabled(state)

    @Slot(bool)
    @override
    def setDisabled(self: Self, state: bool) -> None:
        return self.setEnabled(not state)

    async def query(self: Self) -> None:
        global GLO_DEL
        self.querying.emit(True)

        def do() -> None:
            nonlocal self
            self.progress = QProgressBar(self.variables)
            self.progress.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed
            )
            self.variables.layout().addWidget(self.progress)

        GLO_DEL.call(do)

        if not self.__lock.acquire(blocking=False):
            GLO_DEL.call(
                lambda _self: errorFactory(
                    "Error in querying",
                    "Instance of querying already in process.",
                    _self
                ).show(),
                _self=self
            )
            raise ValueError()

        """
        --------------------------------------------------
                        Parameter validation
        --------------------------------------------------
        """

        # Temporary variable
        text: str = self.directory_edit.text()
        self.directory: Path = Path(text if text else First.DEFAULT_QUERY)
        del text

        funcs.mkabsent(self.directory)

        if not osp.exists(Parameters.FILE) or not osp.isfile(Parameters.FILE):
            GLO_DEL.call(
                lambda _self: errorFactory(
                    "Parameters not saved",
                    "The parameters file was not found.",
                    _self
                ).show(),
                _self=self
            )
            raise ValueError()

        val_per: float = 0
        bad_val: bool = False
        try: val_per = float(self.sample_edit.text())
        except: bad_val = True
        if val_per <= 0 or 1 <= val_per or bad_val:
            GLO_DEL.call(
                lambda _self: errorFactory(
                    "Bad validation size",
                    "The validation size isn't a valid percentage.",
                    _self
                ).show(),
                _self=self
            )
            raise ValueError()
        del bad_val

        sample_per: float = 0
        bad_sample: bool = False
        try: sample_per = float(self.sample_edit.text())
        except: bad_sample = True
        if sample_per <= 0 or 1 <= sample_per or val_per + sample_per >= 1 or bad_sample:
            GLO_DEL.call(
                lambda _self: errorFactory(
                    "Bad sample size",
                    "The sample size isn't a valid percentage.",
                    _self
                ).show(),
                _self=self
            )
            raise ValueError()
        del bad_sample

        date:    str = self.date_edit.text()
        matches: list[str] = re.findall(R"\d{4}-\d{4}", date)
        if (len(matches) != 1 and date) or (date and len(date) != len(matches[0])) or (int(date[:4]) > int(date[5:])):
            GLO_DEL.call(
                    lambda _self : errorFactory(
                    "Bad date range",
                    "The date range given isn't valid.",
                    _self
                ).show(),
                _self=self
            )
            raise ValueError()
        else: date = matches[0]
        del matches

        params: dict[str, str] = appendParams()

        """
        --------------------------------------------------
                            Querying
        --------------------------------------------------
        """

        limit: int = int(params["limit"])
        key:   str = params["api"]
        processed: Any = 0

        # Left as string to print them in file, where the parsing will happen later
        received: dict[str, list[str]] = {
            "titles":    [],
            "abstracts": [],
            "journals":  [],
            "dates":     [],
            "doi":       []
        }

        GLO_DEL.call(lambda _self: _self.progress.setMaximum(limit), _self=self)
        while processed < limit:
            # TODO: CHANGE THE VIEW FOR THE FINAL
            url: str = f"https://api.elsevier.com/content/search/scopus?apiKey={key}{f"&date={date}" if date else ""}&query={quote(self.query_box.document().toPlainText())}&view=STANDARD&start={processed}&count={min(First.COUNT, limit - processed)}"
            processed += First.COUNT

            entries: Any = None
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        raise_for_status=True,
                        headers={
                            'X-ELS-APIKey':    key,
                            'X-ELS-Insttoken': params["token"]
                        }
                    ) as response:
                        entries = await response.json()
                        # Funny waiting in coroutine; __getitem__(key) must be called after
                        entries = entries['search-results']

                        if not entries: break  # No more entries => Nothing else to fetch
            except aiohttp.ClientResponseError as ex:
                GLO_DEL.call(
                    lambda _self, ex: errorFactory(
                        "Critical Error",
                        f"Fetching of scopus failed with status {ex.status}. Hint: Check your API key if it allows a COMPLETE view.",
                        _self
                    ).show(),
                    _self=self,
                    ex=ex
                )
                raise ex
            except Exception as ex:
                GLO_DEL.call(
                    lambda _self: errorFactory(
                        "Critical Error",
                        f"Fetching of scopus failed",
                        _self
                    ).show(),
                    _self=self
                )
                raise ex

            # Assured that entry is the response from the server
            for entry in entries["entry"]:
                short: Callable[..., Any] = lambda var: entry.get(var, None)
                received["abstracts"].append(entry.get("dc:description", ""))
                received["journals"].append(short("prism:publicationName"))
                received["titles"].append(short("dc:title"))
                received["dates"].append(short("prism:coverDate"))
                received["doi"].append(short("prism:doi"))

                GLO_DEL.call(lambda _self: _self.progress.setValue(processed), _self=self)

        # In case it wasn't fully loaded
        GLO_DEL.call(lambda _self: _self.progress.setValue(_self.progress.maximum()), _self=self)
        del limit, params, processed


        """
        --------------------------------------------------
                            Printing
        --------------------------------------------------
        """

        citations: np.ndarray =  np.stack(
            (
                received["titles"],
                received["abstracts"],
                received["journals"],
                received["dates"],
                received["doi"]
            ),
            axis=-1
        )

        """
        Notice that, here, we do not delete the papers that are in the validation
        and the training dataset. That is, for two reasons:

        First of which, in any case, it will be added back to the main dataset with
        all the other papers because of how the add method works in the Data class.

        Second of which, I believe it would be nice to have the AI still include a probability
        for the given papers that were labeled. That acts both as a confirmation of the validity of
        the AI (in addition to the statistics window) and a failsafe in case the researcher read too rapidly.

        You can modify this, if needed, to remove from the dataset the papers that are in the validation
        and training, but note that if this is done, the add method in the Data class must also be modified.
        """
        # Please leave the variables used once, it's easier to understand the code this way

        number: Final[int] = len(citations)

        sample_index: list[int] = random.sample(range(number), math.ceil(number * sample_per))
        sample:      np.ndarray = citations[sample_index, :]
        citations:   np.ndarray = np.delete(citations, sample_index, axis=0)

        val_index:   list[int] = random.sample(range(number - len(sample)), math.ceil(number * val_per))
        validation: np.ndarray = citations[val_index, :]
        citations:  np.ndarray = np.delete(citations, val_index, axis=0)

        del number, sample_index, val_index

        funcs.mkabsent(First.DEFAULT_QUERY)

        # Shortcut
        printer: Callable[..., None] = lambda file, data: (
            funcs.mkabsent(self.directory),
            pd.DataFrame(data).to_csv(
                osp.join(self.directory, file),
                quoting=csv.QUOTE_ALL,
                doublequote=False,
                escapechar="\\",
                index=False,
                header=False
            )
        )

        printer(First.FILES[0], validation)
        printer(First.FILES[1], citations)
        printer(First.FILES[2], sample)

"""
Class representing the second window.
Like all other classes, will emit signals representing which actions to chose.

@author  Thomas Gauthier
@version 0.1
"""
@final
class Second(QWidget, second.Ui_second_option):
    clear_signal: Signal = Signal()
    plot_signal:  Signal = Signal()

    # Default initializer
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        # Classic setup
        super().__init__(parent)
        self.setupUi(self)

        self.clear.clicked.connect(self.clear_signal.emit)
        self.plot.clicked.connect(self.plot_signal.emit)

    @Slot(bool)
    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.clear.setEnabled(state)
        self.plot.setEnabled(state)

    @Slot(bool)
    @override
    def setDisabled(self: Self, state: bool) -> None:
        self.setEnabled(not state)

    # Sends the report of the parameter used
    def sendReport(self: Self) -> dict[str, float] | None:
        unverified_report: dict[str, float] | None = None

        try:
            edits: list[str] = (
                                 self.alpha_edit.text(),
                                 self.beta_edit.text(),
                                 self.param_edit_1.text(),
                                 self.param_edit_2.text()
                               )
            for count in range(len(edits)):
                if not edits[count]: edits.insert(count, '0')

            unverified_report = {
                "alpha":  float(edits[0]),
                "beta":   float(edits[1]),
                "param1": float(edits[2]),
                "param2": float(edits[3])
            }

            for value in unverified_report.values():
                if value < 0 or value > 1: raise Exception()

            if unverified_report["param1"] > unverified_report["param2"]: raise Exception()
        except *Exception as error:
            errorFactory("Wrong parameters", "Parameters entered are wrong", self).show()
            raise error

        return unverified_report

"""
Class representing the third and final option window.

@author  Thomas Gauthier
@version 0.0
"""
@final
class Third(QWidget, third.Ui_third_option):
    # Default initializer
    def __init__(self: Self, parent: QWidget | None = None) -> None:
        # Classic setup
        super().__init__(parent)
        self.setupUi(self)

        # Decisions
        self.model_box.addItems(("Linear Regression", "Decision Tree"))
        self.sampling_box.addItems(("Oversampling", "Undersampling"))

    # Sends a report of the current state of this window
    def sendReport(self: Self) -> dict[str, Any] | None:
        unverified_report: dict[str, Any] | None = None

        # Shortcut used to quickly verify the information
        def testing(name: str, param: Paper, criteria: Callable[..., bool] | None = None) -> None:
            nonlocal unverified_report
            if criteria and not criteria(param): raise Exception()
            else: unverified_report[name] = param

        try:
            testing("pos",    float(self.pos_edit.text()),  lambda x: x > 0 and x < 1)
            testing("size",   float(self.size_edit.text()), lambda x: x > 0 and x < 1)
            testing("splits", int(self.splits_edit.text()), lambda x: x >= 0)

            text: str = self.seed_edit.text()
            unverified_report["seed"]     = int(text) if text else 0
            unverified_report["model"]    = self.model_box.currentIndex()
            unverified_report["sampling"] = self.sampling_box.currentIndex()
        except *Exception as error:
            errorFactory("Wrong parameters", "Parameters entered are wrong", self).show()
            raise error

        return unverified_report

"""
Basic factory for toggling windows.

Overrides the close event with a hidden event to not create new windows each time.

@author  Thomas Gauthier
@version 1.1
"""
from PySide6.QtWidgets import QWidget
def toggler(window: QWidget) -> Callable[..., None]:
    window.closeEvent = lambda ignored: window.hide()
    def inner() -> None:
        if not window.isHidden(): window.hide()
        else: window.show()
    return inner

"""
Function that appends the previous parameters in the standard
file path.

@author  Thomas Gauthier
@version 0.0
"""
import json
def appendParams() -> dict[str, str]:
    global GLO_DEL
    params: dict[str, str] = {}
    with open(Parameters.FILE) as file:
        js: Any = json.loads(file.read())
        for key, val in Parameters.LABELS.items():
            try: params[key] = js[val]
            except:
                GLO_DEL.call(
                    lambda key: errorFactory(
                        "Bad argument",
                        "Parameter received had an error (" + key + ')'
                    ).show(),
                    key=key
                )
                return
    return params
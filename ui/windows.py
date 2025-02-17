#!/usr/bin/env python3
"""
File containing all instances of the windows defined with the xml files.
It also connects the signals/callbacks of those windows.

@author  Thomas Gauthier
@version 0.2
"""
from multiprocessing   import Process
from pathlib           import Path
from threading         import Lock
from typing            import *
from PySide6.QtWidgets import (
                                QDialog,
                                QWidget,
                                QMessageBox,
                                QProgressBar,
                                QSizePolicy,
                                QTableView
                              )
from PySide6.QtCore    import QEvent, Signal, SignalInstance
from PySide6.QtGui     import QStandardItemModel

from python.src.utils.files     import Paper, CENTRAL
from ui.compiled                import (
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
from ui.compiled.options        import first, second, third
from ui.display.entities        import (
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
from this program. For anywho who tries to add it back, go ahead, but I seriously
doubt about its efficacy.

@author  Thomas Gauthier
@version 0.1
"""
@final
class Stem(QDialog, stem.Ui_mainwindow):
    # Default initializer
    def __init__(self: Self, data: list[str], parent: QWidget | None = None) -> None:
        # Basic initialization
        super().__init__(parent)
        self.setupUi(self)

        # Received a numpy ndarray, but removing and appending is faster in a list
        self.__stems: list[str] = list(data)

        # ListView manipulation
        self.__model: QStandardItemModel = QStandardItemModel()
        for item in self.__stems:
            self.__model.appendRow(item)
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
    DIR:    Final[str] = CENTRAL + "files/"
    FILE:   Final[str] = DIR + "params.json"

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
                    try: return values[key]
                    except: return ""

                self.query_edit.setText(get(Parameters.LABELS["limit"]))
                self.api_edit.setText(get(Parameters.LABELS["api"]))
                self.token_edit.setText(get(Parameters.LABELS["token"]))
                self.thr_edit.setText(get(Parameters.LABELS["thr"]))
                self.step_edit.setText(get(Parameters.LABELS["step"]))

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.save_info.setEnabled(state)

    @override
    def setDisabled(self: Self, state: bool) -> None:
        self.setEnabled(not state)

    # Prints the data in the line edits to the json file
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

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.clear_search.setEnabled(state)
        self.clear_edits.setEnabled(state)
        self.find_but.setEnabled(state)

    @override
    def setDisabled(self: Self, state: bool) -> None:
        return self.setEnabled(not state)

    # Clear all edits
    def clear(self: Self) -> None:
        for edit in (self.title_edit, self.journal_edit, self.doi_edit, self.label): edit.clear()

    # Sends a report of the given values that the user assigned
    # A quick shortcut with kwargs manipulation
    def sendReport(self: Self) -> dict[str, Any]:
        arguments: dict[str, Any] = {}

        # Shortcuts
        def appendNotEmpty[Q](name: str, arg: Q | None) -> None:
            nonlocal arguments
            if not arg: arguments[name] = arg

        # Note that these names correspond to the args in the "find" function
        # In the FindingView so that when calling "find" with "send_report",
        # You can assign the parameters directly with **report
        appendNotEmpty("date",    (self.from_date.date().toPython(), self.to_date.date().toPython()))
        appendNotEmpty("journal", (self.journal_regex.isChecked(), self.journal_edit.text()))
        appendNotEmpty("title",   (self.title_regex.isChecked(), self.title_edit.text()))
        appendNotEmpty("label",   self.label_box.currentText())
        appendNotEmpty("doi",     self.doi_edit.text())

        return arguments


"""
The class representing the complete dataset.
That imply that this manages all the data by itself.

It also supports multithreading. See method signature and definition
for a specific method.

@author  Thomas Gauthier
@version 0.3
"""
@final
class Data(QWidget, data.Ui_mainwindow):
    # Main file where the dumps are..... well..... dumped
    CORE_DUMP: Final[str] = CENTRAL + "dumps/"

    """
    An inner class used to link up the different kinds of
    datasets.

    This is the easiest option since the model from the TableView
    can modify the data without this class being notified.
    """
    class JoinedList(list):
        @overload
        def __init__(self: Self) -> None:
            super().__init__()
            self.leaser: list[tuple[SignalInstance, list] | None] = []
            self.emiters: set[SignalInstance] = {}

        @override
        def __init__(self: Self, iterable: Iterable[tuple[tuple[SignalInstance, list] | None, Paper]] | Iterable, /) -> None:
            super.__init__()
            self.leaser: list[tuple[SignalInstance, list] | None] = []
            self.emiters: set[SignalInstance] = {}
            self.extend(iterable)

        @override
        def copy(self: Self) -> list:
            clone: Data.JoinedList = []
            for item in range(len(self)): clone.append((self[item], self.leaser[item]))
            return clone

        @override
        def append(self: Self, instance: Paper) -> Never:
            self.append((None, instance))

        @overload
        def append(self: Self, instance: Paper, lease: tuple[SignalInstance, list] | None = None, /) -> None:
            super().append(instance)
            self.leaser.append(lease)
            self.emiters.add(lease)

        @overload
        def append(self: Self, combined: tuple[tuple[SignalInstance, list] | None, Paper], /) -> None:
            super().append(combined[1])
            self.leaser.append(combined[0])
            self.emiters.add(combined[0])

        @override
        def extend(self: Self, iterable: Iterable[tuple[tuple[SignalInstance, list] | None, Paper]] | Iterable, /) -> None:
            for item in iterable: self.append(item)

        @override
        def pop(self: Self, index: Any = -1, /) -> tuple[tuple[SignalInstance, list] | None, Paper]:
            val: tuple[tuple[SignalInstance, list]] | None = self.leaser.pop(index)
            self.emiters.add(val)
            return (val, super().pop(index))

        @override
        def insert(self: Self, index: Any, obj: Paper, /) -> Never:
            raise self.insert(index, (None, obj))

        @overload
        def insert(self: Self, index: Any, obj: Paper, lease: tuple[SignalInstance, list] | None = None, /) -> Never:
            super().insert(index, obj)
            self.leaser.insert(index, lease)
            self.emiters.add(lease)

        @overload
        def insert(self, index: Any, obj: tuple[tuple[SignalInstance, list] | None, Paper], /) -> Never:
            super().insert(index, obj[1])
            self.leaser.insert(index, obj[0])
            self.emiters.add(obj[0])

        @override
        def remove(self: Self, value: Paper) -> None:
            index: int = super().index(value)
            super().pop(index)

            obj: tuple[SignalInstance, list] | None = self.leaser.pop(index)
            if obj[0] is not None:
                obj[1].pop(index)
                self.emiters.add(obj[0]) # Faster than calling each time the emit()

        @overload
        def remove(self: Self, other: tuple[tuple[SignalInstance, list] | None, Paper]) -> None:
            self.remove(other[1])

        def remove_all(self: Self, other: tuple[tuple[SignalInstance, list] | None, Paper] | Paper) -> None:
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
            return (self.leaser[index], super()[index])

        # The rest of the methods will be the same

        def emit(self: Self) -> None:
            for emiter in self.emiters:
                if emiter is not None: emiter.emit()

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
        self.add_but.clicked.connect(lambda ignored: Process(target=self.add).run())
        self.remove_but.clicked.connect(lambda ignored: Process(target=self.remove).run())
        self.find_but.clicked.connect(toggler(self.find_window))

        # Data attributes which are represented as a tuple of a Signal and list
        # This was done for the signals of the Qt API
        self.validation: tuple[SignalInstance, list] = (self.validation_signal, [])
        self.training:   tuple[SignalInstance, list] = (self.training_signal, [])
        self.dataset:    tuple[SignalInstance, Data.JoinedList] = (self.dataset_signal, [])

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
        self.find_window.find_signal.connect(
        lambda : Process(
            target=self.accessPapers,
            args=(lambda papers: papers.find(**self.find_window.sendReport()),)
            ).run()
        )
        self.find_window.remove_signal.connect(
            lambda : Process(
                target=self.accessPapers,
                args=(lambda papers: papers.removeFound(**self.find_window.sendReport()),)
            ).run()
        )

        # Connecting with the finding view for updating
        self.dataset[0].connect(self.__papers.model.updateShowing)
        self.being_modified.connect(self.setDisabled)
        self.being_modified.connect(self.find_window.setDisabled)

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.remove_button.setEnabled(state)
        self.find_window.setEnabled(state)
        self.add_button.setEnabled(state)

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
    def accessPapers(self: Self, function: Callable[[FindingView], None], wait: bool = True) -> bool:
        print("Lock Access")
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
        print("Unlocked Access")

        return res

    """
    A static useful method used for dumping the failures based on the constant CORE_DUMP in Data.

    @author  Thomas Gauthier
    @version 0.0
    """
    @staticmethod
    def dump[Q](failures: list[Q], parent: QWidget | None = None) -> None:
        print("Dumped")
        if not failures:
            funcs.mkabsent(Data.CORE_DUMP)
            with open(Data.CORE_DUMP + "dump.txt", mode="w") as file:
                file.writelines(failures)

            mbFactory(
                "Core Dumped",
                "Parsing of lines failed. Core dumped in the \"files\" directory",
                QMessageBox.Icon.Warning,
                QMessageBox.StandardButton.Ok,
                parent
            ).show()

    # Default callback for the adder
    def add(self: Self, text: str | None = None,  path: str | None = None) -> None:
        _path: Path = Path(self.path.text() if path is not None else path)
        text:   str = self.specifier.itemText() if text is None else text

        # Could also be done with a dictionary
        # But I don't feel like it
        dataset: tuple[SignalInstance, list] | None = None
        match text:
            case "Validation": dataset = self.validation
            case "Training":   dataset = self.training
            case _: dataset = None

        failures: list[tuple[Path, int]] = []
        try: self.accessPapers(lambda ignored: failures.extend(self._recursiveAdd(_path, dataset)))
        except *Exception as ex:
            errorFactory("Error in adding", ex.message + " in " + _path.absolute()).show()

        print(f"Dataset Length: {len(self.dataset[1])}")

        Data.dump(failures, self)

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
            for other in path.iterdir(): failures.extend(self._recursiveAdd(other.absolute()))
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path) as file:
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
                    present: bool = False
                    for paper in self.dataset[1]:
                        if paper[1] == parsed:
                            if paper[0] is not None and paper[0] != dataset:  # The received dataset not the self.dataset
                                paper[0][1].remove(paper[1])
                                dataset[1].append(paper[1])
                                paper[0] = dataset
                            present = True
                            break

                    if not present:
                        self.dataset[1].append((dataset, parsed))
                        if dataset is not None: dataset[1].append(parsed)
        else: raise Exception("Bad file type")

        self.dataset[0].emit()
        self.dataset[1].emit()

        return failures

    # Default callback for the remover
    # Will show a QMessageBox based on the removal process
    def remove(self: Self, path: str | None = None) -> None:
        _path: Path = Path(self.path.text() if path is not None else path)

        failures: list[tuple[Path, str, int]] = []
        try: self.accessPapers(lambda ignored: failures.extend(self._recursivePemove(_path)))
        except *Exception as ex:
            errorFactory("Error in removing", ex.message + " in " + _path.absolute()).show()

        print(f"Dataset Length: {len(self.dataset[1])}")

        Data.dump(failures, self)

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
    def _recursiveRemove(self: Self, path: Path) -> list[tuple[Path, str, int]]:
        failures: list[tuple[Path, str, int]] = []

        if path.is_dir():
            for other in path.iterdir(): failures.extend(self._recursiveRemove(other.absolute()))
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path) as file:
                parsed: Paper | None = None
                count:  int = -1
                for line in file.readlines():
                    count += 1

                    # Must do this manually since the remove function will
                    # Search for a tuple and not an element
                    try: parsed = Paper.parseLine(line, str(path))
                    except:
                        failures.append((path, "parsing", count))
                        continue

                    removal: tuple[tuple[SignalInstance, list] | None, Paper] | None = None
                    for element in self.dataset[1]:
                        if element[1] == parsed:
                            removal = element
                            break

                    if removal is not None: self.dataset[1].remove(removal)
                    else: failures.append((path, "deleting", count))
        else: raise Exception("Bad file type")

        self.dataset[0].emit()
        self.dataset[1].emit()

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
        results:  list = []

        if path.is_dir():
            for other in path.iterdir():
                data: tuple[list[tuple[Path, int]], list] = self.parse(other.absolute())
                failures.extend(data[0])
                results.extend(data[1])
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path) as file:
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

Note that T must be an instance of the Paper class.

@author  Thomas Gauthier, Janosch Ortmann
@version 0.1
"""
@final
class Loading(QDialog, loading.Ui_mainwindow):
    # Default write location of the stems
    DEFAULT_WRITE: Final[str] = CENTRAL + "results/"

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
        self.__samples: list = samples # Not copied here since it will be afterwards

    def getStems(self: Self) -> None:
        # This is why this is not synchronized
        # Introspection for callable. Required later for the function "fit_transform"
        moving: Callable[..., Any] = self.__samples.__getitem__
        self.samples.__getitem__ = lambda index: str(moving(index))

        self.process.setText("Tokenizing papers by their representation...")
        tokens: np.ndarray = np.empty((len(self.__samples), 1))
        length: int = len(self.__samples) # Current Length for the progress bar
        for count in range(length):
            # Faster than to always append
            tokens[count] = wordpunct_tokenize(self.__samples[count].lower())
            self.bar.setValue(round(25 * count / length))
        tokens = np.ravel(tokens)

        self.process.setText("Removing invalid words...")
        sw = stopwords.words('english')
        punct = list(string.punctuation)

        length = len(sw)
        for count in range(length):
            tokens = np.delete(tokens, tokens == sw[count])
            self.bar.setValue(25 + round(10 * count / length))

        length = len(punct)
        for count in range(length):
            tokens = np.delete(tokens, tokens == punct[count])
            self.bar.setValue(35 + round(10 * count / length))

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

        self.process.setText("Stemming words...")
        stemmer: PorterStemmer = PorterStemmer()
        for count in range(len(tokens)):
            tokens[count] = stemmer.stem(tokens[count])
            self.bar.setValue(45 + round(10 * count / length))
        del stemmer, numeric_mask

        self.process.setText("Counting uni grams...")
        count:     np.ndarray =  np.asarray(list(Counter(tokens).items()))
        uni_found: np.ndarray = count[count[:, 1].astype(np.int_) >= self.__min_words]
        del count

        # Removing common words
        uni_found = uni_found[(len(string) >= 4 for string in uni_found[:, 0]), :]

        # Counting each valid gram in the texts. Note that this is required since
        # We want the stems and not the words, thus a CountVectorizer cannot be used
        uni_sample_found: np.ndarray = np.empty((len(self.__samples), uni_found.size[0]))
        for sample_index in range(uni_sample_found.shape[0]):
            for stem_index in range(uni_sample_found.shape[1]):
                uni_sample_found[sample_index, stem_index] = self.__samples[sample_index].count(uni_found[stem_index, 0])
        self.bar.setValue(65)

        # Initializing for bi grams
        self.process.setText("Counting bi grams...")
        vectorizer: CountVectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=sw)
        bi_found: np.ndarray        = vectorizer.fit_transform(self.__samples).toarray()
        bi_words: np.ndarray        = vectorizer.get_feature_names_out()

        def update_bi(respecting: np.ndarray | list[bool]) -> None:
            nonlocal bi_words, bi_found
            bi_words = bi_words[respecting]
            bi_found = bi_found[:, respecting]

        update_bi(np.sum(bi_found, axis=0) >= self.__min_words)                       # Removing grams with not enough words
        update_bi(~np.isin(bi_words, punct))                                          # Removing grams with punctuation
        update_bi(not any(char.isdigit() for char in string) for string in bi_words)  # Or containing digits
        self.bar.setValue(75)

        del update_bi

        # Initializing for tri grams
        self.process.setText("Counting tri grams...")
        vectorizer: CountVectorizer = CountVectorizer(ngram_range=(3, 3), stop_words=sw)
        tri_found:       np.ndarray = vectorizer.fit_transform(self.__samples).toarray()
        tri_words:       np.ndarray = vectorizer.get_feature_names_out()

        def update_tri(respecting: np.ndarray | list[bool]) -> None:
            nonlocal tri_words, tri_found
            tri_words = tri_words[respecting]
            tri_found = tri_found[:, respecting]

        # Removing unacceptable tri grams
        update_tri(np.sum(tri_found, axis=0) >= self.__min_words)                         # Removing grams with not enough words
        update_tri(~np.isin(tri_words, punct))                                            # Removing grams with punctuation
        update_tri((not any(char.isdigit() for char in string) for string in tri_words))  # Or containing digits
        self.bar.setValue(90)

        del update_tri

        self.process.setText("Printing grams...")  # Done for safekeeping
        self.grams: np.ndarray = np.concatenate((uni_found[:, 0], bi_words, tri_words))

        # Shortcut so that you won't have to do this in the MainWindow
        self.grams_found: np.ndarray = np.concatenate((uni_sample_found, bi_found, tri_found), axis=1)
        funcs.mkabsent(Loading.DEFAULT_WRITE)
        pd.DataFrame(self.grams).to_csv(Loading.DEFAULT_WRITE + "stems.txt", sep='\n')

        self.process.setText("Finished")
        self.bar.setValue(100)

        # Removing introspection funnies
        self.__samples.__getitem__ = moving
        self.done.emit()

"""
Widget that represent the first parameters.

This class can also query from the Scopus database, but be advised
that since this does not control the data (that is the role of the Data class),
that means that, after querying, it will emit information about its current state
so that the main window can latch on it widhout doing any modifications to the stems.

The querying starts a new process

@author  Thomas Gautier, Janosch Ortmann
@version 0.1
"""
@final
class First(QWidget, first.Ui_first_option):
    # The default setting for when the directory isn't specified
    DEFAULT_QUERY: Final[str] = CENTRAL + "query/"
    # The number of parallel processing threads
    POOL_COUNT:    Final[int] = 4
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

        self.__lock:      Lock = Lock()
        # Shortcut for declaring the processes
        shortcut: Callable[[SignalInstance, QEvent], None] = lambda signal: Process(
            target=(lambda : (self.query(), signal.emit()))
        ).run()

        # Asynchronously querying
        self.remove.clicked.connect(lambda ignored: shortcut(self.remove_signal))
        self.add.clicked.connect(lambda ignored: shortcut(self.add_signal))

        # Deactivate multiple querying at the same time
        self.querying.connect(self.setDisabled)

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.add.setEnabled(state)
        self.remove.setEnabled(state)

    @override
    def setDisabled(self: Self, state: bool) -> None:
        return self.setEnabled(not state)

    async def query(self: Self) -> None:
        self.__lock.acquire(blocking=False)

        """
        --------------------------------------------------
                        Parameter validation
        --------------------------------------------------
        """
        self.querying.emit(True)

        # Temporary variable
        text: str = self.directory_edit.text()
        directory: Path = Path(text if text is not None else First.DEFAULT_QUERY)
        del text

        if not osp.exists(directory) or not osp.isdir(directory):
            errorFactory(
                "Invalid Directory",
                "The directory given was invalid",
                self
            ).show()
            return

        self.directory: Path = directory
        del directory

        self.setEnabled(False)
        self.loading: QProgressBar = QProgressBar()
        self.loading.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.variables.addChildWidget(self.loading)

        if not osp.exists(Parameters.FILE) or not osp.isfile(Parameters.FILE):
            errorFactory(
                "Parameters not saved",
                "The parameters file was not found",
                self
            ).show()
            return

        val_per: float = 0
        bad_val: bool = False
        try: val_per = float(self.sample_edit.text())
        except: bad_val = True
        if val_per <= 0 or 1 <= val_per or bad_val:
            errorFactory(
                "Bad validation size",
                "The validation size isn't a valid percentage.",
                self
            ).show()
            return
        del bad_val

        sample_per: float = 0
        bad_sample: bool = False
        try: sample_per = float(self.sample_edit.text())
        except: bad_sample = True
        if sample_per <= 0 or 1 <= sample_per or val_per + sample_per >= 1 or bad_sample:
            errorFactory(
                "Bad sample size",
                "The sample size isn't a valid percentage.",
                self
            ).show()
            return
        del bad_sample

        params: dict[str, str] = appendParams()

        """
        --------------------------------------------------
                            Querying
        --------------------------------------------------
        """

        limit: int = int(params[Parameters.LABELS["limit"]])
        processed: Any = 0

        # Left as string to print them in file, where the parsing will happen later
        received: dict[str, str] = {
            "titles": [],
            "abstracts": [],
            "journals": [],
            "dates": [],
            "doi": [],
            "missing": []
        }

        self.loading.setMaximum(limit)

        while processed < limit:
            processed += First.PROCESSING
            url: str = f"https://api.elsevier.com/content/search/scopus?apiKey={params[Parameters.LABELS["api"]]}&query={quote(self.query.document().toPlainText())}&view=\"COMPLETE\"&start={processed}&count={First.COUNT}"

            entry: Any = None
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, raise_for_status=True) as response:
                        entries = await response.json()['search-results']
                        if not entry: break # No more entries => Nothing else to fetch
            except:
                errorFactory(
                    "Critical Error",
                    "Fetching of scopus failed. Please check your internet connection.",
                    self
                ).show()
                return

            # Assured that entry is the response from the server
            for entry in entries:
                if not entry["dc:description"]: received["missing"].append()
                else:
                    shorting: Callable[..., Any] = lambda x: "Unknown" if x is None else x
                    received["abstracts"].append(entry["dc:description"])
                    received["doi"].append(shorting(entry["prism:doi"]))
                    received["titles"].append(shorting(title_value = entry['dc:title']))
                    received["journals"].append(shorting(entry['prism:publicationName']))
                    received["dates"].append(shorting(entry['prism:coverDate']))

            self.loading.setValue(processed)

        # In case it wasn't fully loaded
        self.loading.setValue(self.loading.maximum())
        del limit, params, processed, start

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
        sample:     np.ndarray = citations[sample_index, :]

        validation_index: list[int] = random.sample(range(number), math.ceil(number * val_per))
        validation: np.ndarray = citations[validation_index, :]

        del number, sample_index, validation_index

        funcs.mkabsent(First.DEFAULT_QUERY)
        # Shortcut
        printer: Callable[..., None] = lambda path, data: pd.DataFrame(data).to_csv(path, quoting=csv.QUOTE_ALL)

        printer(self.directory + First.FILES[0], validation)
        printer(self.directory + First.FILES[1], citations)
        printer(self.directory + First.FILES[2], sample)

        # Cleaning up
        self.setEnabled(True)
        self.variables.removeWidget(self.loading)
        del self.loading

        self.querying.emit(False)
        self.__lock.release()

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

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.clear.setEnabled(state)
        self.plot.setEnabled(state)

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
    params: dict[str, str] = {}
    with open(Parameters.FILE) as file:
        js: Any = json.loads(file.read())

        for param in Parameters.LABELS.keys():
            try: params[param] = js[param]
            except:
                errorFactory(
                    "Bad argument",
                    "Parameter received had an error (" + param + ')'
                ).show()
                return
    return params
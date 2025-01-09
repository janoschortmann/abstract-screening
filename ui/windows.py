#!/usr/bin/env python3
"""
File containing all instances of the windows defined with the xml files.
It also connects the signals/callbacks of those windows.

@author  Thomas Gauthier
@version 0.0
"""
from functools         import reduce
from pathlib           import Path
from threading         import Lock
from typing            import *
from PySide6.QtWidgets import QWidget, QMessageBox, QProgressBar, QSizePolicy
from PySide6.QtCore    import QEvent, QObject

from python.src.utils.files     import Paper
from python.src.utils.functions import toggler
from ui.compiled                import params, data, find
from ui.compiled.options        import first, second, third
from ui.display.entities        import FindingView, mb_factory, error_factory

# Web and async
from urllib.parse import quote
import asyncio
import aiohttp

import json
import math
import random

import numpy   as np
import os.path as osp
import pandas  as pd

"""
Default implementation of the parameters window.

@author  Thomas Gauthier
@version 0.0
"""
@final
class Parameters(QWidget, params.Ui_mainwindow):
    # Tuple representing the labels associated with the line edits
    # Dictionary between their name in code and the ones printed out
    LABELS: Final[dict[str, str]] = {"limit": "Limit", "api": "API Key", "token": "Token"}
    FILE:   Final[str] = "./files/params.json"

    # Default initializer
    def __init__(self: Self) -> None:
        # Setting up the ui
        super().__init__()
        self.setupUi(self)

        # Callback
        self.save_info.clicked.connect(self.print_info)

        if osp.exists(Parameters.FILE) and osp.isfile(Parameters.FILE):
            with open(Parameters.FILE) as file:
                values: Any = json.loads(reduce(lambda acc, other: acc + other, file.readlines()))

                # Shortcut to catch errors
                def get(key: str) -> str:
                    nonlocal values
                    try: return values[key]
                    except: return ""

                self.query_input.setText(get(Parameters.LABELS["limit"]))
                self.api_input.setText(get(Parameters.LABELS["api"]))
                self.token_input.setText(get(Parameters.LABELS["token"]))

    # Prints the data in the line edits to the json file
    def print_info(self: Self, event: QEvent) -> None:
        with open(Parameters.FILE, mode="w") as file:
            file.writelines(
                json.dumps(
                    {
                        Parameters.LABELS["limit"]: self.query_input.text(),
                        Parameters.LABELS["api"]: self.api_input.text(),
                        Parameters.LABELS["token"]: self.token_input.text()
                    },
                    indent=4
                ).__str__()
            )
        self.save_info.setText("Saved!")
        event.accept()

"""
A window for finding documents and removing them.
Is used, for example, in the FindTableView.

Note that it does not directly access the data
but only emits signals for the table to connect to.

@author  Thomas Gauthier
@version 0.0
"""
@final
class Find(QWidget, find.Ui_mainwindow):
    # Default initializer
    def __init__(self: Self, title: str | None = None) -> None:
        #Initializing
        super().__init__()
        self.setupUi(self)
        self.label_box.addItems(Paper.POSSIBILITIES)

        # Custom title to differentiate different find windowr
        if title is not None:
            self.setWindowTitle(title)

        # Signals
        self.remove_signal: QObject = QObject()
        self.find_signal:   QObject = QObject()

        # Callbacks
        self.find_but.clickled.clicked.connect(self.find_signal.emit)
        self.clear_search.clicked.connect(self.remove_signal.emit)
        self.clear_edits.clicked.connect(self.clear)

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.find_but.setEnabled(state)
        self.clear_search.setEnabled(state)
        self.clear_edits.setEnabled(state)

    # Clear all edits
    def clear(self: Self) -> None:
        for edit in (self.title_edit, self.journal_edit, self.doi_edit, self.label): edit.clear()

    # Sends a report of the given values that the user assigned
    # A quick shortcut with kwargs manipulation
    def send_report(self: Self) -> Any:
        arguments: dict[str, Any] = {}

        # Shortcuts
        def append_not_empy[Q](name: str, arg: Q | None) -> None:
            nonlocal arguments
            if not arg: arguments[name] = arg

        # Note that these names correspond to the args in the "find" function
        # In the FindingView so that when calling "find" with "send_report",
        # You can assign the parameters directly with **report
        append_not_empy("date",    (self.from_date.date().toPython(), self.to_date.date().toPython()))
        append_not_empy("journal", (self.journal_regex.isChecked(), self.journal_edit.text()))
        append_not_empy("title",   (self.title_regex.isChecked(), self.title_edit.text()))
        append_not_empy("label",   self.label_box.currentText())
        append_not_empy("doi",     self.doi_edit.text())


"""
The class representing the complete dataset.
That imply that this manages all the data by itself.

Note that T must be an instance of the Paper class.

It also supports multithreading.

@author  Thomas Gauthier
@version 0.1
"""
@final
class Data[T](QWidget, data.Ui_mainwindow):
    # Main file where the dumps are..... well..... dumped
    CORE_DUMP: Final[str] = "./files/dump.txt"

    """
    An inner class used to link up the different kinds of
    datasets.

    This is the easiest option since the model from the TableView
    can modify the data without this class being notified.
    """
    """
    Todo : Turns out that this original Band-Aid solution works quite nicely
    whist still preserving the overall encapsulation. When someone has more time,
    creating an "official" version might be nice... You could name it delegator or something; idk...
    """
    class JoinedList(list):
        @overload
        def __init__(self: Self) -> None:
            super().__init__()
            self.leaser: list[tuple[QObject, list[T]] | None] = []
            self.emiters: set[QObject] = {}

        @override
        def __init__(self: Self, iterable: Iterable[tuple[QObject, list[T]] | None, T] | Iterable[T], /) -> None:
            super.__init__()
            self.leaser: list[tuple[QObject, list[T]] | None] = []
            self.emiters: set[QObject] = {}
            self.extend(iterable)

        @override
        def copy(self: Self) -> list[T]:
            clone: Data.JoinedList[T] = []

            for item in range(len(self) - 1):
                clone.append((self[item], self.leaser[item]))

            return clone

        @override
        def append(self: Self, instance: T) -> Never:
            self.append((None, instance))

        @overload
        def append(self: Self, instance: T, lease: tuple[QObject, list[T]] | None = None, /) -> None:
            super().append(instance)
            self.leaser.append(lease)
            self.emiters.add(lease)

        @overload
        def append(self: Self, combined: tuple[tuple[QObject, list[T]] | None, T], /) -> None:
            super().append(combined[1])
            self.leaser.append(combined[0])
            self.emiters.add(combined[0])

        @override
        def extend(self: Self, iterable: Iterable[tuple[QObject, list[T]] | None, T] | Iterable[T], /) -> None:
            for item in iterable: self.append(item)

        @override
        def pop(self: Self, index: Any = -1, /) -> tuple[tuple[QObject, list[T]] | None, T]:
            val: tuple[tuple[QObject, list[T]]] | None = self.leaser.pop(index)
            self.emiters.add(val)
            return (val, super().pop(index))

        @override
        def insert(self: Self, index: Any, obj: T, /) -> Never:
            raise self.insert(index, (None, obj))

        @overload
        def insert(self: Self, index: Any, obj: T, lease: tuple[QObject, list[T]] | None = None, /) -> Never:
            super().insert(index, obj)
            self.leaser.insert(index, lease)
            self.emiters.add(lease)

        @overload
        def insert(self, index: Any, obj: tuple[tuple[QObject, list[T]] | None, T], /) -> Never:
            super().insert(index, obj[1])
            self.leaser.insert(index, obj[0])
            self.emiters.add(obj[0])

        @override
        def remove(self: Self, value: T) -> None:
            index: int = super().index(value)
            super().pop(index)

            obj: tuple[QObject, list[T]] | None = self.leaser.pop(index)
            if obj[0] is not None:
                obj[1].pop(index)
                self.emiters.add(obj[0]) # Faster than calling each time the emit()

        @overload
        def remove(self: Self, other: tuple[tuple[QObject, list[T]] | None, T]) -> None:
            self.remove(other[1])

        @override
        def sort(self: Self, *, key: Callable[..., Any], reverse: bool = False) -> Never:
            raise Exception("Not implemented")

        @override
        def __iter__(self: Self) -> Iterator[tuple[tuple[QObject, list[T]] | None, T]]:
            self.counter: int = 0
            return self

        @override
        def __next__(self: Self) -> tuple[tuple[QObject, list[T]] | None, T]:
            if self.counter == len(self):
                del self.counter
                raise StopIteration()
            item: tuple[tuple[QObject, list[T]] | None, T]  = self[self.counter]
            self.counter += 1
            return item

        @override
        def __getitem__(self: Self, index: Any, /) -> tuple[tuple[QObject, list[T]] | None, T]:
            return (self.leaser[index], super()[index])

        @override
        def __setitem__(self: Self, key: slice, value: Iterable[T], /) -> None:
            raise Exception("Not implemented")

        @override
        def __delitem__(self: Self, key: Any | slice, /) -> None:
            raise Exception("Not implemented")

        @override
        def __add__(self: Self, value: list[T], /) -> list[T]:
            raise Exception("Not implemented")

        @override
        def __add__[S](self: Self, value: list[S], /) -> list[S | Any]:
            raise Exception("Not implemented")

        @override
        def __iadd__(self: Self, value: Iterable[T], /) -> Self:
            raise Exception("Not implemented")

        @override
        def __mul__(self: Self, value: Any, /) -> list[T]:
            raise Exception("Not implemented")

        @override
        def __rmul__(self: Self, value: Any, /) -> list[T]:
            raise Exception("Not implemented")

        @override
        def __imul__(self: Self, value: Any, /) -> Self:
            raise Exception("Not implemented")

        @override
        def __contains__(self: Self, key: T, /) -> bool:
            return key in super()

        @override
        def __reversed__(self: Self) -> Iterator[T]:
            raise Exception("Not implemented")

        @override
        def __gt__(self: Self, value: list[T], /) -> bool:
            raise Exception("Not implemented")

        @override
        def __ge__(self: Self, value: list[T], /) -> bool:
            raise Exception("Not implemented")

        @override
        def __lt__(self: Self, value: list[T], /) -> bool:
            raise Exception("Not implemented")

        @override
        def __le__(self: Self, value: list[T], /) -> bool:
            raise Exception("Not implemented")

        @override
        def __eq__(self: Self, value: object, /) -> bool:
            raise Exception("Not implemented")

        @override
        def __class_getitem__(cls, item: Any, /) -> Any:
            raise Exception("Not implemented")

        def emit(self: Self) -> None:
            for emiter in self.emiters:
                if emiter is not None: emiter.emit()

    # Default constructor
    def __init__(self: Self) -> None:
        # Default initializer
        super().__init__()
        self.setupUi(self)
        self.find_window: Find = Find("Finder for Data")

        # Multithreading and parallelism
        self.being_modified: QObject = QObject()
        self.__lock: Lock = Lock()

        # Callbacks
        self.add_button.clicked.connect(self.add)
        self.remove_button.clicked.connect(self.remove)
        self.find_but.clicked.connect(toggler(self.find_window))

        # Data attributes which are represented as a tuple of a QObject and list
        # This was done for the signals of the Qt API
        self.validation: tuple[QObject, list[T]] = (QObject(), [])
        self.training:   tuple[QObject, list[T]] = (QObject(), [])
        self.dataset:    tuple[QObject, Data.JoinedList[T]] = (QObject(), [])

        # Widgets
        self.__papers: FindingView = FindingView(self.dataset[1])
        self.specifier.addItems(["Training", "Validation", "None"])

        # Connect callbacks in the find window
        self.find_window.find_signal.connect(lambda : self.access_papers(lambda papers: papers.find(**self.find_window.send_report())))
        self.find_window.remove_signal.connect(lambda : self.access_papers(lambda papers: papers.remove_found(**self.find_window.send_report())))

        # Connecting with the finding view for updating
        # Side Note : It is a true pain of connecting all the data between three different classes
        # that must be synchronized. Espectially when it must be fast.
        self.dataset[0].connect(self.__papers.model.update_showing)
        self.being_modified.connect(self.setEnabled)

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.find_window.setEnabled(state)
        self.add_button.setEnabled(state)
        self.remove_button.setEnabled(state)

    """
    Function used to access the papers and call a function
    upon them.

    This is used since this data window will be multithreaded.

    Note that this calls, automatically, the functions self.dataset[0].emit() and self.dataset[1].emit()
    """
    def access_papers(self: Self, function: Callable[[FindingView], None], wait: bool = False) -> bool:
        # Note that, in theory, the result shouldn't be necessary, since you could
        # Very well just get the state from the QObject of this widget, but this could still be useful in the future
        res: bool = self.__lock.acquire(wait)
        if not res: return res

        self.being_modified.emit(True)
        function(self.__papers)
        self.being_modified.emit(False)
        self.__lock.release()

        self.dataset[0].emit()
        self.dataset[1].emit()

        return res


    """
    A static useful method used for dumping the failures
    based on the constant CORE_DUMP in Data.

    @author  Thomas Gauthier
    @version 0.0
    """
    @staticmethod
    def dump[Q](failures: list[Q], parent: QWidget | None = None) -> None:
        if not failures:
            with open(Data.CORE_DUMP, mode="w") as file:
                file.writelines(failures)

            mb_factory(
                "Core Dumped",
                "Parsing of lines failed. Core dumped in the \"files\" directory",
                QMessageBox.Icon.Warning,
                QMessageBox.StandardButton.Ok,
                parent
            ).exec()

    # Default callback for the adder
    def add(self: Self, event: QEvent) -> None:
        path: Path = Path(self.path.text)
        text:  str = self.specifier.itemText()

        # Could also be done with a dictionary
        # But I don't feel like it
        dataset: tuple[QObject, list[T]] = None
        match text:
            case "Training": dataset = self.training
            case "Validation": dataset = self.validation
            case _: dataset = None

        failures: list[tuple[Path, int]] = []
        try:
            self.being_modified.emit(True)
            self.__lock.acquire()
            failures.extend(self._recursive_add(path, dataset))
            self.__lock.release()
            self.being_modified.emit(False)
        except *Exception as ex:
            error_factory("Error in adding", ex.message + " in " + path.absolute()).exec()


        if not failures:
            Data.dump(failures, self)
            mb_factory(
                    "Failure dump",
                    "Some failures occured while querying. Dumped in " + Data.CORE_DUMP,
                    QMessageBox.Icon.Information,
                    None,
                    self
                )

        event.accept()

    """
    A method used for recursive adding.
    Returns a list of lines which could not be pared.

    Note that this method is not *required* (since you could only call the parse method
    and then remove all hits), but this is faster since you don't need to append to a list
    and then remove it, which would bring the runtime at twice the time.
    """
    def _recursive_add(self: Self, dataset: tuple[QObject, list[T]] | None, path: Path) -> list[tuple[Path, int]]:
        failures: list[tuple[Path, int]] = []
        if path.is_dir():
            for other in path.iterdir():
                failures.extend(self._recursive_add(other.absolute()))
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path) as file:
                parsed: T | None = None
                count:  int = -1

                for line in file.readlines():
                    count += 1

                    try: parsed = T.parse_line(line)
                    except:
                        failures.append((path, count))
                        continue

                    # Parsed will not be None
                    # Must be done manually to find the lines
                    present: bool = False
                    for paper in self.dataset[1]:
                        if paper[1] == parsed:
                            if paper[0] is not None and paper[0] != dataset:
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
    def remove(self: Self, event: QEvent) -> None:
        path: Path = Path(self.path.text())

        failures: list[tuple[Path, str, int]] = []
        try:
            self.being_modified.emit(True)
            self.__lock.acquire()
            failures.extend(self._recursive_remove(path))
            self.__lock.release()
            self.being_modified.emit(False)
        except *Exception as ex:
            error_factory("Error in removing", ex.message + " in " + path.absolute()).exec()

        if not failures:
            Data.dump(failures, self)
            mb_factory(
                    "Failure dump",
                    "Some failures occured while querying. Dumped in " + Data.CORE_DUMP,
                    QMessageBox.Icon.Information,
                    None,
                    self
                )
        event.accept()

    """
    A method used for recursive adding.
    Returns a list of lines which could not be pared.

    Note that this method is not *required* (since you could only call the parse method
    and then remove all hits), but this is faster since you don't need to append to a list
    and then remove it, which would bring the runtime at twice the time.
    """
    def _recursive_remove(self: Self, path: Path) -> list[tuple[Path, str, int]]:
        failures: list[tuple[Path, str, int]] = []

        if path.is_dir():
            for other in path.iterdir():
                failures.extend(self._recursive_remove(other.absolute()))
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path) as file:
                parsed: T | None = None
                count:  int = -1
                for line in file.readlines():
                    count += 1

                    # Must do this manually since the remove function will
                    # Search for a tuple and not an element
                    try: parsed = T.parse_line(line)
                    except:
                        failures.append((path, "parsing", count))
                        continue

                    removal: tuple[tuple[QObject, list[T]] | None, T] | None = None
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
    def parse(self: Self, path: Path) -> tuple[list[tuple[Path, int]], list[T]]:
        failures: list[tuple[Path, int]] = []
        results:  list[T] = []

        if path.is_dir():
            for other in path.iterdir():
                data: tuple[list[tuple[Path, int]], list[T]] = self.parse(other.absolute())
                failures.extend(data[0])
                results.extend(data[1])
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            with open(path) as file:
                parsed: T | None = None
                count:   int = -1

                for line in file.readlines():
                    count += 1

                    try: parsed = T.parse_line(line)
                    except:
                        failures.append((path, count))
                        continue

                    results.append(parsed)
        else: raise Exception("Bad file type")
        return (failures, results)


"""
Widget that represent the first parameters.

This class can also query from the Scopus database, but be advise
that since this does not control the data (that is the role of the Data class),
that means that, after querying, it will emit information about its current state
so that the main window can latch on it.

@author  Thomas Gautier, Janosch Ortmann
@version 0.0
"""
class First(QWidget, first.Ui_first_option):
    # The default setting for when the directory isn't specified
    DEFAULT_QUERY: Final[str] = "./files/query/"
    # The number of parallel processing threads
    POOL_COUNT:    Final[int] = 4
    # Number of queries per iteration
    COUNT:         Final[int] = 100

    # Default initializer
    def __init__(self: Self) -> None:
        super().__init__()
        self.setupUi(self)

        self.remove_signal: QObject = QObject()
        self.add_signal:    QObject = QObject()
        self.find_signal:   QObject = QObject()

        self.add.clicked.connect(lambda event: event.accept(), asyncio.run(self.query()), self.add_signal.emit())
        self.find_but.clicked.connect(lambda event: event.accept(), asyncio.run(self.query()), self.find_signal.emit())
        self.remove.clicked.connect(lambda event: event.accept(), asyncio.run(self.query()), self.remove_signal.emit())

    @override
    def setEnabled(self: Self, state: bool) -> None:
        self.add.setEnabled(state)
        self.find_but.setEnabled(state)
        self.remove.setEnabled(state)

    async def query(self: Self) -> None:
        """
        --------------------------------------------------
                            Parameter validation
        --------------------------------------------------
        """
        # Temporary variable
        text: str = self.directory_edit.text()
        directory: Path = Path(text if text is not None else First.DEFAULT_QUERY)
        del text

        if not osp.exists(directory) or not osp.isdir(directory):
            error_factory(
                "Invalid Directory",
                "The directory given was invalid",
                self
            ).exec()
            return

        self.directory: Path = directory
        del directory

        self.setEnabled(False)
        self.loading: QProgressBar = QProgressBar()
        self.loading.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.variables.addChildWidget(self.loading)

        if not osp.exists(Parameters.FILE) or not osp.isfile(Parameters.FILE):
            error_factory(
                "Parameters not saved",
                "The parameters file was not found",
                self
            ).exec()
            return

        val_per: float = 0
        bad_val: bool = False
        try: val_per = float(self.sample_edit.text())
        except: bad_val = True
        if val_per <= 0 or 1 <= val_per or bad_val:
            error_factory(
                "Bad validation size",
                "The validation size isn't a valid percentage.",
                self
            ).exec()
            return
        del bad_val

        sample_per: float = 0
        bad_sample: bool = False
        try: sample_per = float(self.sample_edit.text())
        except: bad_sample = True
        if sample_per <= 0 or 1 <= sample_per or val_per + sample_per >= 1 or bad_sample:
            error_factory(
                "Bad sample size",
                "The sample size isn't a valid percentage.",
                self
            ).exec()
            return
        del bad_sample

        params: dict[str] = []
        with open(Parameters.FILE) as file:
            js: Any = json.loads(reduce(lambda acc, other: acc + other, file.readlines()))

            for param in Parameters.LABELS:
                try: params[param] = js[param]
                except:
                    error_factory(
                        "Bad argument",
                        "Parameter received had an error (" + param + ')',
                        self
                    ).exec()
                    return

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
            url: str = f"https://api.elsevier.com/content/search/scopus?apiKey={param[Parameters.LABELS["api"]]}&query={quote(self.query.document().toPlainText())}&view=\"COMPLETE\"&start={processed}&count={First.COUNT}"

            entry: Any = None
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, raise_for_status=True) as response:
                        entries = await response.json()['search-results']
                        if not entry: break # No more entries => Nothing else to fetch
            except:
                error_factory(
                    "Critical Error",
                    "Fetching of scopus failed. Please check your internet connection.",
                    self
                ).exec()
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
            axis=1
        )

        number: Final[int] = len(citations)

        sample_index: list = random.sample(number - 1, math.ceil(number * sample_per))
        sample:     np.ndarray = citations[sample_index, :]
        citations = np.delete(citations, sample_index, axir=0)

        validation_index: list = random.sample(number - 1, math.ceil(number * val_per))
        validation: np.ndarray = citations[validation_index, :]
        citations = np.delete(citations, validation_index, axis=0)

        del number, sample_index, validation_index

        # Shortcut
        printer: Callable[..., None] = lambda path, data: pd.DataFrame(data).to_csv(path)

        printer(self.directory + "validation.csv", validation)
        printer(self.directory + "citations.csv", citations)
        printer(self.directory + "sample.csv", sample)

        # Cleaning up
        self.setEnabled(True)
        self.variables.removeWidget(self.loading)
        del self.loading
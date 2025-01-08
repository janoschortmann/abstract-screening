#!/usr/bin/env python3
"""
File containing all instances of the windows defined with the xml files.
It also connects the signals/callbacks of those windows.

@author  Thomas Gauthier
@version 0.0
"""
from functools         import reduce
from pathlib           import Path
from typing            import (
                                Any,
                                Callable,
                                Iterable,
                                Iterator,
                                Final,
                                Never,
                                Self,
                                overload,
                                override,
                                final
                              )
from PySide6.QtWidgets import QWidget, QMessageBox
from PySide6.QtCore    import QEvent, QObject

from python.src.utils.files     import Paper
from python.src.utils.functions import toggler
from ui.compiled                import params, data, find
from ui.compiled.options        import first, second, third
from ui.display.entities        import FindingView, mb_factory, error_factory

import json
import os.path as osp

"""
Default implementation of the parameters window.

@author  Thomas Gauthier
@version 0.0
"""
@final
class Parameters(QWidget, params.Ui_mainwindow):
    # Tuple representing the labels associated with the line edits
    LABELS: tuple[str] = ("Limit", "API Key", "Token")
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

                self.query_input.setText(get(Parameters.LABELS[0]))
                self.api_input.setText(get(Parameters.LABELS[1]))
                self.token_input.setText(get(Parameters.LABELS[2]))

    # Prints the data in the line edits to the json file
    def print_info(self: Self, event: QEvent) -> None:
        with open(Parameters.FILE, mode="w") as file:
            file.writelines(
                json.dumps(
                    {
                        Parameters.LABELS[0]: self.query_input.text(),
                        Parameters.LABELS[1]: self.api_input.text(),
                        Parameters.LABELS[2]: self.token_input.text()
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
        self.find.clickled.clicked.connect(self.find_signal.emit)
        self.clear_search.clicked.connect(self.remove_signal.emit)
        self.clear_edits.clicked.connect(self.clear)

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

@author  Thomas Gauthier
@version 0.0
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
            for item in iterable: self.append(item)

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

        @overload
        def append(self: Self, combined: tuple[tuple[QObject, list[T]] | None, T], /) -> None:
            super().append(combined[1])
            self.leaser.append(combined[0])

        @override
        def extend(self: Self, iterable: Iterable[tuple[QObject, list[T]] | None, T] | Iterable[T], /) -> None:
            for item in iterable: self.append(item)

        @override
        def pop(self: Self, index: Any = -1, /) -> tuple[tuple[QObject, list[T]] | None, T]:
            return (self.leaser.pop(index), super().pop(index))

        @override
        def insert(self, index: Any, obj: T, /) -> Never:
            raise self.insert(index, (None, obj))

        @overload
        def insert(self, index: Any, obj: T, lease: tuple[QObject, list[T]] | None = None, /) -> Never:
            super().insert(index, obj)
            self.leaser.insert(index, lease)

        @overload
        def insert(self, index: Any, obj: tuple[tuple[QObject, list[T]] | None, T], /) -> Never:
            super().insert(index, obj[1])
            self.leaser.insert(index, obj[0])

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
                emiter.emit()

    # Default constructor
    def __init__(self: Self) -> None:
        # Default initializer
        super().__init__()
        self.setupUi(self)
        self.find_window: Find = Find("Finder for Data")

        # Callbacks
        self.add_button.clicked.connect(self.add)
        self.remove_button.clicked.connect(self.remove)
        self.find.clicked.connect(toggler(self.find_window))

        # Data attributes which are represented as a tuple of a QObject and list
        # This was done for the signals of the Qt API
        self.validation: tuple[QObject, list[T]] = (QObject(), [])
        self.training:   tuple[QObject, list[T]] = (QObject(), [])
        self.dataset:    tuple[QObject, Data.JoinedList[T]] = (QObject(), [])

        # Widgets
        self.papers: FindingView = FindingView(self.dataset[1])
        self.specifier.addItems(["Training", "Validation", "None"])

        # Connect callbacks in the find window
        self.find_window.find_signal.connect(lambda : self.papers.find(**self.find_window.send_report()))
        self.find_window.remove_signal.connect(lambda : self.papers.remove_found(**self.find_window.send_report()), self.dataset[1].emit())

        # Connecting with the finding view for updating
        # Side Note : It is a true pain of connecting all the data between three different classes
        # that must be synchronized. Espectially when it must be fast.
        self.dataset[0].connect(self.papers.model.update_showing)

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
        try: failures.extend(self.recursive_add(path, dataset))
        except *Exception as ex:
            error_factory("Error in adding", ex.message + " in " + path.absolute()).exec()

        Data.dump(failures, self)
        event.accept()

    # A method used for recursive adding
    # Returns a list of lines which could not be parsed
    # Todo : Make this parallel
    def recursive_add(self: Self, dataset: tuple[QObject, list[T]] | None, path: Path) -> list[tuple[Path, int]]:
        if path.is_dir():
            failures: list[tuple[Path, int]] = []
            for other in path.iterdir():
                failures.extend(self.recursive_add(other.absolute()))
            return failures
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            failures: list[tuple[Path, int]] = []
            has_target: Final[bool] = dataset is None
            with open(path) as file:
                count: int = 0
                for line in file.readlines():
                    parsed: T | None = None
                    try: parsed = T.parse_line(line)
                    except:
                        failures.append((path, count))
                        continue
                    finally: count += 1

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
                        if has_target: dataset[1].append(parsed)

            if has_target: dataset[0].emit()
            self.dataset[0].emit()

            return failures
        else:
            raise Exception("Bad file type")

    # Default callback for the remover
    # Will show a QMessageBox based on the removal process
    def remove(self: Self, event: QEvent) -> None:
        path: Path = Path(self.path.text())

        failures: list[tuple[Path, str, int]] = []
        try: failures.extend(self.recursive_remove(path))
        except *Exception as ex:
            error_factory("Error in removing", ex.message + " in " + path.absolute()).exec()

        Data.dump(failures, self)
        event.accept()

    # Useful method for removing recursively
    def recursive_remove(self: Self, path: Path) -> list[tuple[Path, str, int]]:
        if path.is_dir():
            failures: list[tuple[Path, str, int]] = []
            for other in path.iterdir():
                failures.extend(self.recursive_remove(other.absolute()))
            return failures
        elif path.is_file():
            # Could maybe change this if it becomes a problem
            if not path.suffix == ".csv" and not path.suffix == ".txt": return

            failures: list[tuple[Path, str, int]] = []
            with open(path) as file:
                count: int = -1
                for line in file.readlines():
                    count += 1

                    # Must do this manually since the remove function will
                    # Search for a tuple and not an element
                    paper: T | None = None
                    try: paper = T.parse_line(line)
                    except:
                        failures.append((path, "parsing", count))
                        continue

                    removal: tuple[tuple[QObject, list[T]] | None, T] | None = None
                    for element in self.dataset[1]:
                        if element[1] == paper:
                            removal = element
                            break

                    if removal is not None: self.dataset[1].remove(removal)
                    else: failures.append((path, "deleting", count))

            self.dataset[0].emit()
            self.dataset[1].emit()

            return failures
        else:
            raise Exception("Bad file type")
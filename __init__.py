#!/usr/bin/env python3
"""
Main file setting up the UI.

@author  Thomas Gauthier
@version 0.2
"""
import sys

# Done to access all newer features of the typing library such as generics
if sys.version_info < (3, 12):
    sys.stderr.write("Python version inferior to 3.12+. Please make sure to update to a newer version.")
    sys.exit()

from PySide6.QtWidgets import (
                                QApplication,
                                QDialog,
                                QDialogButtonBox,
                                QMainWindow,
                                QMessageBox,
                                QPushButton,
                                QSizePolicy
                              )
from PySide6.QtCore    import QEvent

# Supposed to run first as the name of the file says
# Also, this was done because imports and qt were bothering me
# With imports. That's also why the app is instatiated right after
if __name__ != "__main__":
    sys.exit(-1)

# I <3 "QPixmap: Must construct a QGuiApplication before a QPixmap"
app: QApplication = QApplication([])

from multiprocessing import Barrier, Lock, Queue, Process, current_process
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree    import DecisionTreeClassifier
from typing          import Final, Self, final
from webbrowser      import open_new_tab

from python.src.utils.files     import Paper
from python.src.utils.functions import appendParams, cutoff, toggler
from ui.display.entities        import FindingView, OperatingCurve, SelectionView, SelectionModel
from ui.compiled import mainwindow
from ui.windows  import *

import numpy   as np
import sklearn as sk
import sklearn.model_selection as ms
import sklearn.linear_model    as lm

"""
The main window of the UI.

The main job of the MainWindow class is to synchronize all datasets
from the Data class to everything else.

It sets up the callacks and back processes.

The MainWindow class does not manage directly the data (it synchronizes it) since that
is done with the Data class (which also manages the data between the view,
the backend and sets up parallelism between multiple get and set operations).

You can then say, in gruesome terms, this does the "logic" of the app whilst the Data class
does the "synchronizing" of the app.

@author  Thomas Gauthier
@version 0.1
"""
@final
class MainWindow(QMainWindow, mainwindow.Ui_mainwindow):
    # Github link for the "about" window
    GITHUB_LINK: Final[str] = "https://github.com/janoschortmann/abstract-screening"
    DEFAULT_FILE:       Final[str] = "~/.ACAS/results/papers.txt"

    # The initializer of the window.
    def __init__(self: Self) -> None:
        # Setting up the UI
        super().__init__()
        self.setupUi(self)

        # Other windows
        self.params_window: Parameters = Parameters()
        self.data_window:         Data = Data()

        # Setting up the callbacks
        self.params.clicked.connect(toggler(self.params_window))
        self.data.clicked.connect(toggler(self.data_window))
        self.about.clicked.connect(lambda event: (open_new_tab(MainWindow.GITHUB_LINK), event.accept()))

        # Connecting the data window with the central widget
        # Todo : This

        self.show()

    # Function that sets up the first step of the procedure.
    def mountFirst(self: Self) -> None:
        # Initializing the variables for the finding window
        self.find_but: QPushButton = QPushButton(self)
        self.find_window:     Find = Find("Finder for Central Widget", self)

        # Step defined widgets
        self.display = FindingView(self.data_window.training[1], Model=SelectionModel, View=SelectionView, parent=self)
        self.options = First(self)


        """
        --------------------------------------------------
            Synchronization of the training dataset
        --------------------------------------------------
        """
        """
        This wasn't put in another class since it requires QObjects and
        isn't used anywhere else at this current time.

        An argument can be made to encapsulate this in a factory method here, but, I desire not to
        since that would imply that the MainWindow acts outside of its desired purpose and,
        in the future, can become a dependency on other classes, which is illogical since, well,
        it's the central widget of the app. Having another class be dependent on this would make
        the central widget as "just another class".

        IF thou truly want to put it somewhere else, you must reimplement the QObject
        for signaling.
        """
        training_queue: Queue = Queue[Callable[[None], None]]()
        emptied: bool = True
        lock = Lock()

        # I must admit that it would've been better to create a chain of command
        # Design here instead of random processes all around and windows which
        # States are dictated through signals
        async def _addQueue(exe: Callable[[None], None]) -> None:
            training_queue.put(exe)
            if not emptied: return

            lock.acquire()
            if emptied:
                emptied = False

                async def chaining(prev: Process | None = None) -> None:
                    if prev is not None: prev.join()
                    nonlocal emptied

                    if training_queue.empty():
                        emptied = True
                    else:
                        training_queue.get()()
                        Process(target=chaining, args=(current_process(),)).run()

                chaining()
            lock.release()

        del training_queue, emptied, lock

        # Styling for button
        self.find_but.setMinimumSize(minw=80, minh=35)
        self.find_but.setMaximumSize(maxw=80, maxh=35)
        self.find_but.sizePolicy().setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.find_but.setFlat(True)

        # Widgets
        self.other_but.layout().addWidget(self.find_but)

        """
        --------------------------------------------------
                Callbacks and synchronizing windows
        --------------------------------------------------
        """
        # Tip for the reader: Don't try to think to hard about this parallelism magic... It hurts.

        # Widget callbacks
        self.find_but.clicked.connect(toggler(self.find_window))
        self.data_window.being_modified.connect(self.options.setDisabled)
        self.data_window.being_modified.connect(self.find_window.setDisabled)

        # Synchronizing main dataset with secondary dataset
        self.data_window.training[0].connect(_addQueue(self.display.model.updateShowing))

        # Finding window callbacks
        self.find_window.find_signal.connect(
            lambda : Process(
                target=_addQueue,
                args=(lambda : self.display.find(**self.find_window.sendReport()),)
            ).run()
        )
        self.find_window.remove_signal.connect(
            lambda : Process(
                target=self.data_window.accessPapers,
                args=(lambda papers: papers.model.remove(self.display.found(**self.find_window.sendReport())),)
            ).run()
        )

        # Mutually exclusive parameter manipulation
        self.options.querying.connect(self.params_window.setDisabled)
        self.params_window.writing.connect(self.options.setDisabled)

        # Temporary function definitions since lambdas cannot create objects
        # In that case, the path can become a volatile variable when
        # A query is started whilst the remove/adding process isn't finished
        def _remove(event: QEvent) -> None:
            saved: Path = self.options.directory.absolute()
            Process(target=(lambda : self.data_window.remove(event, saved))).run()

        def _add(event: QEvent) -> None:
            saved: Path = self.options.directory.absolute()
            # Inner definition for the process to hook on
            def _inner() -> None:
                nonlocal event
                self.data_window.add(event, "Validation", saved + First.FILES[0])                 # Validation for the last training step
                self.data_window.add(QEvent(), "Training", saved + First.FILES[2])                # Training for the AI
                # Note that, like the comment said in the `query` function, this will try to add
                # Papers that were already added, but since the add function runs in O(n**2) and
                # This adds the most papers, the cost of putting this function earlier would be greater than leaving it here
                self.data_window.add(QEvent(), "Default case. Hello :)", saved + First.FILES[1])  # All other citations
            Process(_inner).run()

        # Querying callbacks
        self.options.remove_signal.connect(_remove)
        self.options.add_signal.connect(_add)

    # Function that dismounds the first step of the procedure
    def dismountFirst(self: Self) -> None:
        # Exceptions if the inputs were invalid
        if not self.data_window.dataset[1]: raise Exception("The main dataset was empty.")
        if not self.data_window.training[1]: raise Exception("The training dataset was empty.")

        unlabeled: bool = False
        for paper in self.data_window.training[1]:
            if not paper.labeled():
                unlabeled = True
                break

        if unlabeled:
            response: bool = False
            box: QDialog = mbFactory(
                "Papers missing labels",
                "Some papers are missing labels. Do you still wish to proceed?",
                QMessageBox.Icon.Warning,
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Abort,
                self
            )

            # Shortcut since there is an assignment
            def _boxAccepted(event: QEvent) -> None:
                nonlocal response
                event.accept()
                response = True

            box.accepted.connect(_boxAccepted)
            box.exec()

            if response: raise Exception("Couldn't dismount.")

        del unlabeled

        # Removing the callbacks
        self.data_window.being_modified.disconnect(self.options.setDisabled)
        self.data_window.being_modified.disconnect(self.find_window.setDisabled)
        self.options.querying.disconnect(self.params_window.setDisabled)
        self.params_window.writing.disconnect(self.options.setDisabled)

        # Removing the button for the find window
        self.other_but.layout().removeWidget(self.find_but)
        # Deleting the widgets associated with the finding window
        del self.find_window, self.find_but, self.options, self.display

    # Function that mounts the second step
    def mountSecond(self: Self) -> None:
        # Deactivating all possibilities of modifying the data
        self.data_window.setDisabled(True)

        # Step defined widgets
        self.display = OperatingCurve(self)
        self.options = Second(self)

        lock = Lock()
        def _plotSignal(event: QEvent) -> None:
            event.accept()
            lock.acquire()
            try: self.display.setAttributes(**self.options.sendReport())
            except: return

            self.display.plot()
            lock.release()
            self.options.setDisabled(False)

        self.options.plot_signal.connect(lambda event: (self.options.setDisabled(True), Process(target=_plotSignal, args=(event,)).run()))
        self.options.clear.connect(lambda event: (event.accept(), lock.acquire(), self.display.clear(), lock.release()))

    """
    Note that the second window will not be dismounted since the rest of the
    procedure is done in secondary windows and the user may want to have different kinds of options
    relating the NC procedure if some interval doesn't contain enough samples.
    """

    def mountThird(self: Self) -> None:
        value: int | str | None = None
        valid: bool = False
        freq: QDialog = Frequency(self)

        # Necessary for assignment, but this is only temporary
        def _freqCallback(event: QEvent) -> None:
            nonlocal value
            event.accept()
            value = freq.freq_edit.text()

        freq.accepted.connect(_freqCallback)
        del _freqCallback

        while not valid:
            freq.exec()
            try:
                value = int(value)
                valid = value >= 0
            except: pass
        del valid

        load: Loading = Loading(self.data_window.dataset[1], self, min_words=value)
        load.show()

        def _target() -> None:
            load.getStems()
            load.close()

            stem: Stem = Stem(load.grams, self)
            stem.show()

            # Could not include this in lambda, because of the exception
            def _throwing(event: QEvent) -> Never:
                event.accept()
                raise Exception("Cannot put this in lambda")

            def _accepted() -> None:
                self.grams_found = load.grams_found
                self.grams       = load.grams

            stem.rejected.connect(_throwing)
            stem.accepted.connect(_accepted)
            stem.show()

        Process(target=_target).run()

    """
    Function used to make predictions on a specific paper.
    This is the central function which trains the AI.

    @author  Thomas Gauthier, Janosch Ortmann
    @version 0.0
    """
    def predictions(
                     self:     Self,   # Assuming that self has a reference to the ngrams.
                     seed:     int,
                     size:     float,  # Assuming that the size is valid.
                     pos:      float,
                     splits:   int,
                     model:    int,    # 0 is linear regression, 1 is decision tree
                     sampling: int     # 0 is oversampling, 1 is undersampling.
                   ) -> None:
        # Todo : Add both a prompt showing progress and a final list to show the
        # Todo : Probabilities in a decreasing order
        """
        --------------------------------------------------
                        Separating datasets
        --------------------------------------------------
        """
        # For not getting them each time in the loop
        # FIXME : If you need to modify the type, remember that `List` is contravariant,
        # So this may not work anymore. Also, this may be premature optimization, but meh.
        dataset_list:    list[Paper] = self.data_window.dataset[1]
        training_list:   list[Paper] = self.data_window.training[1]
        validation_list: list[Paper] = self.data_window.validation[1]

        length: int = len(self.grams_found[0]) if self.grams_found else 0
        validation_stems: np.ndarray = np.empty((len(validation_list), length))
        training_stems:   np.ndarray = np.empty((len(training_list), length))

        for count in range(len(dataset_list)):
            element: Paper = dataset_list[count][1]

            if element in validation_list:
                validation_stems[len(validation_stems)] = (self.grams_found[count])
                continue  # Mutually exclusive with the training dataset
            if element in training_list:
                training_stems[len(training_stems)] = (self.grams_found[count])

        data_train, data_test, results_train, results_test = ms.train_test_splits(
            training_stems,
            map(
                # As it was pointed out in the instructions, it is better to have false positives than false negatives
                # Thus, the unlabeled papers will automatically be accepted.
                # Also, binary input moment
                lambda paper: not paper.labeled() or paper.label == Paper.LABELS[1],
                training_list
            ),
            test_size=math.ceil(size * len(training_list)),
            random_state=seed
        )

        """
        --------------------------------------------------
                    Oversampling/Undersamping
        --------------------------------------------------
        """
        indexes_train: np.ndarray = results_train == 1
        data_train_p:  np.ndarray = data_train[indexes_train]
        data_train_n:  np.ndarray = data_train[~indexes_train]

        del indexes_train, indexes_test

        length_p: int = len(data_train_p)
        length_n: int = len(data_train_n)
        length        = len(data_train)

        # Significance of the ratios
        EPSILON: Final[float] = 1e-2  # Note the variable here is not needed, but brings context
        if pos - round(length_p / length, 2) < -EPSILON:
            match sampling:  # Note that Strategy / Command design pattern could also be used
                case 0:  # Oversampling
                    """
                    Here is a small proof for the value taken in oversampling:
                    let n be the number of papers in the training set
                    let p be the wanted proportion of positives
                    let a be the number of actual positives

                    Define Delta = np - a
                    Delta then represents the variation between the number of
                    papers tShat you want and the actual ones you have.

                    Let x be a variable such that:
                    (n + x)p - (a + x)    = 0
                <=> np + xp  - a - x    = 0
                <=> (np - a) + x(p - 1) = 0
                <=> Delta               = x(1 - p)
                ==> x = Delta / (1 - p)

                    Thus, you find that you must add Delta / (1 - p) choices
                    back in the positive dataset.
                    """
                    data_train_p = np.concatenate(
                        (
                            data_train_p,
                            np.random.choice(
                                data_train_p,
                                size=math.ceil((length * pos - length_p) / (1 - pos))
                            )
                        ),
                        axis=0
                    )
                case 1:  # Undersampling
                    """
                    Here is a small proof for the value of undersampling:
                    let n be the number of papers in the training set
                    let p be the wanted proportion of positives
                    let a be the number of actual positives

                    Define Delta = np - a
                    Delta then represents the variation between the number of
                    papers that you want and the actual ones you have.

                    Let x be a variable such that:
                    (n - x)p - a  = 0
                <=> np - px  - a  = 0
                <=> (np - a) - px = 0
                <=> Delta         = px
                ==> x = Delta / p = n - a / p

                    Thus, you find that you must remove Delta / p papers so that the ratio
                    is valid. Note that p must be different from 0 and 1, otherwise the AI won't
                    work properly, so the division is also valid.
                    """
                    delta_train_n = np.random.choice(
                        delta_train_n,
                        size=max((length_n + math.ceil(length_p / pos)  - length), 0)
                    )
                case _: raise Exception("Not implemented")
            data_train = np.concatenate((data_train_p, data_train_n))

        model: lm.LogisticRegression | DecisionTreeClassifier = (
            lm.LinearRegression(penalty="l2", random_state=0, max_iter=1000)
            if model == 0 else
            DecisionTreeClassifier(random_state=0)
        ).fit(data_train, results_train)

        """
        --------------------------------------------------
                        Evaluating the AI
        --------------------------------------------------
        """

        results_train_pred: np.ndarray = model.predict(data_train)
        results_test_pred:  np.ndarray = model.predict(data_test)

        def _shortcutTrain(func: Callable[..., float]) -> float:
            nonlocal results_train, results_train_pred
            return round(func(results_train, results_train_pred) * 100, 2)

        def _shortcutTest(func: Callable[..., float]) -> float:
            nonlocal results_test, results_test_pred
            return round(func(results_test, results_test_pred))

        confused_train: np.ndarray = confusion_matrix(results_train, results_train_pred)
        confused_test:  np.ndarray = confusion_matrix(results_test, results_test_pred)

        trained_win: QDialog = Statistics(
            an=confused_train[0,0],
            fn=confused_train[1, 0],
            ap=confused_train[1, 1],
            fp=confused_train[0, 1],
            acc=_shortcutTrain(accuracy_score),
            rec=_shortcutTrain(recall_score),
            pre=_shortcutTrain(precision_score),
            f1=_shortcutTrain(f1_score)
        )

        trained_win.setWindowTitle("Training Window")

        test_win: QDialog = Statistics(
            an=confused_test[0,0],
            fn=confused_test[1, 0],
            ap=confused_test[1, 1],
            fp=confused_test[0, 1],
            acc=_shortcutTest(accuracy_score),
            rec=_shortcutTest(recall_score),
            pre=_shortcutTest(precision_score),
            f1=_shortcutTest(f1_score)
        )

        test_win.setWindowTitle("Testing Window")

        crossval_win: Statistics | None = None
        if splits > len(data_train):
            errorFactory(
                "Couldn't split",
                "The cross-validation could not be performed due to a lack of training data"
            ).show()
            crossval_win = Statistics()
        else:
            def _shortcutCross(string: str) -> float:
                return round(np.mean(ms.cross_val_score(model, data_train, results_train, scoring=string, cv=splits)) * 100, 2)

            crossval_win = Statistics(
                f1=_shortcutCross("f1"),
                acc=_shortcutCross(""),
                pre=_shortcutCross("precision"),
                rec=_shortcutCross("recall")
            )

        crossval_win.setWindowTitle("Cross Validation Window")

        barrier = Barrier(1, timeout=0)
        def _closing(accepted: bool) -> None:
            barrier.wait()
            crossval_win.close()
            trained_win.close()
            test_win.close()

            if not accepted: raise Exception("Didn't accept")

            params: dict[str, str] = appendParams()

            pred: np.ndarray = model.predict_proba(self.grams_found)
            for count in range(len(pred)): dataset_list[count][1].assign(pred[count])
            del pred

            # Shortcut
            moved: Callable[..., Any]  = sorted
            sorted = lambda dataset : moved(dataset, key=lambda paper : paper.prob)

            sorted(dataset_list)
            sorted(validation_list)
            sorted(training_list)

            sorted = moved
            del moved

            base: float = params["thr"]
            grow: float = params["step"]

            final_validation: Final[float] | None = None
            final_training:   Final[float] | None = None

            # Shortcut used to exit the loop with the defined threshold
            # Needed because of the weird Python scopes and lambdas
            def _outsideTraining(threshold: float) -> Never:
                nonlocal final_training
                final_training = threshold
                raise Exception("Exited")

            def _outsideValidation(threshold: float) -> Never:
                nonlocal final_validation
                final_validation = threshold
                raise Exception("Exited")

            def _show(li: list[Paper], prob: list[float], func: Callable[[float], Never]) -> None:
                second: float = base + grow
                first:  float = base

                index_first:  int = 0
                index_second: int = 0

                le: int = len(li)

                try:
                    # index_second is used, since this while loop is actually a do while (which python doesn't have)
                    while index_second < le:
                        index_first  = cutoff(prob, first) + 1
                        index_second = cutoff(prob, second) + 1

                        window: QDialog = Interval(li[index_first:index_second], first, second)

                        window.accepted.connect(lambda event : (event.accept(), func(first)))
                        window.exec()

                        second += grow
                        first  += grow
                except: pass

            _show(training_list, map(lambda paper : paper.prob, training_list), _outsideTraining)
            _show(validation_list, map(lambda paper : paper.prob, validation_list), _outsideValidation)

            if final_validation != final_training:
                mbFactory(
                    "Different cutoff",
                    "The cutoff for the validation is different from the training",
                    QMessageBox.Icon.Warning,
                    QMessageBox.StandardButton.Ok,
                    self
                ).show()

            final: Final[int] = min(final_validation, final_training)
            del _outsideTraining, _outsideValidation

            cutoff_index: int = cutoff(map(lambda paper : paper.prob, dataset_list), final) + 1
            with open(MainWindow.DEFAULT_FILE, "w") as file:
                moving: Callable[..., Any] = dataset_list.__next__
                def _write() -> str:
                    paper = moving()[1]
                    return paper.title + " " + str(paper.date) + " " + paper.jour + " " + str(paper.prob)
                dataset_list.__next__ = _write
                file.writelines(dataset_list[cutoff_index:])

            mbFactory(
                "Values printed",
                "The final values were printed at " + MainWindow.DEFAULT_FILE,
                QMessageBox.StandardButton.Ok,
                QMessageBox.Icon.Information,
                self
            ).exec()
            # Programs ends here

        crossval_win.connect(_closing)
        trained_win.connect(_closing)
        test_win.connect(_closing)

        crossval_win.show()
        trained_win.show()
        test_win.show()

main: QMainWindow = MainWindow()
app.exec()
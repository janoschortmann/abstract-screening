#!/usr/bin/env python3
"""
Main file setting up the UI and the logic.

It only contains the MainWindow.

@author  Thomas Gauthier
@version 0.3
"""
import sys
import nltk
nltk.download('stopwords')

# Done to access all newer features of the typing library such as generics
if sys.version_info < (3, 12):
    sys.stderr.write("Python version inferior to 3.12+. Please make sure to update to a newer version.")
    sys.exit()

from PySide6.QtCore    import Qt
from PySide6.QtWidgets import (
                                QApplication,
                                QDialog,
                                QMainWindow,
                                QMessageBox,
                                QPushButton,
                                QSizePolicy
                              )
from PySide6.QtGui     import QCloseEvent

# Supposed to run first as the name of the file says
if __name__ != "__main__":
    sys.exit(-1)

# I <3 "QPixmap: Must construct a QGuiApplication before a QPixmap"
import qdarkstyle
app: QApplication = QApplication([])
app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="PySide6"))

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree    import DecisionTreeClassifier
from threading       import Condition, Lock, Thread
from typing          import Final, Self, final
from webbrowser      import open_new_tab

from python.src.utils.classes   import TreatedException
from python.src.utils.files     import Paper, CENTRAL
from python.src.utils.functions import cutoff, mkabsent
from ui.display.entities        import FindingView, OperatingCurve, SelectionView, waitFactory
from ui.compiled                import mainwindow
from ui.windows                 import *

import numpy                   as np
import sklearn.model_selection as ms
import sklearn.linear_model    as lm

"""
The main window of the UI.

The main job of the MainWindow class is to synchronize all datasets
from the Data class to everything else.

It sets up the callacks and backprocesses.

The MainWindow class does not manage directly the data (it synchronizes it) since that
is done with the Data class (which also manages the data between the view,
the backend and sets up parallelism between multiple get and set operations).

You can then say, in gruesome terms, this does the "logic" of the app whilst the Data class
does the "synchronizing" of the app.

@author  Thomas Gauthier
@version 0.4
"""
@final
class MainWindow(QMainWindow, mainwindow.Ui_mainwindow):
    # Github link for the "about" window
    GITHUB_LINK:   Final[str] = "https://github.com/janoschortmann/abstract-screening"
    # The default directory for the results of the probabilities
    DEFAULT_DIR:   Final[str] = osp.join(CENTRAL, "results")
    # The deault style of some QWidget
    DEFAULT_STYLE: Final[str] = "border: 1px solid grey;"

    # The initializer of the window.
    def __init__(self: Self) -> None:
        # Setting up the UI
        super().__init__()
        self.setupUi(self)

        # Other windows
        self.params_window: Parameters = Parameters()
        self.data_window:         Data = Data()

        # Multithreading manipulation
        """
        So, in theory, this is not the best solution there is since there is a few
        milliseconds window where the next button can be clicked and start a new process, leading
        to undefined behaviour. Fortunately, since this is a local app, it doesn't really matter in
        practice, since the client would need to go out of their way to break the program.
        """
        self.data_window.being_modified.connect(self.next.setDisabled)

        # Setting up the callbacks
        self.params.clicked.connect(toggler(self.params_window))
        self.data.clicked.connect(toggler(self.data_window))
        self.about.clicked.connect(lambda : open_new_tab(MainWindow.GITHUB_LINK))

        """
        A tuple representing the sequence of events this class will call.
        This is done as a way to encapsulate methods in different steps.

        Also, the exceptions must be thrown before the method changes the UI,
        otherwise, the modifications will stay, which can lead to multiple issues
        such as being locked out of the app.

        In theory, there should be a "undo" of some step so that the exceptions can be
        thrown everywhere, but that seems unnecessary, since the jobs of the procedures
        here are only to setup and dismount components of the UI, not to do logic
        (except the last one, obviously).
        """
        sequence: tuple[Callable[..., Any]] = (
            lambda : (self.dismountFirst(), self.mountSecond()),
            self.mountThird,
            self.mountForth,
        )

        ite: Iterable[Any] = iter(sequence)
        cur: Callable[[None], None] | None = None
        err: bool = False
        def _executeNext() -> None:
            nonlocal ite, cur, err
            cur = next(ite) if not err else cur
            try:
                cur()
                err = False
            except Exception as ex:
                errorFactory(
                    "Error in next",
                    f"Couldn't proceed because of : {ex}.",
                    self
                ).show()
                err = True

        self.next.clicked.connect(_executeNext)
        self.mountFirst()
        self.show()

    @override
    def closeEvent(self: Self, event: Any) -> None:
        sys.exit(0)

    # Function that sets up the first step of the procedure.
    def mountFirst(self: Self) -> None:
        # Initializing the variables for the finding window
        self.find_but: QPushButton = QPushButton()
        self.find_window:     Find = Find("Finder for Central Widget")

        # Step defined widgets
        self.display = FindingView(SelectionView(self.data_window.training[1]))
        self.options = First()

        self.body.insertWidget(0, self.display)
        self.bottom.insertWidget(0, self.options)

        self.display.setParent(self.body)
        self.options.setParent(self.bottom)

        """
        --------------------------------------------------
            Synchronization of the training dataset
        --------------------------------------------------
        """
        emptied: bool = True
        lock:    Lock = Lock()
        # List, in python, are synchronized and the Queue, from multiprocessing, was making me angry
        queue:   list = []

        # I must admit that it would've been better to create a chain of command
        # Design here instead of random processes all around and windows which
        # States are dictated through signals
        def _addQueue(exe: Callable[[None], None]) -> None:
            nonlocal queue, emptied, lock
            queue.append(exe)
            lock.acquire(timeout=0)
            if emptied:
                emptied = False

                def chaining(call: Callable[[None], None] | None = None) -> None:
                    if call is not None: call()  # Should not create UI elements
                    nonlocal emptied, queue

                    if len(queue) == 0: emptied = True
                    else: chaining(queue.pop(0))
                Thread(target=chaining).start()
            lock.release()

        # Styling for button
        self.find_but.setMinimumSize(80, 35)
        self.find_but.setMaximumSize(80, 35)
        self.find_but.sizePolicy().setVerticalPolicy(QSizePolicy.Policy.Fixed)

        self.find_but.setFlat(True)
        self.find_but.setText("Find")
        self.find_but.setStyleSheet(MainWindow.DEFAULT_STYLE)

        # Styling for the display
        self.display.setStyleSheet(MainWindow.DEFAULT_STYLE)

        # Widgets
        self.other_but.layout().addWidget(self.find_but)
        self.find_but.setParent(self.other_but)

        """
        --------------------------------------------------
                Callbacks and synchronizing windows
        --------------------------------------------------
        """
        # Tip for the reader: Don't try to think to hard about this parallelism magic... It hurts

        # Widget callbacks
        self.find_but.clicked.connect(toggler(self.find_window))
        self.data_window.being_modified.connect(self.options.setDisabled)
        self.data_window.being_modified.connect(self.find_window.setDisabled)

        """
        This needs a bit of explanation.

        So, because in python lists are synchronized (you can check this manually with an
        `iter` calling `next` after some `pop`), that implies that the model.__hidden will
        always raise `StopIteration` and not some other exception (which will be caught by the `for`).

        Now, it is possible that model.__hidden is changing at the same time that `updateShowing`
        is running because data_window called `add` or `remove`, which changes the training dataset
        because of the way JoinedList's `append` and `remove` work.

        That means that some paper that is currently being removed in `data_window.remove` will be shown
        because of `updateShowing`. Luckily, we know that if `data_window.remove` is called, after the
        removal process is done, training[0] will emit. That implies that the papers shown will also update.

        Therefore, this will always, theoretically, be valid, because of the way the GIL and lists work in python.
        """
        self.data_window.training[0].connect(lambda : _addQueue(self.display.model.updateShowing))

        # Finding window callbacks
        self.find_window.find_signal.connect(
            lambda : _addQueue(lambda : self.display.find(**self.find_window.sendReport()))
        )
        self.find_window.remove_signal.connect(
            lambda : Thread(  # Thread necessary to prevent lock of ui
                target=self.data_window.accessPapers,
                args=(lambda papers: papers.model.remove(self.display.found(**self.find_window.sendReport())),)
            ).start()
        )

        # Mutually exclusive parameter manipulation
        self.options.querying.connect(self.params_window.setDisabled)
        self.params_window.writing.connect(self.options.setDisabled)

        # Temporary function using the directory as a variable
        def _add() -> None:
            saved: Path = self.options.directory.absolute()
            self.data_window.add("Validation", osp.join(saved, First.FILES[0]))   # Validation for the last training step
            self.data_window.add("Training", osp.join(saved, First.FILES[2]))  # Training for the AI
            # Note that, like the comment said in the `query` function, this will try to add
            # Papers that were already added, but since the add function runs in O(n**2) and
            # This adds the most papers, the cost of putting this function earlier would be greater than leaving it on this line
            self.data_window.add("", osp.join(saved, First.FILES[1]))  # All other citations

        # Callbacks for adding and removing the queries
        # Note that since the connection type is Direct, that implies that another thread will call this,
        # Making it useless to create another Thread
        self.options.remove_signal.connect(
            lambda : self.data_window.remove(self.options.directory.absolute()),
            type=Qt.ConnectionType.DirectConnection
        )
        # Note that since the connection type is Direct, that implies that another thread will call this,
        # Making it useless to create another Thread
        self.options.add_signal.connect(_add, type=Qt.ConnectionType.DirectConnection)

    # Function that dismounds the first step of the procedure
    def dismountFirst(self: Self) -> None:
        # Exceptions if the inputs were invalid
        if not self.data_window.dataset[1]:  raise Exception("The main dataset was empty.")
        if not self.data_window.training[1]: raise Exception("The training dataset was empty.")

        has_both: int = 0x00
        for paper in self.data_window.training[1]:
            if paper.labeled() and paper.label == Paper.LABELS[2]: has_both = has_both | 0b01
            elif not paper.labeled() or (paper.label == Paper.LABELS[1]): has_both = has_both | 0b10

            if has_both == 0b11: break
        else: raise Exception("All papers are either included or excluded")

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
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Abort,
                self
            )

            # Shortcut since there is an assignment
            def _boxAccepted() -> None:
                nonlocal response
                response = True

            box.rejected.connect(_boxAccepted)
            box.exec()

            # This could be changed
            if response: raise Exception("Couldn't dismount")

        del unlabeled

        # Removing the callbacks
        self.data_window.being_modified.disconnect(self.data_window.find_window.setDisabled)
        self.data_window.being_modified.disconnect(self.find_window.setDisabled)
        self.data_window.being_modified.disconnect(self.data_window.setDisabled)
        self.data_window.being_modified.disconnect(self.options.setDisabled)
        self.options.querying.disconnect(self.params_window.setDisabled)
        self.params_window.writing.disconnect(self.options.setDisabled)

        # Removing the button for the find window
        # Credit goes to @Neuron for https://stackoverflow.com/questions/5899826/pyqt-how-to-remove-a-widget
        self.find_but.setParent(None)
        self.display.setParent(None)
        self.options.setParent(None)

        # Deleting the widgets associated with the finding window
        self.find_window.deleteLater()
        self.find_but.deleteLater()
        self.options.deleteLater()
        self.display.deleteLater()

    # Function that mounts the second step
    def mountSecond(self: Self) -> None:
        global GLO_DEL
        # Deactivating all possibilities of modifying the data
        self.data_window.setDisabled(True)
        # Step defined widgets
        self.display = OperatingCurve()
        self.options = Second()

        self.body.insertWidget(0, self.display)
        self.bottom.insertWidget(0, self.options)

        self.display.setParent(self.body)
        self.options.setParent(self.bottom)

        # Styling for the display
        self.display.setStyleSheet(MainWindow.DEFAULT_STYLE)

        def _plotSignal() -> None:
            try:
                self.display.setAttributes(**self.options.sendReport())
                self.display.plotf()
            except Exception as ex:
                GLO_DEL.call(
                    lambda _self, ex: errorFactory(
                        "Could not plot",
                        f"Error in the plotting. Error: {ex}.",
                        _self
                    ).show(),
                    _self=self,
                    ex=ex
                )

        # Disabling callbacks
        self.display.modifying.connect(self.next.setDisabled)
        self.display.modifying.connect(self.options.setDisabled)

        # Plotting callbacks
        self.options.plot_signal.connect(lambda : Thread(target=_plotSignal).start())
        self.options.clear.pressed.connect(self.display.clear)

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
        def _closing() -> None:
            nonlocal value, freq
            value = freq.freq_edit.text().strip()
            freq.close()

        freq.closeEvent = lambda ignored: _closing()
        freq.ok.pressed.connect(lambda : freq.closeEvent(None))

        while not valid:
            freq.exec()
            try:
                value = int(value) if value else 0
                valid = value >= 0
            except: pass
        del valid

        load: Loading = Loading(self.data_window.dataset[1], parent=self, min_words=value)
        load.show()

        def _target() -> None:
            load.getStems()
            GLO_DEL.call(load.close)

            def _do() -> None:
                nonlocal load, self
                global app
                stem: Stem = Stem(load.grams, self)

                def _throwing(_) -> None:
                    errorFactory(
                        "Stems rejected",
                        "The stems were rejected. If more stems are wanted, please add more papers to the training dataset.",
                        self
                    ).exec()
                    sys.exit(1)

                move = stem.closeEvent
                def _accepted() -> None:
                    nonlocal stem, move
                    self.grams_found = load.grams_found
                    self.grams       = load.grams
                    self.next.setDisabled(False)
                    move(QCloseEvent())

                stem.closeEvent = _throwing
                stem.cancel.pressed.connect(stem.close)
                stem.ok.pressed.connect(_accepted)

                stem.show()

            GLO_DEL.call(_do)

        self.next.setDisabled(True)  # The data_window is already disabled
        Thread(target=_target).start()

    # Forth and final step of the mounting
    def mountForth(self: Self) -> None:
        self.options.setParent(None)
        self.options.deleteLater()

        self.options = Third()
        self.bottom.insertWidget(0, self.options)
        self.options.setParent(self.bottom)

        def _predict() -> None:
            nonlocal self
            # This should be left with no parent, otherwise, the windows created
            # By the global delegator will be left under, which is sub-optimal,
            # Especially when showing the statistics of the model
            mb: QMessageBox = waitFactory(
                "Waiting window",
                "Waiting for predictions",
                (1000/3)  # A third of a second
            )
            self.next.setDisabled(True)
            def _threadcall() -> None:
                nonlocal self
                global GLO_DEL, app
                try:
                    # Shortcut since self.options.rendReport also raises exceptions
                    ret_val: tuple[bool, str] = self.predictions(**self.options.sendReport())
                    if not ret_val[0]: raise Exception(f"Cancelled the predictions because : {ret_val[1]}")
                    GLO_DEL.call(lambda mb, app: (mb.close(), sys.exit(0)), mb=mb)
                except TreatedException as _:
                    GLO_DEL.call(
                        lambda _self, mb: (mb.close(), _self.next.setDisabled(False)),
                        _self=self,
                        mb=mb,
                    )
                except Exception as ex:
                    GLO_DEL.call(
                        lambda _self, mb, ex: (
                            mb.close(),
                            _self.next.setDisabled(False),
                            errorFactory(
                                "Prediction Error",
                                f"Error in the predictions with : {ex}.",
                                _self
                            ).show()
                        ),
                        _self=self,
                        mb=mb,
                        ex=ex
                    )

            Thread(target=_threadcall).start()

        self.next.pressed.connect(_predict)

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
                   ) -> bool:
        global GLO_DEL

        """
        --------------------------------------------------
                        Separating datasets
        --------------------------------------------------
        """
        # For not getting them each time in the loop
        # FIXME : If you need to modify the type, remember that `List` is invariant,
        # So this may not work anymore. Also, this may be premature optimization, but meh.
        dataset_list:    list[Paper] = self.data_window.dataset[1]
        training_list:   list[Paper] = self.data_window.training[1]
        validation_list: list[Paper] = self.data_window.validation[1]

        length: int = len(self.grams_found[0]) if self.grams_found.any() else 0
        validation_stems: np.ndarray = np.empty((len(validation_list), length))
        training_stems:   np.ndarray = np.empty((len(training_list), length))

        count_val:   int = 0
        count_train: int = 0
        for count in range(len(dataset_list)):
            element: Paper = dataset_list[count]

            if element in validation_list:
                validation_stems[count_val] = self.grams_found[count]
                count_val += 1
                continue  # Mutually exclusive with the training dataset
            if element in training_list:
                training_stems[count_train] = self.grams_found[count]
                count_train += 1

        data_train, data_test, results_train, results_test = ms.train_test_split(
            training_stems,
            list(
                map(
                    lambda paper: int(not paper.labeled() or paper.label == Paper.LABELS[1]),
                    training_list
                )
            ),
            test_size=size,
            random_state=seed
        )

        # List and not numpy objects
        results_train = np.asarray(results_train)
        results_test  = np.asarray(results_test)

        """
        --------------------------------------------------
                    Oversampling/Undersamping
        --------------------------------------------------
        """
        indexes: np.ndarray = results_train == 1
        data_train_p:  np.ndarray = data_train[indexes]
        data_train_n:  np.ndarray = data_train[~indexes]

        length_p: int = len(data_train_p)
        length_n: int = len(data_train_n)
        length:   int = len(data_train)

        # Significance of the ratios
        EPSILON: Final[float] = 1e-2  # Note the variable here is not needed, but brings context
        if pos - round(length_p / length, 2) >= EPSILON:
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

                    Note that Delta > 0 because of the if statement.

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
                            data_train_p[
                                np.random.choice(
                                    len(data_train_p),
                                    size=math.ceil((length * pos - length_p) / (1 - pos))
                                )
                            ]
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

                    Note that Delta > 0 because of the if statement.

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
                    data_train_n = data_train_n[
                        np.random.choice(
                            len(data_train_n),
                            size=max((length_n + math.floor(length_p / pos)  - length), 0)
                        )
                    ]
                case _: raise Exception("Not implemented")

            data_train    = np.concatenate((data_train_p, data_train_n), axis=0)
            results_train = np.concatenate(
                (
                    np.full(data_train_p.shape[0], 1),
                    np.full(data_train_n.shape[0], 0)
                ),
                axis=0
            )

        model: lm.LogisticRegression | DecisionTreeClassifier = (
            lm.LogisticRegression(penalty="l2", random_state=0, max_iter=1000)
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

        confused_train: np.ndarray = confusion_matrix(
            y_true=results_train,
            y_pred=results_train_pred,
            labels=[0, 1]
        )
        confused_test:  np.ndarray = confusion_matrix(
            y_true=results_test,
            y_pred=results_test_pred,
            labels=[0, 1]
        )

        waiter:  Condition = Condition()
        waiting:      bool = False
        ret_val: list = [False, ""]

        # Callback since this is UI manipulation
        def _uiCall() -> None:
            nonlocal self, confused_train, confused_test, data_test, data_train, model, splits,\
            _shortcutTest, _shortcutTrain, results_test, results_train, ret_val, waiter, waiting
            self.trained_win = Statistics(
                an=confused_train[0,0],
                fn=confused_train[1, 0],
                ap=confused_train[1, 1],
                fp=confused_train[0, 1],
                acc=_shortcutTrain(accuracy_score),
                rec=_shortcutTrain(recall_score),
                pre=_shortcutTrain(precision_score),
                f1=_shortcutTrain(f1_score)
            )
            self.trained_win.setWindowTitle("Training Window")

            self.test_win = Statistics(
                an=confused_test[0,0],
                fn=confused_test[1, 0],
                ap=confused_test[1, 1],
                fp=confused_test[0, 1],
                acc=_shortcutTest(accuracy_score),
                rec=_shortcutTest(recall_score),
                pre=_shortcutTest(precision_score),
                f1=_shortcutTest(f1_score)
            )
            self.test_win.setWindowTitle("Testing Window")

            self.crossval_win = None
            if splits > len(data_train):
                errorFactory(
                    "Couldn't split",
                    "The cross-validation could not be performed due to a lack of training data."
                ).show()
                self.crossval_win = Statistics()
            else:
                def _shortcutCross(string: str) -> float:
                    return round(
                             np.mean(
                               ms.cross_val_score(
                                  model,
                                  data_train,
                                  results_train,
                                  scoring=string,
                                  cv=splits,
                                  error_score='raise'
                              )
                          ) * 100,
                          2
                      )
                
                try:
                    self.crossval_win = Statistics(
                        f1=_shortcutCross("f1"),
                        acc=_shortcutCross("accuracy"),
                        pre=_shortcutCross("precision"),
                        rec=_shortcutCross("recall")
                    )
                except Exception as ex:
                    with waiter:
                        ret_val[1] = f"Error in cross validation {ex}"
                        if waiting: waiter.notify()
                        else: waiting = True
                        return

            self.crossval_win.setWindowTitle("Cross Validation Window")

            blocker = Lock()
            def _closing() -> None:
                nonlocal blocker, model, ret_val, self, waiting, waiter
                global GLO_DEL

                blocker.acquire(timeout=0)
                GLO_DEL.call(
                    lambda _self: (
                        _self.crossval_win.close(),
                        _self.trained_win.close(),
                        _self.test_win.close()
                    ),
                    _self=self
                )

                try: params: dict[str, str] = appendParams()
                except:
                    with waiter:
                        ret_val[1] = "Parameters are not defined"
                        if waiting: waiter.notify()
                        else: waiting = True
                        return

                pred: np.ndarray = model.predict_proba(self.grams_found)
                for count in range(len(pred)): self.data_window.dataset[1][count].assign(float(pred[count, 1]))
                del pred

                # I don't know why the lambda function didn't work (since it was recognized as a numpy
                # Object for some reason, so I deleted it)
                # The lambda in quesion: moved = lambda dataset: dataset.sort(key=lambda paper: paper.prob)
                self.data_window.validation[1].sort(key=lambda paper: paper.prob)
                self.data_window.training[1].sort(key=lambda paper: paper.prob)
                self.data_window.dataset[1].sort(key=lambda paper: paper.prob)

                base: float = float(params["thr"])
                grow: float = float(params["step"])

                final_validation: float = 0
                final_training:   float = 0

                # Shortcut used to exit the loop with the defined threshold
                # Needed because of the weird Python scopes and lambdas
                def _outsideTraining(threshold: float) -> Never:
                    nonlocal final_training
                    final_training = threshold

                def _outsideValidation(threshold: float) -> Never:
                    nonlocal final_validation
                    final_validation = threshold

                def _show(li: list[Paper], func: Callable[[float], Never]) -> None:
                    global GLO_DEL
                    second: float = base + grow
                    first:  float = base

                    index_first:  int = 0
                    index_second: int = 0

                    proceed: bool = True
                    le:       int = len(li)

                    # Edge case to skip the break
                    if le == 0: return

                    # index_second is used, since this while loop is actually a do while (which python doesn't have)
                    while index_second < le:
                        index_first  = index_second
                        index_second = cutoff(li, second, key=lambda paper: paper.prob) + 1

                        def _handler() -> None:
                            nonlocal proceed, func, first
                            proceed = False
                            func(first)

                        def _showInterval() -> None:
                            nonlocal first, second, index_first, index_second, li, first, _handler
                            window: QDialog = Interval(li[index_first:index_second], first, second)
                            window.accepted.connect(_handler)
                            window.exec()

                        GLO_DEL.wait(_showInterval)
                        if not proceed: break

                        second += grow
                        first  += grow
                    else:
                        with waiter:
                            ret_val[1] = "Didn't select a cutoff, no papers met the criteria"
                            if waiting: waiter.notify()
                            else: waiting = True
                            return

                _show(self.data_window.training[1],   _outsideTraining)
                _show(self.data_window.validation[1], _outsideValidation)

                if final_validation != final_training:
                    GLO_DEL.call(
                        lambda _self: mbFactory(
                            "Different cutoff",
                            "The cutoff for the validation is different from the training; choosing the minimal one.",
                            QMessageBox.Icon.Warning,
                            QMessageBox.StandardButton.Ok,
                            _self
                        ).show(),
                        _self=self
                    )

                final: Final[int] = min(final_validation, final_training)
                del _outsideTraining, _outsideValidation

                # Cannot use the pred np.ndarray, since it is unsorted
                cutoff_index: int = cutoff(self.data_window.dataset[1], final, key=lambda paper: paper.prob)

                mkabsent(MainWindow.DEFAULT_DIR)
                file: str = osp.join(MainWindow.DEFAULT_DIR, f"papers-{datetime.now().strftime("%Y-%m-%d-%Hh%Mm")}.txt")
                with open(file, mode="w", encoding="utf8") as writable:
                    if cutoff_index < len(self.data_window.dataset[1]):  # By one error
                        writable.writelines(
                            map(
                                lambda paper: f"{paper.title} | {paper.date} | {paper.jour} | {paper.prob}\n",
                                self.data_window.dataset[1][(cutoff_index + 1):]
                            )
                        )

                GLO_DEL.wait(
                    lambda file, _self: mbFactory(
                        "Values printed",
                        f"The final values were printed at {file}.",
                        QMessageBox.Icon.Information,
                        QMessageBox.StandardButton.Ok,
                        _self
                    ).exec(),
                    file=file,
                    _self=self
                )

                # Programs ends here after going back to `mountForth`
                blocker.release()
                ret_val[0] = True
                with waiter:
                    if waiting: waiter.notify()
                    else: waiting = True

            num_close: int = 0
            def _call(state: int) -> None:
                nonlocal _closing, waiter, num_close, self
                if state == 0 or (num_close == 2 and state == -1):
                    with waiter:
                        ret_val[1] = "Rejected"
                        if waiting:  waiter.notify()
                        else: waiting = True

                    # The windows that are already closed don't matter
                    self.crossval_win.close()
                    self.trained_win.close()
                    self.test_win.close()
                elif state == 1:
                    num_close = -1
                    Thread(target=_closing).start()
                else: num_close += 1

            self.crossval_win.result.connect(_call)
            self.trained_win.result.connect(_call)
            self.test_win.result.connect(_call)

            self.crossval_win.show()
            self.trained_win.show()
            self.test_win.show()

            # Closing and releasing locks
            global app
            def _close(event) -> None:
                nonlocal waiter, waiting
                global app
                with waiter:
                    if waiting: waiter.notify()
                    else: waiting = True
                    sys.exit(0)
            self.closeEvent = _close

        GLO_DEL.call(_uiCall)
        move = self.closeEvent

        # In theory, there may be a race condition with the ui calling the _uiCall function
        # Before this is acquired and the user accepting or refusing everything. But,
        # Because this is so unlikely, it is left as such. There is no easy fix, since that
        # Would imply the UI waiting for this thread to acquire the lock only to notify it immediately,
        # Which would imply a rework of the Delegator class in its entirety, which seems unnecessary
        with waiter:
            if waiting: return ret_val  # If an error is thrown before waiting
            waiting = True
            waiter.wait()

        return ret_val

main: QMainWindow = MainWindow()
sys.exit(app.exec())

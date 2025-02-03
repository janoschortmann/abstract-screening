#/usr/bin/env python3
"""
File containing all sorts of useful functions,
But doesn't hold any classes or anything relating to a specific field.

@author  Thomas Gauthier
@version 0.1
"""
from ctypes    import _Pointer, pointer # Bad practice, but needed
from threading import Lock
from typing    import Any, Callable, Self, Iterable

"""
Will modify the list given as argument to have unique elements inside,
without any duplicates depending on their implementation of __eq__.

It does this with a sliding window over a sorted list.
That implies that, in addition, the elements can be compared with __le__
or __ge__.

Note that if you do not want the list to be modified
but only have a unique clone, you must pass, as an argument, a copy of the
list and not the original instance.

@author  Thomas Gauthier
@version 0.0
"""
def unique[T](collection: list[T]) -> None:
    collection.sort()
    index: int = 0
    current: T | None = None

    for element in collection:
        if element == current:
            collection.pop(index)
        else:
            index += 1
            current = collection[index]

"""
A function that will clear each instance that is considered as
empty inside the iterable.

This is done with `del collection[element]` and the element T
is designated to be empty if it returns false when casted to a boolean.

Note that the collection must be iterable and that it changes the collection
and does not copy it. That implies that if you want another collection, you must first
copy it before calling this method.

@author  Thomas Gauthier
@version 0.1
"""
def clearEmpty[T](collection: Iterable[T]) -> None:
    for element in collection:
        if isinstance(element, None) or not bool(element): del collection[element]

"""
A useful function that will make a dictionary based on a list of
unique values as keys and map them (with the mapper) to their values.

@author  Thomas Gauthier
@version 0.0
"""
def mkdict[T, Q](keys: Iterable[T], mapper: Callable[[T], Q]) -> dict[T, Q]:
    dic: dict[T, Q] = {}

    for item in keys:
        if dic[item] is not None: raise Exception("Duplicate values in iterable")
        else: dic[item] = mapper(item)

    return dic

"""
Combines an iterable of unique keys to their respective values
in the iterable as a dictionary.

@author  Thomas Gauthier
@version 0.0
"""
def cbdict[T, Q](keys: Iterable[T], values: Iterable[Q]) -> dict[T, Q]:
    if len(keys) != len(values):
        raise Exception("The iterables are different in lenght")

    values_iter: Any = iter(values)
    dic:  dict[T, Q] = {}
    for item in keys:
        if dic[item] is not None: raise Exception("Duplicate values in iterable")
        else: dic[item] = next(values_iter)

    return dic

"""
Basic factory for toggling windows.

Overrides the close event with a hidden event to not create new windows each time.

@author  Thomas Gauthier
@version 1.0
"""
from PySide6.QtWidgets import QWidget
from PySide6.QtCore    import QEvent
def toggler(window: QWidget) -> Callable[..., None]:
    window.closeEvent = lambda : window.hide()
    def inner(event: QEvent) -> None:
        nonlocal window
        if not window.isHidden(): window.hide()
        else: window.show()
        event.accept()
        activated = not activated
    return inner

import json
from ....ui.windows import Parameters, errorFactory
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

"""
A simple binary search implementation.

@author  Thomas Gauthier
@version 0.0
"""
import math
def binarySearch[T](ite: Iterable[T], target: T) -> int:
    if len(ite) == 0: return -1

    high:  int = len(ite) - 1
    if ite[high] == target: return high  # Saves me from headaches of by one errors

    low:   int = 0
    mid:   int = math.floor((high - low) / 2)
    mid_v:  T = ite[mid]

    while mid_v != target:
        if low == high + 1: return -1
        elif mid_v < target: low = mid
        else: high = mid

        mid   = math.floor((high - low) / 2)
        mid_v = ite[mid]

    return mid

"""
A function that mimics a binary search, but for finding the lowest index where
ite[index] < ite[other]. Note that if, foreach index in the array, the cutoff
is less than or equal to ite[index], this will return the index 0.

@author  Thomas Gauthier
@version 0.0
"""
def cutoff[T](ite: Iterable[T], cutoff: T) -> int:
    if len(ite) == 0: return 0

    high: int = len(ite) - 1
    if ite[high] < cutoff: return high   # Saves me from headaches of by one errors

    low:  int = 0
    mid:  int = math.floor((high - low) / 2)
    mid_v:  T = ite[mid]

    while True:
        if low == high + 1: return low
        elif mid_v < cutoff: low = mid
        else: high = mid

        mid   = math.floor((high - low) / 2)
        mid_v = ite[mid]

"""
A class representing an atomic pointer.

@author  Thomas Gauthier
@version 0.0
"""
class _AtomicInstance[T](object):
    @property
    def val(self: Self) -> T:
        return self._val

    @property.getter
    def val(self: Self) -> T:
        self.lock.acquire()
        ref: T = self._val.contents
        self.lock.release()
        return ref

    @property.setter
    def val(self: Self, other: T) -> None:
        temp: T = other # Forces it to be a lvalue
        self.lock.acquire()
        self._val = pointer(temp)
        self.lock.release()
        # Temp will then only be accessed by the pointer
        # That means that it does not really matter whether
        # The given instance was a rvalue or a lvalue (or glvalue, or etc...)

    # Default initializer
    def __init__(self: Self, val: T) -> None:
        self.lock:    Lock = Lock()
        self._val: _Pointer = val

# Factory method for returning an atomic of the instance
def atomic[T](instance: T) -> _AtomicInstance[T]:
    return _AtomicInstance(instance)

del _AtomicInstance
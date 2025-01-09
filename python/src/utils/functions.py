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
@version 0.0
"""
def clear_empty[T](collection: Iterable[T]) -> None:
    for element in collection:
        if isinstance(element, None) or not bool(element):
            del collection[element]

"""
Function that transforms a list to a dictionary based on a map function.
Note that it will directly return the instance if it's a dictionary.

@author  Thomas Gauthier
@version 0.0
"""
def transform_as_dict[T, Q](collection: dict[T, Q] | list[Q], mapper: Callable[[Q], T] | None = None) -> dict[T, Q]:
    if isinstance(collection, dict):
        return collection
    return dict.fromkeys(map(lambda key: mapper(key), collection), collection)

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

@author  Thomas Gauthier
@version 0.0
"""
from PySide6.QtWidgets import QWidget
def toggler(window: QWidget) -> Callable[..., None]:
    activated: bool = False
    def inner() -> None:
        nonlocal activated, window
        if activated: window.hide()
        else: window.show()
        activated = not activated
    return inner

"""
A class representing an atomic pointer.

@author  Thomas Gauthier
@version 0.0
"""
class _AtomicInstance[T](object):
    @property.getter
    def val(self) -> T:
        self.lock.acquire()
        ref: T = self.val.contents
        self.lock.release()
        return ref

    @property.setter
    def val(self, other: T) -> None:
        temp: T = other # Forces it to be a lvalue
        self.lock.acquire()
        self.val = pointer(temp)
        self.lock.release()
        # Temp will then only be accessed by the pointer
        # That means that it does not really matter whether
        # The given instance was a rvalue or a lvalue (or glvalue, or etc...)

    # Default initializer
    def __init__(self: Self, val: T) -> None:
        self.lock:    Lock = Lock()
        self.val: _Pointer = val

# Factory method for returning an atomic of the instance
def atomic[T](instance: T) -> _AtomicInstance[T]:
    return _AtomicInstance(instance)

del _AtomicInstance
#/usr/bin/env python3
"""
File containing all sorts of useful functions,
But doesn't hold any classes or anything relating to a specific field.

@author  Thomas Gauthier
@version 0.2
"""
from typing import Any, Callable, Iterable

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

    for _ in range(1, len(collection)):
        if collection[index + 1] == collection[index]: collection.pop(index)
        else: index += 1

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
    index: int = 0
    real:  int = 0
    for count in range(len(collection)):
        real = count - index
        if not bool(collection[real]):
            del collection[real]
            index += 1

"""
A useful function that will make a dictionary based on a list of
unique values as keys and map them (with the mapper) to their values.

@author  Thomas Gauthier
@version 0.0
"""
def mkdict[T, Q](keys: Iterable[T], mapper: Callable[[T], Q]) -> dict[T, Q]:
    dic: dict[T, Q] = {}

    for item in keys:
        if dic.get(item, None) is not None: raise Exception("Duplicate values in iterable")
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
        if dic.get(item, None) is not None: raise Exception("Duplicate values in iterable")
        else: dic[item] = next(values_iter)

    return dic


"""
A simple binary search implementation.

Returns -1 if the target does not exist.

@author  Thomas Gauthier
@version 0.1
"""
from math import floor
def binarySearch[T](ite: Any, target: T, key: Callable[..., T] | None = None) -> int:
    if len(ite) == 0: return -1
    if key is None: key = lambda x: x

    low:   int = 0
    high:  int = len(ite)

    mid:   int = floor((high - low) / 2)
    midv:    T = key(ite[mid])

    while midv != target:
        if low == high - 1: return -1
        elif midv < target: low = mid
        else: high = mid

        mid   = floor((high + low) / 2)
        midv  = key(ite[mid])

    return mid

"""
A function that mimics a binary search, but for finding the lowest index where
ite[index] < ite[other].

Note that if, foreach index in the array, the cutoff is less than or equal to
ite[index], this will return the index 0. The same is done, but with len(ite) - 1,
if, foreach index, ite[index] is less or equal to the cutoff.

@author  Thomas Gauthier
@version 0.1
"""
def cutoff[T](ite: Any, cutoff: T, key: Callable[..., T] | None = None) -> int:
    if len(ite) == 0: return 0
    if key is None: key = lambda x: x

    low:  int = 0
    high: int = len(ite)

    mid:  int = floor((high - low) / 2)
    midv:   T = key(ite[mid])

    while True:
        if low == high - 1: return low
        elif midv < cutoff: low = mid
        else: high = mid

        mid   = floor((high + low) / 2)
        midv  = key(ite[mid])

import pathlib as pl
import os.path as osp
import os

"""
Makes subdirectories from a given path.

@author  Thomas Gauthier
@version 0.0
"""
def mkabsent(di: pl.Path | str):
    if isinstance(di, str):
        di = pl.Path(di)

    if not osp.exists(di) or not osp.isdir(di):
        os.makedirs(di)
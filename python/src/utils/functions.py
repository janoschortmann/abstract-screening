#/usr/bin/env python3
"""
File containing all sorts of useful functions,
But doesn't hold any classes or anything relating to a specific field.

@author  Thomas Gauthier
@version 0.0
"""

from typing import Callable, Iterable

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

# Todo : Do the unique function for arbitrary collections

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

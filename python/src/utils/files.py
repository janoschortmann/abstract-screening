#!/usr/bin/env python3
"""
This file is used to declare useful/recurring structures in the project
that concern file manipulation/objects referring to files.

@author  Thomas Gauthier
@version 0.2
"""

from typing import Self

import datetime as dt

"""
A simple object class that has, as an objective, to only store information
relevant to a specific paper that was queried.

It can better be represented as a structure rather than an object.

This will be used, for example, to store all given papers in an array
for reference when showing them in the TableView and/or when doing further queries.

@author  Thomas Gauthier
@version 0.2
"""
class Paper(object):
    # Default initializer
    def __init__(
                self: Self,
                title: str,
                date: dt.date,
                directory: str,
                doi: str | None = None,
                label: int | None = None
                ) -> None:
        self.title:     str = title
        self.date:  dt.date = date
        self.directory: str = directory
        self.doi:       str = doi
        self.label:     int = label

    """
    Changes the state of the label to "1"

    Note that this could also be done with :
        ref.label = 1

    But this function is more meaningful

    The overall cost will be minimal since this function won't be called often
    """
    def accept(self: Self) -> None:
        self.label = 1

    """
    Changes the state of the label to "2"

    Note that this could also be done with :
        ref.label = 2

    But this function is more meaningful

    The overall cost will be minimal since this function won't be called often
    """
    def reject(self: Self) -> None:
        self.label = 2

    """
    Changes the state of the label to 0"

    Note that this could also be done with :
        ref.label = 0

    But this function is more meaningful

    The overall cost will be minimal since this function won't be called often
    """
    def reset(self: Self) -> None:
        self.label = 0

    """
    Basic hash funcion that returns the hash of this paper's title.

    This was chosen because a title should be unique and a given paper
    may have no DOI.
    """
    def __hash__(self: Self) -> int:
        return hash(self.title + str(self.date))

    """
    Basic equality function that compares both the titles and the dates.
    If they are the same, then you consider the papers to be the same.
    """
    def __eq__(self: Self, other) -> bool:
        # Impossibility of self reference in python
        return isinstance(other, self.__class__) and other.title == self.title and other.date == self.date

    # Lesser function. Used for data structures like sorting
    def __le__(self: Self, other) -> bool:
        return (self.title + str(self.date)) < (other.title + str(other.date))

    # Greater function. Used for data structures like sorting
    def __ge__(self: Self, other) -> bool:
        return not self.__le__(other) and not self.__eq__(other)
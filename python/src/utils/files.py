#!/usr/bin/env python3
"""
This file is used to declare useful/recurring structures in the project
that concern file manipulation/objects referring to files.

@author  Thomas Gauthier
@version 0.4
"""

from typing import Final, Self

import re

import os.path  as osp
import datetime as dt

CENTRAL: Final[str] = osp.expanduser("~/.ACAS/")
type lab = str

"""
A simple object class that has, as an objective, to only store information
relevant to a specific paper that was queried.

It can better be represented as a structure rather than an object.

This will be used, for example, to store all given papers in an array
for reference when showing them in the TableView and/or when doing further queries.

@author  Thomas Gauthier
@version 0.6
"""
class Paper(object):
    # Could be done with enums, but there are a pain to work with in python...
    LABELS: Final[list[lab]] = ["Unlabeled", "Accepted", "Rejected"]
    # The possibilities of a given button
    POSSIBILITIES: Final[list[lab]] = ["Labeled"] + LABELS

    # Default initializer
    def __init__(
                  self:  Self,
                  title: str,
                  abstr: str,
                  jour:  str,
                  date:  dt.date | None,
                  dire:  str,
                  doi:   str | None = None,
                  label: lab | None = LABELS[0]
                ) -> None:
        self.title:     str = title
        self.abstr:     str = abstr
        self.jour:      str = jour
        self.date:  dt.date = date
        self.dire:      str = dire
        self.doi:       str = doi
        self.label:     str = label
        self.prob:    float = -1.

    """
    Changes the state of the label to the one received.
    """
    def give(self: Self, label: lab) -> None:
        if not label in Paper.LABELS: raise Exception("Not in possible Labels")
        self.label: lab = label

    """
    Returns if a given label is labeled
    """
    def labeled(self: Self) -> bool:
        return self.label != Paper.LABELS[0]

    """
    A private function that will assign the probability of acceptance of a certain paper.
    """
    def assign(self: Self, prob: float) -> None:
        self.prob: float = prob

    """
    A static method used by the paper class for parsing a line.
    For any class that inherits from this, this function must be reimplemented.

    The string received is in the format of a csv with backslashes before quotes.
    """
    @staticmethod
    def parseLine(line: str, dire: str | None = None) -> ...:
        """
        Easier than to do the regex and less error prone.
        """
        strings: Final[list[str]] = line.split("\"")[1:-1]

        final:  list[str] = []
        concat: str = ""

        for string in strings:
            concat += string
            if not string or string[-1] != "\\":
                final.append(concat)
                concat = ""
            elif string and string[-1] == "\\":
                concat  = concat[:-1] + "\""
        paper: Paper = Paper(
            final[0],
            final[2],
            final[4],
            dt.datetime.strptime(final[6], "%Y-%m-%d",).date(),
            dire=dire,
            doi=final[8]
        )
        paper.dire = dire
        return paper

    """
    Basic hash funcion that returns the hash of this paper's title.

    This was chosen because a title should be unique and a given paper
    may have no DOI.
    """
    def __hash__(self: Self) -> int:
        return hash(self.title + str(self.date))

    """
    Returns the representation of this paper. Used for the stemming
    and vectorizing pocesses.
    """
    def __str__(self: Self) -> str:
        return self.title + " " + self.abstr + " " + self.jour

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
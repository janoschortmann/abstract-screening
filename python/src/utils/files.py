#!/usr/bin/env python3
"""
This file is used to declare useful/recurring structures in the project
that concern file manipulation/objects referring to files.

@author  Thomas Gauthier
@version 0.0
"""

from typing import Self

"""
A simple object class that has, as an objective, to only store information
relevant to a specific paper that was queried.

It can better be represented as a structure rather than an object.

This will be used, for example, to store all given papers in an array
for reference when showing them in the TableView and/or when doing further queries.

@author  Thomas Gauthier
@version 0.0
"""
class Paper(object):
    # Default initializer
    def __init__(self: Self, title: str, directory: str, doi: str, label: bool | None = None) -> None:
        self.title      = title
        self.directory  = directory
        self.doi        = doi
        self.label      = label

    """
    Changes the state of the label to "True"

    Note that this could also be done with :
        ref.label = True
    
    But this function is more meaningful

    The overall cost will be minimal since this function won't be called often
    """
    def accept(self) -> None:
        self.label = True

    """
    Changes the state of the label to "False"

    Note that this could also be done with :
        ref.label = False
    
    But this function is more meaningful

    The overall cost will be minimal since this function won't be called often
    """    
    def reject(self) -> None:
        self.label = False

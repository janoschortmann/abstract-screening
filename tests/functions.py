#!/usr/bin/env python3
"""
Basic testing file for the low level functions.

High level testing will not be performed with tests, but
with trial and errors.

@author  Thomas Gauthier
@version 0.0
"""
from typing import Callable, Iterable, Self

import python.src.utils.functions as funcs
import python.src.utils.files     as files

import datetime as dt
import unittest as uni

"""
Class that test the simple functions.

@author  Thomas Gauthier
@version 0.0
"""
class Functions(uni.TestCase):
    def setUp(self: Self) -> None:
        self.empty:    list[int] = []
        self.single:   list[int] = [1]
        self.unsorted: list[int] = [2, -3, 4, 1]

    def test_unique(self: Self) -> None:
        unique: Callable[..., bool] = lambda li: len(li) == len(set(li))
        # Checking the references
        self.assertTrue(unique(self.empty))
        funcs.unique(self.empty)

        self.assertEqual(self.empty, self.empty)
        self.assertTrue(unique(self.empty))

        self.assertTrue(unique(self.single))
        funcs.unique(self.single)

        self.assertEqual(self.single, self.single)
        self.assertTrue(unique(self.single))

        self.assertTrue(unique(self.unsorted))
        funcs.unique(self.unsorted)

        self.assertEqual(self.unsorted, self.unsorted)
        self.assertTrue(unique(self.unsorted))

        combined: list[int] = self.unsorted + self.unsorted
        self.assertFalse(unique(combined))
        funcs.unique(combined)

        self.assertEqual(combined, combined)
        self.assertTrue(unique(combined))

    def test_clear(self: Self) -> None:
        def hasEmpty(li: list[bool]) -> bool:
            for item in li:
                if not item: return True
            return False

        self.assertFalse(hasEmpty(self.empty))
        self.assertFalse(hasEmpty(self.single))

        copy: list[int] = self.single.copy()
        funcs.clearEmpty(self.single)
        self.assertEqual(copy, self.single)

        li: list[bool] = [True, True, False, False]
        self.assertTrue(hasEmpty(li))

        funcs.clearEmpty(li)
        self.assertEqual([True, True], li)

        self.assertEqual(len(li), 2)
        self.assertFalse(hasEmpty(li))

    def test_mkdict(self: Self) -> None:
        def mapped(keys: list[int], vals: list[int], mapper: Callable[[int], int]) -> bool:
            for count in range(len(keys)):
                if mapper(keys[count]) != vals[count]: return False
            return True

        call: Callable[[int], int] =  lambda val: val + 1
        identity: Callable[[int], int] = lambda val : val
        self.assertTrue(mapped(self.unsorted, list(funcs.mkdict(self.unsorted, identity).values()), identity))
        self.assertTrue(mapped(self.unsorted, list(funcs.mkdict(self.unsorted, call).values()),     call))
        self.assertTrue(mapped(self.empty,    list(funcs.mkdict(self.empty, call).values()),        call))

    def test_cbdict(self: Self) -> None:
        with self.assertRaises(Exception):
            funcs.cbdict([1, 2], [3])
        self.assertEqual({}, funcs.cbdict([], []))
        self.assertEqual({1: 1}, funcs.cbdict([1], [1]))

    """
    `toggler()` is fine, as shown when run with python, so no
    test will be created for this function.

    The same is said for the `appendParams()` function.
    """

    def test_search(self: Self) -> None:
        bs: Callable[[Iterable[int], int], int] = funcs.binarySearch
        self.assertEqual(-1, bs(self.empty, 1))
        self.assertEqual(-1, bs(self.single, 2))
        self.assertEqual(0,  bs(self.single, 1))

        twice: list[int] = [1, 2]
        self.assertEqual(0, bs(twice, 1))
        self.assertEqual(1, bs(twice, 2))


        thrice: list[int] = twice + [3]
        self.assertEqual(0, bs(thrice, 1))
        self.assertEqual(1, bs(thrice, 2))
        self.assertEqual(2, bs(thrice, 3))

    def test_cutoff(self: Self) -> None:
        cf: Callable[[Iterable[int], int], int] = funcs.cutoff
        self.assertEqual(0, cf(self.empty, 0))

        self.assertEqual(0, cf(self.single, 0))
        self.assertEqual(0, cf(self.single, 1))
        self.assertEqual(0, cf(self.single, 2))

        twice: list[int] = [1, 2]
        self.assertEqual(0, cf(twice, 0))
        self.assertEqual(0, cf(twice, 1))
        self.assertEqual(0, cf(twice, 2))
        self.assertEqual(1, cf(twice, 3))

        thrice: list[int] = twice + [3]
        self.assertEqual(0, cf(thrice, 0))
        self.assertEqual(0, cf(thrice, 1))
        self.assertEqual(0, cf(thrice, 2))
        self.assertEqual(1, cf(thrice, 3))
        self.assertEqual(2, cf(thrice, 4))

    """
    As the `mkabsent()` function requires access to the file system,
    the only reliable way to test this is with a build from github.

    Since this code is done as a PR and not a push, it is left to
    the maintainer to chose whether a build is required for this.

    Thus, the `mkabsent()` test function is left absent.
    """

"""
Class that test the simple functions inside the files file

@author  Thomas Gauthier
@version 0.0
"""
class Files(uni.TestCase):
    """
    Left empty, for now...
    """
    def setUp(self: Self) -> None: ...

    def test_parse(self: Self) -> None:
        first: list[str] = ["Title", "Abstract", "Journal", dt.datetime(1969, 7, 21).strftime("%Y-%m-%d"), ""]
        easy:  str = '\"' + "\", \"".join(first) + '\"'

        second: list[str] = [R"No, \"Title\"", R"Missing?!\"", "", dt.datetime(1989, 8, 18).strftime("%Y-%m-%d"), "Pump it Up"]
        hard:  str = '\"' + "\", \"".join(second) + '\"'

        error: str = """ ""Why."", Why should I be Hamlet? WHY??"""
        parse: Callable[[str, str], files.Paper] = files.Paper.parseLine

        # This must be done since the __eq__ in the Paper class does not compare each field
        def eq(first: files.Paper, second: files.Paper) -> bool:
            return (
                first.title     == second.title
                and first.abstr == second.abstr
                and first.jour  == second.jour
                and first.date  == second.date
                and first.doi   == second.doi
                and first.dire  == second.dire
            )

        dirf: str = "/dev/null/"
        first_parsed: files.Paper =  parse(easy, dirf)
        first = list(map(lambda string: string.replace(R"\"", "\""), first))
        self.assertTrue(
            eq(
                first_parsed,
                files.Paper(
                    first[0],
                    first[1],
                    first[2],
                    dt.datetime.strptime(first[3], "%Y-%m-%d").date(),
                    dire=dirf,
                    doi=first[4]
                )
            )
        )

        dirs: str = "rm -rf /"
        second_parsed: files.Paper =  parse(hard, dirs)
        second = list(map(lambda string: string.replace(R"\"", "\""), second))
        print(second)
        self.assertTrue(
            eq(
                second_parsed,
                files.Paper(
                    second[0],
                    second[1],
                    second[2],
                    dt.datetime.strptime(second[3], "%Y-%m-%d").date(),
                    dire=dirs,
                    doi=second[4]
                )
            )
        )

        with self.assertRaises(Exception):
            parse(error)

if __name__ == "__main__":  # Skip imports
    uni.main()
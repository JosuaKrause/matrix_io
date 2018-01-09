#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import gzip

import numpy as np


def _is_num(v):
    try:
        np.float64(v)
        return True
    except ValueError:
        return False


def _get_header(header, defaults, numbers, copy):
    if copy:
        header = list(header)
    if numbers is None:
        numbers = [ _is_num(d) for d in defaults ]
    elif copy:
        numbers = list(numbers)
    if len(header) != len(numbers):
        raise ValueError("header size != numbers size")
    if copy:
        defaults = [
            np.float64(d) if numbers[fix] else d
            for (fix, d) in enumerate(defaults)
        ]
    if len(header) != len(defaults):
        raise ValueError("header size != defaults size")
    return header, defaults, numbers


class SparseRow(object):
    def __init__(self, header, defaults, numbers=None, coo=[], copy=True):
        self._header, self._defaults, self._numbers = \
            _get_header(header, defaults, numbers, copy)
        self._fix_set = set()
        self._duplicates = False

        def check(fix, v):
            fix = int(fix)
            self._check_range(fix)
            if fix in self._fix_set:
                self._duplicates = True
            self._fix_set.add(fix)
            v = self._convert(fix, v)
            return (fix, v)

        self._row = [
            check(fix, v)
            for (fix, v) in coo
            if not self._is_default(int(fix), self._convert(int(fix), v))
        ]
        self._dense = None

    def _convert(self, fix, v):
        return np.float64(v) if self._numbers[fix] else v

    def _check_range(self, fix):
        if fix < 0 or fix >= len(self._defaults):
            raise IndexError("index out of bounds: {0}".format(fix))

    def _is_default(self, fix, v):
        default = self._defaults[fix]
        if v == default:
            return True
        if self._numbers[fix]:
            return np.isnan(v) and np.isnan(default)
        return False

    def remove_duplicates(self):
        if self._duplicates:
            self._get_row()
            self._from_dense()

    def get_values(self):
        self.remove_duplicates()
        for e in self._row:
            yield e

    def get_name(self, fix):
        return self._header[fix]

    def is_num(self, fix):
        return self._numbers[fix]

    def _from_dense(self):
        self._duplicates = False
        self._row = []
        self._fix_set = set()
        for (fix, v) in enumerate(self._dense):
            if not self._is_default(fix, v):
                self._row.append((fix, v))
                self._fix_set.add(fix)

    def from_dense(self, row):
        dense = [ self._convert(fix, v) for (fix, v) in enumerate(row) ]
        if len(dense) != len(self._defaults):
            raise ValueError("header")
        self._dense = dense
        self._from_dense()

    def _get_row(self):
        if self._dense is None:
            dense = list(self._defaults)
            for (fix, v) in self._row:
                dense[fix] = v
            self._dense = dense
        return self._dense

    def get_row(self):
        return list(self._get_row())

    def __getitem__(self, fix):
        fix = int(fix)
        self._check_range(fix)
        return self._get_row()[fix]

    def __setitem__(self, fix, v):
        fix = int(fix)
        self._check_range(fix)
        v = self._convert(fix, v)
        if fix in self._fix_set:
            self._duplicates = True
        elif self._is_default(fix, v):
            return
        self._fix_set.add(fix)
        self._row.append((fix, v))
        if self._dense is not None:
            self._dense[fix] = v

    def __delitem__(self, fix):
        fix = int(fix)
        self._check_range(fix)
        if fix not in self._fix_set:
            return
        self._duplicates = True
        self._fix_set.add(fix)
        v = self._defaults[fix]
        self._row.append((fix, v))
        if self._dense is not None:
            self._dense[fix] = v

    def clear(self):
        self._row = []
        self._dense = None
        self._fix_set = set()
        self._duplicates = False


class BaseFile(object):
    def __init__(self, fn, is_write, is_zip):
        if is_zip:
            mode = "wt" if is_write else "rt"
            self._f = gzip.open(fn, mode=mode, encoding="utf8", compresslevel=9)
        else:
            mode = "w" if is_write else "r"
            self._f = open(fn, mode=mode)
        self._csv = csv.writer(self._f) if is_write else csv.reader(self._f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._csv = None
        res = self._f.__exit__(exc_type, exc_value, traceback)
        return res

    def seek(self, pos, whence=0):
        return self._f.seek(pos, whence)

    def tell(self):
        return self._f.tell()

    def seekable(self):
        return self._f.seekable()

    def readable(self):
        return self._f.readable()

    def fileno(self):
        return self._f.fileno()

    def isatty(self):
        return self._f.isatty()

    def flush(self):
        self._f.flush()

    def close(self):
        self._csv = None
        self._f.close()

    @property
    def closed(self):
        return self._f.closed

    @property
    def __closed(self):
        return self._f.closed


class SparseWriter(BaseFile):
    def __init__(self, fn, header, defaults, numbers=None, copy=True, is_zip=True):
        BaseFile.__init__(self, fn, True, is_zip)
        try:
            self._header, self._defaults, self._numbers = \
                _get_header(header, defaults, numbers, copy)
            self._csv.writerow(self._header)
            self._csv.writerow(self._defaults)
        except:
            self.close()
            raise

    def write_dense_row(self, values):
        row = SparseRow(self._header, self._defaults, self._numbers, copy=False)
        row.from_dense(values)
        self.write_sparse_row(row.get_values())

    def write_sparse_row(self, sparse):
        self._csv.writerow([ "{0}:{1}".format(fix, v) for (fix, v) in sparse ])


class SparseLoader(BaseFile):
    def __init__(self, fn, header=None, defaults=None,
            numbers=None, copy=True, is_zip=True):
        BaseFile.__init__(self, fn, False, is_zip)
        try:
            if header is None:
                self._header = next(self._csv)
            if defaults is None:
                defaults = next(self._csv)
            self._header, self._defaults, self._numbers = \
                _get_header(header, defaults, numbers, copy)
        except:
            self.close()
            raise

    def get_header(self):
        return list(self._header)

    def get_defaults(self):
        return list(self._defaults)

    def __iter__(self):
        return self

    def __next__(self):
        return SparseRow(self._header, self._defaults, self._numbers,
            (e.split(':', 1) for e in next(self._csv)), copy=False)

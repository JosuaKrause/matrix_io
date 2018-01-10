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
        self._values = {}

        def add(fix, v):
            fix = int(fix)
            self._check_range(fix)
            v, is_d = self._convert(fix, v)
            if not is_d:
                self._values[fix] = v

        for (fix, v) in coo:
            add(fix, v)

    def __iter__(self):
        row = self

        class RowIter(object):
            def __init__(self):
                self.ix = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.ix >= len(row._defaults):
                    raise StopIteration
                res = row._get(self.ix)
                self.ix += 1
                return res

        return RowIter()

    def _convert(self, fix, v):
        default = self._defaults[fix]
        if self._numbers[fix]:
            v = np.float64(v)
            is_d = v == default or (np.isnan(v) and np.isnan(default))
            return v, is_d
        return v, v == default

    def _check_range(self, fix):
        if fix < 0 or fix >= len(self._defaults):
            raise IndexError("index out of bounds: {0}".format(fix))

    def get_values(self):
        return self._values.items()

    def get_name(self, fix):
        return self._header[fix]

    def is_num(self, fix):
        return self._numbers[fix]

    def from_dense(self, row):
        self._values = {}
        last_fix = 0
        for (fix, v) in enumerate(row):
            v, is_d = self._convert(fix, v)
            if not is_d:
                self._values[fix] = v
            last_fix = fix
        self._check_range(last_fix)

    def _get(self, fix):
        try:
            return self._values[fix]
        except KeyError:
            return self._defaults[fix]

    def __getitem__(self, fix):
        return self._get(int(fix))

    def __setitem__(self, fix, v):
        fix = int(fix)
        self._check_range(fix)
        v, is_d = self._convert(fix, v)
        if is_d:
            try:
                del self._values[fix]
            except KeyError:
                pass
            return
        self._values[fix] = v

    def __delitem__(self, fix):
        fix = int(fix)
        self._check_range(fix)
        try:
            del self._values[fix]
        except KeyError:
            pass

    def clear(self):
        self._values = {}


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

    def get_empty_row(self):
        return SparseRow(self._header, self._defaults, self._numbers, copy=False)

    def write_dense_row(self, values):
        row = self.get_empty_row()
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

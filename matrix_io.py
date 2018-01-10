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


def _get_numbers(defaults):
    return [ _is_num(d) for d in defaults ]


def _get_defaults(defaults, numbers):
    return [
        np.float64(d) if numbers[fix] else d
        for (fix, d) in enumerate(defaults)
    ]


def _get_converts(defaults, numbers):

    def get(fix):
        default = defaults[fix]

        def num(v):
            v = np.float64(v)
            is_d = v == default or (np.isnan(v) and np.isnan(default))
            return v, is_d

        def obj(v):
            return v, v == default

        return num if numbers[fix] else obj

    return [ get(fix) for fix in range(len(defaults)) ]


def _get_header(header, defaults, numbers, converts, copy):
    if copy:
        header = list(header)
    if numbers is None:
        numbers = _get_numbers(defaults)
    elif copy:
        numbers = list(numbers)
    if len(header) != len(numbers):
        raise ValueError("header size != numbers size")
    if copy:
        defaults = _get_defaults(defaults, numbers)
    if len(header) != len(defaults):
        raise ValueError("header size != defaults size")
    if converts is None:
        converts = _get_converts(defaults, numbers)
    elif copy:
        converts = list(converts)
    if len(header) != len(converts):
        raise ValueError("header size != converts size")
    return header, defaults, numbers, converts


class SparseRow(object):
    def __init__(self, header, defaults, numbers=None, converts=None,
            coo=[], copy=True, init={}, _shallow_copy=None):
        if _shallow_copy is not None:
            self._header = _shallow_copy._header
            self._defaults = _shallow_copy._defaults
            self._numbers = _shallow_copy._numbers
            self._converts = _shallow_copy._converts
            self._init = _shallow_copy._init
            self._values = _shallow_copy._values
            return
        self._header, self._defaults, self._numbers, self._converts = \
            _get_header(header, defaults, numbers, converts, copy)
        self._init = init.copy() if copy else init
        self.from_coo(coo) # initializes _values

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

    def _check_range(self, fix):
        if fix < 0 or fix >= len(self._defaults):
            raise IndexError("index out of bounds: {0}".format(fix))

    def get_values(self):
        return self._values.items()

    def get_name(self, fix):
        return self._header[fix]

    def is_num(self, fix):
        return self._numbers[fix]

    def from_coo(self, coo):

        def coo_iter():
            for (fix, v) in coo:
                fix = int(fix)
                self._check_range(fix)
                v, is_d = self._converts[fix](v)
                if not is_d:
                    yield (fix, v)

        self._values = self._init.copy()
        self._values.update(coo_iter())

    def from_dense(self, row):
        self._values = self._init.copy()
        last_fix = 0
        for (fix, v) in enumerate(row):
            v, is_d = self._converts[fix](v)
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
        v, is_d = self._converts[fix](v)
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
        self._values = self._init.copy()


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
    def __init__(self, fn, header, defaults, numbers=None, converts=None, copy=True, is_zip=True):
        BaseFile.__init__(self, fn, True, is_zip)
        try:
            self._header, self._defaults, self._numbers, self._converts = \
                _get_header(header, defaults, numbers, converts, copy)
            self._csv.writerow(self._header)
            self._csv.writerow(self._defaults)
        except:
            self.close()
            raise

    def get_empty_row(self):
        return SparseRow(self._header, self._defaults, self._numbers, self._converts, copy=False)

    def write_dense_row(self, values):
        row = self.get_empty_row()
        row.from_dense(values)
        self.write_sparse_row(row.get_values())

    def write_sparse_row(self, sparse):
        self._csv.writerow([ "{0}:{1}".format(fix, v) for (fix, v) in sparse ])


class SparseLoader(BaseFile):
    def __init__(self, fn, header=None, defaults=None, numbers=None,
            converts=None, copy=True, is_zip=True, map_defaults=None, overwrite_row=False):
        BaseFile.__init__(self, fn, False, is_zip)
        try:
            self._overwrite_row = overwrite_row
            if header is None:
                header = next(self._csv)
            if defaults is None:
                defaults = next(self._csv)
                if numbers is None:
                    numbers = _get_numbers(defaults)
                defaults = _get_defaults(defaults, numbers)
            init = {}
            if map_defaults is not None:

                def do_map(fix, d):
                    new_d = map_defaults(fix, d)
                    if d is not new_d:
                        init[fix] = d
                    return new_d

                defaults = [ do_map(fix, d) for (fix, d) in enumerate(defaults) ]
            self._header, self._defaults, self._numbers, self._converts = \
                _get_header(header, defaults, numbers, converts, copy)
            self._init = init
            self._row = SparseRow(self._header, self._defaults,
                self._numbers, self._converts, copy=False, init=self._init)
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
        if self._overwrite_row:
            row = self._row
        else:
            row = SparseRow(None, None, _shallow_copy=self._row)
        row.from_coo(e.split(':', 1) for e in next(self._csv))
        return row

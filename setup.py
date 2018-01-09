# -*- coding: utf-8 -*-
"""
Created on 2018-01-09

@author: joschi <josua.krause@gmail.com>

I/O for stramable sparse matrices.
"""
from setuptools import setup

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# NOTE! steps to distribute:
#$ python setup.py sdist bdist_wheel
#$ twine upload dist/... <- here be the new version!

setup(
    name='matrix_io',
    version='0.0.1',
    description='I/O for stramable sparse matrices.',
    long_description=long_description,
    url='https://github.com/JosuaKrause/matrix_io',
    author='Josua Krause',
    author_email='josua.krause@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    keywords='sparse matrix io stream',
    py_modules=['matrix_io'],
    install_requires=[],
)

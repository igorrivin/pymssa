#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import subprocess
import pip
import sys

REQUIREMENTS = [
    'pandas',
    'numpy',
    'pytest',
    'scipy',
    'numba',
    'tqdm',
    'matplotlib',
    'toolz',
]


setup(
    name='pymssa2',
    version='0.1.0',
    description="Multivariate Singular Spectrum Analysis (MSSA)",
    author="Kiefer Katovich and Igor Rivin",
    author_email='rivinh@temple.edu',
    url='https://github.com/igorrivin/pymssa.git',
    packages=find_packages(),
    package_dir={'pymssa2':'pymssa2'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    zip_safe=False,
    keywords='Python Multivariate Singular Spectrum Analysis MSSA',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)

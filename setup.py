#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 27 1:47 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/setup.py

@author: Dylan Neff, Dylan
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "bunch_density_cpp",
        ["bunch_density.cpp"],
    ),
]

setup(
    name="bunch_density_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

# -*- coding: utf-8 -*-
"""A python package to parse logged signia mat files.

@author: Ann-Kristin Seifer
@author: Arne Küderle
@author: Nils Roth

This package was build based on the python package NilsPodLib (https://nilspodlib.readthedocs.io/en/latest/)
and modified, such that it fits to data recorded using Signia hearing aids.
"""
from pathlib import Path

from .dataset import Dataset  # noqa: F401
from .session import Session  # noqa: F401

__all__ = ["Dataset", "Session"]
__version__ = "1.0.0"

SIGNIA_CAL_PATH = Path(__file__).parent / "calibrations"

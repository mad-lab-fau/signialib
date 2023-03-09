"""Exceptions and warnings used in the library."""
# import warnings
from pathlib import Path


def file_exists(path):
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        raise FileExistsError(f"File does not exists: {path}")

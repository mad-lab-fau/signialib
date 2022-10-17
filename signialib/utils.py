"""Set of helper functions used throughout the library."""

from pathlib import Path
from typing import TypeVar

import numpy as np
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct

path_t = TypeVar("path_t", str, Path)  # noqa: invalid-name
T = TypeVar("T")


def load_matlab(path: path_t, top_level: str) -> dict:
    """Load a mat file by matlab in python.

    Parameters
    ----------
    path :
        File path of the mat file.
    top_level:
        Variable name. Determined in matlab while saving.

    """
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False, mat_dtype=True)
    data = data[top_level]
    data = _convert_mat_struct_to_dict(data)
    if "s0" in data["data"]:
        raise ValueError(
            "Some data entries of mat files are probably in table format. Please make sure that the format is struct."
        )
    return data


def _convert_mat_struct_to_dict(struct: mat_struct) -> dict:
    """Convert the data from the matlab struct into a python dict.

    Parameters
    ----------
    struct:
       Matlab struct

    """
    out = {}
    for key in struct._fieldnames:
        val = getattr(struct, key)
        if isinstance(val, sio.matlab.mio5_params.mat_struct):
            # recursive function call
            out[key] = _convert_mat_struct_to_dict(val)
        elif isinstance(val, (list, np.ndarray)) and len(val) == 0:
            out[key] = val
        elif isinstance(val, (list, np.ndarray)) and isinstance(val[0], sio.matlab.mio5_params.mat_struct):
            tmp = [_convert_mat_struct_to_dict(v) for v in val]
            out[key] = tmp
        else:
            out[key] = val
    return out

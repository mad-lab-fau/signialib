# -*- coding: utf-8 -*-
"""Fundamental Datastream class, which holds any type of sensor_type data and handles basic interactions with it."""

from typing import Iterable, List, Optional

import numpy as np
import seaborn as sns
from fau_colors import register_cmaps
from nilspodlib.datastream import Datastream as DatastreamNilsPodLib

register_cmaps()
sns.set_palette("faculties")


class Datastream(DatastreamNilsPodLib):
    """Object representing a single set of data of one sensor_type.

    Copy NilsPodLib with additional plotting function.

    Usually it is not required to directly interact with the datastream object (besides accessing the data attribute).
    Most important functionality can/should be used via a dataset or session object to manage multiple datasets at once.

    Attributes
    ----------
    data
        The actual data of the sensor_type as `np.array`.
        Can have multiple dimensions depending on the sensor_type.
    sensor_type
        The name of the sensor_type
    is_calibrated
        If the sensor_type is in a raw format or the expected final output units
    is_factory_calibrated
        If the datastream was factory calibrated and hence, provided in physical meaningful units.
        This should be True if the datastream was loaded using the methods to load datasets and sessions provided in
        this library.
    sampling_rate_hz
        The sampling rate of the datastream

    """

    data: np.ndarray
    is_calibrated: bool = False
    is_factory_calibrated: bool = False
    sampling_rate_hz: float
    sensor_type: Optional[str]
    calibrated_unit: Optional[str]
    columns: List[str]

    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: float = 1.0,
        columns: Optional[Iterable] = None,
        calibrated_unit: Optional[str] = None,
        sensor_type: Optional[str] = None,
    ):
        """Get new datastream instance.

        Parameters
        ----------
        data :
            The actual data to be stored in the datastream.
            Should be 2D (even if only 1D data is stored).
            First dimension should be the time axis and second dimension the different data vector entries
            (e.g. acc_x, acc_y, acc_z).
        sampling_rate :
            The sampling rate of the datastream.
            Is used for all calculations that require sampling info.
        columns :
            Optional list of names for the data vector entries.
            Only used to make it easier to understand the content of a datastream.
        calibrated_unit :
            The expected unit of the datastream after calibration.
        sensor_type :
            Type of sensor_type the data is produced from.
            This allows to automatically get default values for columns and units from :py:mod:`nilspodlib.consts`.

        """
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            columns=columns,
            calibrated_unit=calibrated_unit,
            sensor_type=sensor_type,
        )

    def plot(self, ax, x_axis):
        ax.plot(x_axis, self.data, label=self.columns)
        ax.set_ylabel(self.sensor_type + " (" + self.unit + ")")
        ax.legend()

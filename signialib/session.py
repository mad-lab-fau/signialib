# -*- coding: utf-8 -*-
"""Session groups multiple Datasets from sensors recorded at the same time."""
import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from nilspodlib.exceptions import SynchronisationError
from nilspodlib.utils import inplace_or_copy, path_t
from scipy.signal import resample

from signialib._session_base import _MultiDataset
from signialib.dataset import Dataset
from signialib.exceptions import file_exists
from signialib.header import _ProxyHeader

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401
    from imucal import CalibrationInfo  # noqa: F401

T = TypeVar("T", bound="Session")


class Session(_MultiDataset):
    """Object representing a collection of Datasets.

    .. note:: By default a session makes no assumption about when and how datasets are recorded.
              It just provides an interface to manipulate multiple datasets at once.
              If you have datasets that were recorded simultaneously with active sensor_type synchronisation,
              you should use a `SyncedSession` instead of a `Session` to take full advantage of this.

    A session can access all the same attributes and most of the methods provided by a dataset.
    However, instead of returning a single value/acting only on a single dataset, it will return a tuple of values (one
    for each dataset) or modify all datasets of a session.
    You can also use the `self.info` object to access header information of all datasets at the same time.
    All return values will be in the same order as `self.datasets`.

    Attributes
    ----------
    datasets
        A tuple of the datasets belonging to the session
    info
        Get the metadata (Header) information for all datasets
    labels :
        Labels: Optional

    See Also
    --------
    signialib.session

    """

    datasets: Tuple[Dataset]
    info: _ProxyHeader
    labels: Optional[pd.Series]
    _synced = False

    @property
    def info(self):
        """Get metainfo for all datasets.

        All attributes of :py:class:`nilspodlib.header.HeaderInfo` are supported in read-only mode.
        """
        return _ProxyHeader(tuple(d.info for d in self.datasets))

    def __init__(self, datasets: Iterable[Dataset]):
        """Create new session.

        Instead of this init you can also use the factory methods :py:meth:`~nilspodlib.session.Session.from_file_paths`
        and :py:meth:`~nilspodlib.session.Session.from_folder_path`.

        Parameters
        ----------
        datasets :
            List of :py:class:`nilspodlib.dataset.Dataset` instances, which should be grouped into a session.

        """
        self.datasets = tuple(datasets)

    @classmethod
    def from_file_path(cls: Type[T], path: str) -> T:
        """Create a new session from a list of files pointing to valid .mat files.

        Parameters
        ----------
        path :
            A path pointing to a .mat or .txt file.

        """
        file_exists(path)
        if Path(path).suffix == ".txt":
            ds, labels = Dataset.from_txt_file(path)
        elif Path(path).suffix == ".mat":
            ds = Dataset.from_mat_file(path)
            labels = None
        session = cls([ds])
        session.labels = labels
        return session

    @classmethod
    def from_file_paths(cls: Type[T], paths: Iterable[path_t]) -> T:
        """Create a new session from a list of files pointing to valid .mat files.

        Parameters
        ----------
        paths :
            List of paths pointing to files to be included

        Note
        ----
        Works only for .mat files. For .txt files please use Session.from_file_path().

        """
        if str == type(paths):
            raise ValueError("No iterable input.")
        ds = (Dataset.from_mat_file(p) for p in paths)
        return cls(ds)

    @classmethod
    def from_folder_path(
        cls: Type[T],
        base_path: path_t,
        filter_pattern: str = "*.mat",
    ) -> T:
        """Create a new session from a folder path containing valid .mat files.

        Parameters
        ----------
        base_path :
            Path to the folder
        filter_pattern :
            regex that can be used to filter the files in the folder. This is passed to Pathlib.glob()

        """
        if not Path(base_path).is_dir():
            raise ValueError(f"The folder {base_path} does not exist.")
        ds = list(Path(base_path).glob(filter_pattern))
        if not ds:
            if list(Path(base_path).glob("*.txt")):
                raise ValueError(
                    f'No files matching "{filter_pattern}" where found in {base_path}. '
                    f'For "*.txt" files, please consider using the "Session.from_file_path" function.'
                )
            raise ValueError(f'No files matching "{filter_pattern}" where found in {base_path}.')
        return cls.from_file_paths(ds)

    def get_dataset_by_id(self, sensor_id: str) -> Dataset:
        """Get a specific dataset by its sensor_type id.

        Parameters
        ----------
        sensor_id :
            Four letter/digit unique id of the sensor

        """
        return self.datasets[self.info.sensor_id.index(sensor_id)]

    def get_dataset_by_position(self, sensor_position: str) -> Dataset:
        """Get a specific dataset by the sensor's position.

        Parameters
        ----------
        sensor_position :
            Position of the placement of the sensor, e.g. ha_left for hearing aid on the left.

        """
        if sensor_position not in self.info.sensor_position:
            raise ValueError(
                f"Sensor position is not definded: {sensor_position}.\n"
                f" Please choose a valid sensor_position: {self.info.sensor_position}"
            )

        return self.datasets[self.info.sensor_position.index(sensor_position)]

    def calibrate_imu(self: T, calibrations: Iterable[Union["CalibrationInfo", path_t]], inplace: bool = False) -> T:
        """Calibrate the imus of all datasets by providing a list of calibration infos.

        If you do not want to calibrate a specific IMU, you can pass `None` for its position.

        Parameters
        ----------
        calibrations :
            List of calibration infos in the same order than `self.datasets`
        inplace :
            If True this methods modifies the current session object. If False, a copy of the sesion and all
            dataset objects is created

        See Also
        --------
        nilspodlib.dataset.Dataset.calibrate_imu

        """
        s = inplace_or_copy(self, inplace)
        s.datasets = [d.calibrate_imu(c, inplace=True) if c else d for d, c in zip(s.datasets, calibrations)]
        return s

    def align_to_syncregion(self: T, inplace: bool = False) -> T:
        """Align left and right hearing aid based on data from left device.

        At the end all datasets are cut to the same length.

        Parameters
        ----------
        inplace :
            If operation should be performed on the current Session object, or on a copy

        Returns
        -------
        Session as a SigniaSession

        """
        s = inplace_or_copy(self, inplace)

        if s._synced is True:
            raise SynchronisationError("The session is already aligned/cut to the syncregion and can not be cut again")

        resample_len = s.get_dataset_by_position("ha_left").acc.data.shape[0]

        for idx, sensor in enumerate(s.acc):
            s.acc[idx].data = resample(sensor.data, resample_len, axis=0)
            s.datasets[idx].counter = np.arange(0, resample_len)

        for idx, sensor in enumerate(s.gyro):
            s.gyro[idx].data = resample(sensor.data, resample_len, axis=0)

        s._synced = True
        return s

    def resample(self: T, resample_rate_hz: int, inplace: bool = False) -> T:
        """Resample all datasets.

        Parameters
        ----------
        resample_rate_hz:
            Target sample rate in Hz
        inplace :
            If operation should be performed on the current Session object, or on a copy

        Returns
        -------
        Session as a SigniaSession

        """
        s = inplace_or_copy(self, inplace)
        if self.info.sampling_rate_hz[0] < resample_rate_hz:
            raise ValueError("Can not resample to a larger frequency.")
        if np.mod(self.info.sampling_rate_hz[0], resample_rate_hz) != 0.0:
            raise ValueError("Sample rate must be multiple of resample frequency.")
        factor = int(self.info.sampling_rate_hz[0] // resample_rate_hz)
        s.datasets = [d.downsample(factor=factor, inplace=True) for d in s.datasets]
        return s

    def align_calib_resample(
        self: T,
        calib_path: path_t = None,
        resample_rate_hz: int = None,
        inplace: bool = False,
        skip_calibration: bool = False,
        warn_thres_days: int = 60,
    ) -> T:
        """Aligns calibrates and resamples all datasets in session.

        Parameters
        ----------
        calib_path:
            Local directory containing the calibration json files
        resample_rate_hz:
            Target sample rate in Hz
        inplace:
            If operation should be performed on the current Session object, or on a copy
        skip_calibration:
            Set to 'True' is no calibration files (by Ferraris) are given.
        warn_thres_days:
            Warning if calicration is older than this threshold in days.

        Returns
        -------
        Session as a SigniaSession

        """
        if calib_path is None and skip_calibration is False:
            raise ValueError("Please provide a calibration path or set 'skip_calibration=True'")

        s = inplace_or_copy(self, inplace)

        if len(s.datasets) == 2:
            s = s.align_to_syncregion()

        if skip_calibration is False:
            s = s.calibrate_imu(
                self.find_closest_calibration(calib_path, warn_thres=datetime.timedelta(days=warn_thres_days))  # noqa
            )  # noqa
        if resample_rate_hz is not None:
            s = s.resample(resample_rate_hz)
        return s

    def plot(self, index: str = None):
        """Plot data.

        Parameters
        ----------
        index: {None, "local_datetime"}
            Defines x axis label ticks of plot. Default is None, i.e. samples.

        """
        self.datasets = [d.plot(index=index) for d in self.datasets]
        return self

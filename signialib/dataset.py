# -*- coding: utf-8 -*-
"""Dataset represents a measurement session of a single sensor_type."""
import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from nilspodlib.calibration_utils import find_closest_calibration_to_date, load_and_check_cal_info
from nilspodlib.datastream import Datastream
from nilspodlib.exceptions import RepeatedCalibrationError, datastream_does_not_exist_warning
from nilspodlib.utils import inplace_or_copy, path_t

from signialib.consts import GRAV, SENSOR_MAPPINGS
from signialib.header import Header
from signialib.utils import load_matlab

if TYPE_CHECKING:
    from imucal import CalibrationInfo  # noqa: F401

T = TypeVar("T")


class Dataset:  # noqa: too-many-public-methods
    """Class representing a logged session of a single signia hearing aid.

    .. warning:: Some operations on the dataset should not be performed after each other, as they can lead to unexpected
                 results.
                 The respective methods have specific warnings in their docstring.

    Each instance has 3 important (groups of attributes):

    - self.info: A instance of `signialib.header.Header` containing all the meta info about the measurement.
    - self.counter: The continuous counter created by the sensor.
      It is in particular important to synchronise multiple datasets that were recorded at the same time
      (see `nilspodlib.session.SyncedSession`).
    - datastream: The actual sensor_type data accessed directly by the name of the sensor_type
      (e.g. acc, gyro, baro, ...).
      Each sensor_type data is wrapped in a `signialib.datastream.Datastream` object.

    Attributes
    ----------
    path :
        Path pointing to the recording file (if dataset was loaded from a file)
    info :
        Metadata of the recording
    size :
        The number of samples in the dataset.
    counter :
        The continuous counter of the sensor.
    time_counter :
        Counter in seconds since first sample.
    utc_counter :
        Counter as utc timestamps.
    utc_datetime_counter :
        Counter as np.datetime64 in UTC timezone.
    active_sensor :
        The enabled sensors in the dataset.
    datastreams :
        Iterator over all available datastreams/sensors
    acc :
        Optional accelerometer datastream.
    gyro :
        Optional gyroscope datastream.

    """

    path: path_t
    acc: Optional["Datastream"] = None
    gyro: Optional["Datastream"] = None
    counter: np.ndarray
    info: Header

    @property
    def size(self) -> int:
        """Get the number of samples in the Dataset."""
        return len(self.counter)

    @property
    def active_sensors(self) -> Tuple[str]:
        """Get the enabled sensors in the dataset."""
        return tuple(self.info.enabled_sensors)

    @property
    def datastreams(self) -> Iterable[Datastream]:
        """Iterate through all available datastreams."""
        for i in self.active_sensors:
            yield i, getattr(self, i)

    @property
    def time_counter(self) -> np.ndarray:
        """Counter in seconds since first sample."""
        return (self.counter - self.counter[0]) / self.info.sampling_rate_hz

    def __init__(self, sensor_data: Dict[str, np.ndarray], counter: np.ndarray, info: Header):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_bin_file` constructor to handle loading recorded Signia Hearing Aid Sessions.

        Parameters
        ----------
        sensor_data :
            Dictionary with name of sensor_type and sensor_type data as np.array
            The data needs to be 2D with time/counter as first dimension
        counter :
            The counter created by the sensor_type. Should have the same length as all datasets
        info :
            Header instance containing all Metainfo about the measurement.

        """
        self.counter = counter
        self.info = info

        calibration_dict = {
            "acc": self._factory_calibrate_acc,
            "gyro": self._factory_calibrate_gyro,
        }

        for k, v in sensor_data.items():
            ds = Datastream(v, self.info.sampling_rate_hz, sensor_type=k)
            if k in calibration_dict:
                ds = calibration_dict[k](ds)
            setattr(self, k, ds)

    @classmethod
    def from_mat_file(cls: Type[T], path: path_t) -> T:
        """Create a new Dataset from a valid .mat file.

        Parameters
        ----------
        path :
            Path to the file

        """
        path = Path(path)
        if path.suffix != ".mat":
            ValueError(f'Invalid file type! Only ".bin" files are supported not {path}')

        # todo
        sensor_data, counter, info = parse_mat(path)
        s = cls(sensor_data, counter, info)

        s.path = path
        return s

    def _get_info(self):
        header_dict = {k: v for k, v in self.info.__dict__.items() if k in self.info._header_fields}
        for key, val in header_dict.items():
            print(f"{key}: {val}")

    def calibrate_imu(self: T, calibration: Union["CalibrationInfo", path_t], inplace: bool = False) -> T:
        """Apply a calibration to the Acc and Gyro datastreams.

        The final units of the datastreams will depend on the used calibration values, but must likely they will be "g"
        for the Acc and "dps" (degrees per second) for the Gyro.

        Parameters
        ----------
        calibration :
           calibration object or path to .json file, that can be used to create one.
        inplace :
           If True this methods modifies the current dataset object. If False, a copy of the dataset and all
           datastream objects is created
           Notes:
        inplace :
           If True this methods modifies the current dataset object. If False, a copy of the dataset and all
           datastream objects is created
           Notes:
           This just combines `calibrate_acc` and `calibrate_gyro`.

        """
        s = inplace_or_copy(self, inplace)

        check = [self._check_calibration(s.acc, "acc"), self._check_calibration(s.gyro, "gyro")]
        # check should be check = [True, True], list 2

        if all(check):
            # todo: start
            calibration = load_and_check_cal_info(calibration)
            # todo: stop
            acc, gyro = calibration.calibrate(s.acc.data, s.gyro.data, acc_unit=s.acc.unit, gyr_unit=s.gyro.unit)
            s.acc.data = acc
            s.gyro.data = gyro
            s.acc.is_calibrated = True
            s.acc.calibrated_unit = calibration.acc_unit
            s.gyro.is_calibrated = True
            s.gyro.calibrated_unit = calibration.gyr_unit
        return s

    @staticmethod
    def _check_calibration(ds: Optional[Datastream], name: str, factory: bool = False):
        """Check if a specific datastream is already marked as calibrated, or if the datastream does not exist.

        In case the datastream is already calibrated a `RepeatedCalibrationError` is raised.
        In case the datastream does not exist, a warning is raised.

        Parameters
        ----------
        ds :
            datastream object or None
        name :
            name of the datastream object. Used to provide additional info in error messages.
        factory :
            If we want to check for factory calibration or not.
            If True, it will only be checked if the dataset is factory calibrated.
            If False, it will be checked if the dataset is normally calibrated.

        """
        if ds is not None:
            if factory is True:
                check_val = ds.is_factory_calibrated
            else:
                check_val = ds.is_calibrated
            if check_val is True:
                raise RepeatedCalibrationError(name, factory)
            return True
        datastream_does_not_exist_warning(name, "calibration")
        return False

    def _factory_calibrate_gyro(self, gyro: Datastream) -> Datastream:
        """Apply a factory calibration to the Gyro datastream.

        The values used for that are taken from the datasheet of the sensor_type and are likely not to be accurate.
        For any tasks requiring precise sensor_type outputs, `calibrate_gyro` should be used with measured calibration
        values.

        The final units of the output will be "deg/s" (degrees per second) for the Gyro.

        Parameters
        ----------
        gyro :
            The uncalibrated gyro Datastream

        """
        assert gyro.sensor_type == "gyro"
        if self._check_calibration(gyro, "gyro", factory=True) is True:
            # gyro.data /= 2 ** 16 / self.info.gyro_range_dps / 2
            gyro.is_factory_calibrated = True
        return gyro

    def _factory_calibrate_acc(self, acc: Datastream) -> Datastream:
        """Apply a factory calibration to the Acc datastream.

        The values used for that are taken from the datasheet of the sensor_type and are likely not to be accurate.
        For any tasks requiring precise sensor_type outputs, `calibrate_acc` should be used with measured calibration
        values.

        The final units of the output will be "m/s^2" for the Acc.

        Parameters
        ----------
        acc :
            The uncalibrated acc Datastream

        """
        assert acc.sensor_type == "acc"
        if self._check_calibration(acc, "acc", factory=True) is True:
            # acc.data /= 2 ** 16 / self.info.acc_range_g / 2 / GRAV
            acc.data *= GRAV
            acc.is_factory_calibrated = True
        return acc

    def downsample(self: T, factor: int, inplace: bool = False) -> T:
        """Downsample all datastreams by a factor.

        This applies `scipy.signal.decimate` to all datastreams and the counter of the dataset.
        See :py:meth:`nilspodlib.datastream.Datastream.downsample` for details.

        .. warning::
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the
            header sync index values. Using methods that rely on these values might result in unexpected behaviour.
            For example `cut_to_syncregion` will not work correctly, if `cut`, `cut_counter_val`, or `downsample`
            was used before.

        Parameters
        ----------
        factor :
            Factor by which the dataset should be downsampled.
        inplace :
            If True this methods modifies the current dataset object. If False, a copy of the dataset and all
            datastream objects is created

        """
        from scipy.signal import resample  # noqa: import-outside-toplevel

        s = inplace_or_copy(self, inplace)
        for key, val in s.datastreams:
            setattr(s, key, val.downsample(factor))
        s.counter = resample(s.counter, len(s.counter) // factor, axis=0)
        s.info.sampling_rate_hz /= factor
        return s

    def data_as_df(
        self,
        datastreams: Optional[Sequence[str]] = None,
        index: Optional[str] = None,
        include_units: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Export the datastreams of the dataset in a single pandas DataFrame.

        Parameters
        ----------
        datastreams :
            Optional list of datastream names, if only specific ones should be included. Datastreams that
            are not part of the current dataset will be silently ignored.
        index :
            Specify which index should be used for the dataset. The options are:
            "counter": For the actual counter
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`

        Notes
        -----
        This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        Therefore, it will use the column information of each datastream.

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        index_names = {
            None: "n_samples",
            "counter": "n_samples",
            "time": "t",
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")

        index_name = index_names[index]

        datastreams = datastreams or self.active_sensors
        dfs = [s.data_as_df(include_units=include_units) for k, s in self.datastreams if k in datastreams]

        df = pd.concat(dfs, axis=1)

        if index:
            if index != "counter":
                index += "_counter"
            index = getattr(self, index, None)
            df.index = index
        else:
            df = df.reset_index(drop=True)
        df.index.name = index_name
        return df

    def imu_data_as_df(self, index: Optional[str] = None, include_units: Optional[bool] = False) -> pd.DataFrame:
        """Export the acc and gyro datastreams of the dataset in a single pandas DataFrame.

        See Also
        --------
        nilspodlib.dataset.Dataset.data_as_df

        Parameters
        ----------
        index :
            Specify which index should be used for the dataset. The options are:
            "counter": For the actual counter
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`

        Notes
        -----
        This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        Therefore, it will use the column information of each datastream.


        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        return self.data_as_df(datastreams=["acc", "gyro"], index=index, include_units=include_units)

    #    def find_calibrations(
    #        self,
    #        folder: Optional[path_t] = None,
    #        recursive: bool = True,
    #        filter_cal_type: Optional[str] = None,
    #        ignore_file_not_found: Optional[bool] = False,
    #    ) -> List[Path]:
    #        """Find all calibration infos that belong to a given sensor_type.
    #
    #        As this only checks the filenames, this might return a false positive depending on your folder structure
    #        and naming.
    #
    #        Parameters
    #        ----------
    #        folder :
    #            Basepath of the folder to search. If None, tries to find a default calibration
    #        recursive :
    #            If the folder should be searched recursive or not.
    #        filter_cal_type :
    #            Whether only files obtain with a certain calibration type should be found.
    #            This will look for the `CalType` inside the json file and hence cause performance problems.
    #            If None, all found files will be returned.
    #            For possible values, see the `imucal` library.
    #        ignore_file_not_found :
    #            If True this function will not raise an error, but rather return an empty list, if no
    #            calibration files were found for the specific sensor_type.
    #
    #        See Also
    #        --------
    #        nilspodlib.calibration_utils.find_calibrations_for_sensor
    #
    #        """
    #        # TODO: Test
    #        return find_calibrations_for_sensor(
    #            sensor_id=self.info.sensor_id,
    #            folder=folder,
    #            recursive=recursive,
    #            filter_cal_type=filter_cal_type,
    #            ignore_file_not_found=ignore_file_not_found,
    #        )

    def find_closest_calibration(
        self,
        folder: Optional[path_t] = None,
        recursive: bool = True,
        filter_cal_type: Optional[str] = None,
        before_after: Optional[str] = None,
        ignore_file_not_found: Optional[bool] = False,
        warn_thres: datetime.timedelta = datetime.timedelta(days=60),  # noqa E252
    ) -> Path:
        """Find the closest calibration info to the start of the measurement.

        As this only checks the filenames, this might return a false positive depending on your folder structure and
        naming.

        Parameters
        ----------
        folder :
            Basepath of the folder to search. If None, tries to find a default calibration
        recursive :
            If the folder should be searched recursive or not.
        filter_cal_type :
           Whether only files obtain with a certain calibration type should be found.
           This will look for the `CalType` inside the json file and hence cause performance problems.
           If None, all found files will be returned.
           For possible values, see the `imucal` library.
        before_after :
           Can either be 'before' or 'after', if the search should be limited to calibrations that were
           either before or after the specified date.
        warn_thres :
           If the distance to the closest calibration is larger than this threshold, a warning is emitted
        ignore_file_not_found :
           If True this function will not raise an error, but rather return `None`, if no
           calibration files were found for the specific sensor_type.

        See Also
        --------
        nilspodlib.calibration_utils.find_calibrations_for_sensor
        nilspodlib.calibration_utils.find_closest_calibration_to_date

        """
        return find_closest_calibration_to_date(
            sensor_id=self.info.sensor_id.lower(),
            cal_time=self.info.utc_datetime_start,
            folder=folder,
            recursive=recursive,
            filter_cal_type=filter_cal_type,
            before_after=before_after,
            warn_thres=warn_thres,
            ignore_file_not_found=ignore_file_not_found,
        )


def parse_mat(path: path_t) -> Tuple[Dict[str, np.ndarray], np.ndarray, Header]:
    """Parse a binary NilsPod session file and read the header and the data.

    Parameters
    ----------
    path :
        Path to the file

    Returns
    -------
    sensor_data :
        The sensor data as dictionary
    counter :
        The counter values
    session_header :
        The session header

    """
    data = load_matlab(path, "deviceData")

    session_header = Header.from_dict(data["metaInfo"])

    data_stream = pd.DataFrame(data=data["data"])
    counter, sensor_data = split_into_sensor_data(data_stream, session_header)

    return sensor_data, counter, session_header


def split_into_sensor_data(data: pd.DataFrame, session_header: Header) -> Dict[str, np.ndarray]:
    """Split/Parse the binary data into the different sensors and the counter.

    Parameters
    ----------
    data :
        Data to be split
    session_header:
        The session header

    Returns
    -------
    counter:
        The counter values

    sensor_data :
        The sensor data as dictionary

    """
    sensor_data = {}
    n_samples = data.shape[0]

    for sensor in session_header.enabled_sensors:
        mappings = SENSOR_MAPPINGS[sensor]

        data_stream = np.zeros((n_samples, 3))
        for idx, axis in enumerate(mappings):
            data_stream[:, idx] = data[axis].to_numpy()
        sensor_data[sensor] = data_stream

    counter = np.arange(0, n_samples, 1)

    return counter, sensor_data

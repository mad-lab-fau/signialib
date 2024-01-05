# -*- coding: utf-8 -*-
"""Header class(es), which are used to read and access basic information from a recorded session."""

import datetime
import itertools
import pprint
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytz


class _HeaderFields:
    """Base class listing all the attributes of a session header.

    For documentation see the `Header` object.
    """

    enabled_sensors: Tuple[str]

    sensor_position: str

    sampling_rate_hz: float
    acc_range_g: float
    gyro_range_dps: float

    utc_start: int
    utc_stop: int

    firmware_version: str

    sensor_id: str

    app_version: str

    ruleset_version: str

    model_name: str

    platform: str

    custom_meta_data: Tuple[float]

    n_samples: int

    imu_sensor_type: int

    _SENSOR_FLAGS = OrderedDict(
        [
            ("gyro", "gyroscope"),
            ("acc", "accelerometer"),
        ]
    )

    @property
    def _header_fields(self) -> List[str]:
        """List all header fields.

        This is a little hacky and relies on that the header fields are the only attributes that are type annotated.
        """
        return list(_HeaderFields.__annotations__.keys())

    @property
    def _all_header_fields(self) -> List[str]:
        additional_fields = [
            "duration_s",
            "utc_datetime_start",
            "utc_datetime_stop",
            "has_position_info",
            "sensor_id",
            "strict_version_firmware",
        ]

        return self._header_fields + additional_fields

    @property
    def duration_s(self) -> int:
        """Length of the measurement."""
        return self.utc_stop - self.utc_start

    @property
    def utc_datetime_start(self) -> datetime.datetime:
        """Start time as utc datetime."""
        return datetime.datetime.utcfromtimestamp(self.utc_start).replace(tzinfo=pytz.utc)

    @property
    def utc_datetime_stop(self) -> datetime.datetime:
        """Stop time as utc datetime."""
        return datetime.datetime.utcfromtimestamp(self.utc_stop).replace(tzinfo=pytz.utc)

    @property
    def local_datetime_start(self) -> datetime.datetime:
        """Start time in specified timezone."""
        return datetime.datetime.fromtimestamp(self.utc_start)

    @property
    def local_datetime_stop(self) -> datetime.datetime:
        """Start time in specified timezone."""
        return datetime.datetime.fromtimestamp(self.utc_stop)

    @property
    def has_position_info(self) -> bool:
        """If any information about the sensor position is provided."""
        return not self.sensor_position == "undefined"

    def __str__(self) -> str:
        full_header = {k: getattr(self, k, None) for k in self._all_header_fields}
        return pprint.pformat(full_header)


class Header(_HeaderFields):
    """Additional Infos of recording.

    Usually their is no need to use this class on its own, but it is just used as a convenient wrapper to access all
    information via a dataset instance.

    .. warning :: UTC timestamps and datetime, might not be in UTC. We just provide the values recorded by the sensor
                  without any local conversions.
                  Depending on the recording device, a localized timestamp might have been set for the internal sensor
                  clock

    Attributes
    ----------
    sensor_id
        Get the unique sensor identifier.
    enabled_sensors
        Tuple of sensors that were enabled during the recording.
        Uses typical shorthands.
    imu_sensor_type
        Fabric name of IMU sensor.
    sensor_position
        If a sensor position was specified.
        Can be a position from a list or custom bytes.
    has_position_info
        If any information about the sensor position is provided.
    sampling_rate_hz
        Sampling rate of the recording.
    acc_range_g
        Range of the accelerometer in g.
    gyro_range_dps
        Range of the gyroscope in deg per s.

    utc_start
        Unix time stamp of the start of the recording.

        .. note:: No timezone is assumed and client software is instructed to set the internal sensor clock to utc time.
                  However, this can not be guaranteed.
    utc_datetime_start
        Start time as utc datetime.
    utc_datetime_stop
        Stop time as utc datetime.
    utc_stop
        Unix time stamp of the end of the recording.

        .. note:: No timezone is assumed and client software is instructed to set the internal sensor clock to utc time.
                  However, this can not be guaranteed.

    firmware_version
        Version number of the firmware
    platform
        Platform of hearing aid
    app_version
        Version number of the recording app
    model_name
        Name of hearing aid model
    ruleset_version
        Version number of the ruleset
    custom_meta_data
        Custom meta data which was saved during saving.
    n_samples
        Number of samples recorded during the measurement

        .. note:: Number of samples is not determined during the init but later after data was loaded.

    """

    def __init__(self, **kwargs):
        """Initialize a header object.

        This will just put all values provided in kwargs as attributes onto the class instance.
        If one value has an unexpected name, a warning is raised, and the key is ignored.
        """
        for k, v in kwargs.items():
            if k in self._header_fields:
                setattr(self, k, v)
            else:
                # Should this be a error?
                warnings.warn(f"Unexpected Argument {k} for Header")

    @classmethod
    def from_dict_mat(cls, bin_array: np.ndarray) -> "Header":
        """Create a new Header instance from an array of bytes."""
        header_dict = cls.parse_header_dict_mat(bin_array)
        return cls(**header_dict)

    @classmethod
    def parse_header_dict_mat(cls, meta_info: dict) -> Dict[str, Union[str, int, float, bool, tuple]]:
        """Extract all values from a dict header."""
        header_dict = {}

        sensors = meta_info["activeSensors"]
        sensors = sensors.split(",")
        enabled_sensors = []
        for para, val in cls._SENSOR_FLAGS.items():
            sens_info = [x for x in sensors if val in x]
            assert len(sens_info) == 1
            if "enabled" in str(sens_info):
                enabled_sensors.append(para)
        header_dict["enabled_sensors"] = tuple(enabled_sensors)

        header_dict["sampling_rate_hz"] = np.float64(meta_info["fs"])

        header_dict["acc_range_g"] = float(2)

        header_dict["gyro_range_dps"] = float(1000)

        header_dict["sensor_position"] = "ha_" + meta_info["deviceSide"]

        header_dict["custom_meta_data"] = meta_info["description"]

        # Note: We ignore timezones and provide just the time info, which was stored in the sensor
        date = meta_info["date"].split("-")
        time = meta_info["time"].split(":")
        utc = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2]))
        header_dict["utc_stop"] = int(utc.timestamp())
        header_dict["utc_start"] = header_dict["utc_stop"] - int(meta_info["length"])

        header_dict["firmware_version"] = meta_info["deviceFwVersion"]

        header_dict["sensor_id"] = meta_info["deviceSerialNumber"]

        try:
            header_dict["imu_sensor_type"] = meta_info["IMUSensorType"]
        except KeyError:
            header_dict["imu_sensor_type"] = None

        header_dict["platform"] = meta_info["deviceFwVersion"][0:3]
        header_dict["ruleset_version"] = None
        header_dict["model_name"] = None
        header_dict["app_version"] = "No App Used"

        return header_dict

    @classmethod
    def from_list_txt(cls, info_list: list, stop_time: str, imu_sensor: str) -> "Header":
        """Create a new Header instance from an array of bytes."""
        header_dict = cls.parse_header_txt(info_list, stop_time, imu_sensor)
        return cls(**header_dict)

    @classmethod
    def parse_header_txt(  # noqa: MC0001
        cls, meta_info: list, stop_time: str, imu_sensor: str
    ) -> Dict[str, Union[str, int, float, bool, tuple]]:
        """Extract all values from a dict header."""
        header_dict = {}
        utc_start = datetime.datetime.strptime(meta_info[0], "%d-%m-%Y_%H-%M-%S")
        utc_stop = datetime.datetime.strptime(meta_info[0][0:10] + "_" + stop_time, "%d-%m-%Y_%H:%M:%S.%f")
        header_dict["utc_start"] = int(utc_start.timestamp())
        header_dict["utc_stop"] = int(utc_stop.timestamp())
        header_dict["imu_sensor_type"] = imu_sensor

        for p in meta_info[1::]:
            if "Gyroscope DPS" in p:
                header_dict["gyro_range_dps"] = int(p.split(": ")[1])
            elif "Accelerometer G" in p:
                header_dict["acc_range_g"] = int(p.split(": ")[1])
            elif "Data rate:" in p:
                header_dict["sampling_rate_hz"] = int(p.split(": ")[1])
            elif "Available sensors:" in p:
                avail = []
                if "Gyrosc" in p:
                    avail.append("gyro")
                if "Accelerometer" in p:
                    avail.append("acc")
                header_dict["enabled_sensors"] = tuple(avail)
            elif "Notes:" in p:
                header_dict["custom_meta_data"] = p.split(":")[1][1::]

        if "Application Version" in meta_info[2]:
            header_dict = add_header_infos_with_app_version(meta_info, header_dict)
        else:
            header_dict = add_header_infos_without_app_version(meta_info, header_dict)

        header_dict["platform"] = "D12"
        return header_dict


def add_header_infos_with_app_version(meta_info, header_dict):
    for p in meta_info[1::]:
        if "Application Version" in p:
            header_dict["app_version"] = p.split(": ")[1]
        elif "Serial Number" in p:
            entry, side1 = _split_line(p)
            header_dict["sensor_id"] = entry
        elif "Firmware Version" in p:
            entry, side2 = _split_line(p)
            header_dict["firmware_version"] = entry
        elif "RuleSet Version" in p:
            entry, side3 = _split_line(p)
            header_dict["ruleset_version"] = entry
        elif "Model Name" in p:
            entry, side4 = _split_line(p)
            header_dict["model_name"] = entry
    if not (side1 == side2 == side3 == side4):  # noqa
        raise ValueError("Not consistent left and right configuration.")
    header_dict["sensor_position"] = "ha_left" if side1 == "Left" else "ha_right"
    return header_dict


def _split_line(line):
    r = line.split(": ")
    if r[0][-4::] != "Left" or r[1][-5::] != "Right":
        raise ValueError("Wrong format of lines for left and right configuration")
    left = r[1].split(" Right")[0]
    right = r[2]
    if left == "None":
        return right, "Right"
    return left, "Left"


def add_header_infos_without_app_version(meta_info, header_dict):
    cnt_ha = 2
    for p in meta_info[1::]:
        if "Hearing Aid" in p:
            if p.split("Number: ")[1] == "None":
                cnt_ha = 1
                continue
            side = p.split(" ")[0]
            header_dict["sensor_position"] = "ha_left" if side == "Left" else "ha_right"
            header_dict["sensor_id"] = p.split("Number: ")[1]
    header_dict["firmware_version"] = None
    header_dict["platform"] = "D12"
    header_dict["ruleset_version"] = None
    header_dict["model_name"] = None
    header_dict["app_version"] = "0.0.0"
    if cnt_ha == 2:
        raise ValueError("Two hearing aids were used. Importer is currently only implementer for a single sensor.")
    return header_dict


class _ProxyHeader(_HeaderFields):
    """A proxy header used by session objects to get direct access to multiple headers.

    This allows to access attributes of multiple header instances without reimplementing all of its attributes.
    This is achieved by basically intercepting all getattribute calls and redirecting them to all header instances.

    This concept only allows read only access. However, usually their is no need to modify a header after creation.
    """

    _headers: Tuple[Header]

    def __init__(self, headers: Tuple[Header]):
        self._headers = headers

    def __getattribute__(self, name: str) -> Tuple[Any]:
        if name in ("_headers", "_all_header_fields", "_header_fields", "_ipython_display_"):
            return super().__getattribute__(name)
        if callable(getattr(self._headers[0], name)) is True:
            if name.startswith("__"):
                return super().__getattribute__(name)
            raise ValueError(
                f"_ProxyHeader only allows access to attributes of the info objects. {name} is a callable method."
            )

        return tuple(getattr(d, name) for d in self._headers)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_headers":
            return super().__setattr__(name, value)
        raise NotImplementedError("_ProxyHeader only allows readonly access to the info objects of a dataset")

    def __dir__(self):
        return itertools.chain(super().__dir__(), self._headers[0].__dir__())

    def _ipython_display_(self):
        """ """  # noqa: D419
        import pandas as pd  # noqa: import-outside-toplevel
        from IPython import display  # noqa: import-outside-toplevel

        header = {}
        for k in self._all_header_fields:
            try:
                header[k] = getattr(self, k, None)
            except ValueError:
                continue
        display.display(pd.DataFrame(header, index=self.sensor_id).T)

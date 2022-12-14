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

    version_firmware: str

    sensor_id: str

    custom_meta_data: Tuple[float]

    n_samples: int

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
        """ """  # pydocstyle: noqa
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

    version_firmware
        Version number of the firmware
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
    def from_dict(cls, bin_array: np.ndarray) -> "Header":
        """Create a new Header instance from an array of bytes."""
        header_dict = cls.parse_header_dict(bin_array)
        return cls(**header_dict)

    @classmethod
    def parse_header_dict(cls, meta_info: dict) -> Dict[str, Union[str, int, float, bool, tuple]]:
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

        header_dict["version_firmware"] = meta_info["deviceFwVersion"]

        header_dict["sensor_id"] = meta_info["deviceSerialNumber"]

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

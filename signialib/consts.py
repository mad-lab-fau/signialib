"""Basic constants/names used throughout the lib."""

#: Byte information of each sensor_type in one sample.
#:
#: Format: Overall number of bytes, number of channels, datatype


#: Default legends for all sensors
SENSOR_LEGENDS = {
    "acc": tuple("acc_" + x for x in "xyz"),
    "gyro": tuple("gyr_" + x for x in "xyz"),
}

SENSOR_MAPPINGS = {"acc": ("x", "y", "z"), "gyro": ("hiGyrX", "hiGyrY", "hiGyrZ")}

#: The value of gravity
GRAV = 9.81

#: Default units for all sensors
SENSOR_UNITS = {"acc": "m/s^2", "gyro": "deg/s"}

#: simple unit names
SIMPLE_UNITS = {"m/s^2": "ms2", "deg/s": "dps"}

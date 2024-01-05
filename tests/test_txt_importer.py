from pathlib import Path

from signialib import Session

HERE = Path(__file__).parent


def test_read_in_bma400():
    Session.from_file_path(HERE.joinpath("test_data", "txt_file", "07-03-2023_18-29-52_test_BMA400.txt"))


def test_read_in_bmi270():
    Session.from_file_path(HERE.joinpath("test_data", "txt_file", "08-03-2023_11-12-51_testWithLabels.txt"))


def test_txt_read_in_with_starting_label_bma400():
    Session.from_file_path(HERE.joinpath("test_data", "txt_file", "03-04-2023_11-03-21_BMA400_WalkingLabel.txt"))


def test_labels_readout():
    session = Session.from_file_path(HERE.joinpath("test_data", "txt_file", "08-03-2023_11-12-51_testWithLabels.txt"))
    assert session.labels is not None
    assert session.labels.shape[0] == 5


def test_txt_with_starting_label_bmi270():
    session = Session.from_file_path(
        HERE.joinpath("test_data", "txt_file", "08-03-2023_11-12-51_testWithLabels_2.0.txt")
    )
    assert session.labels is not None
    assert session.labels.shape[0] == 6


def test_header_with_old_app_version():
    session = Session.from_file_path(
        HERE.joinpath("test_data", "txt_file", "08-03-2023_11-12-51_testWithLabels_2.0.txt")
    )
    assert session.info.app_version[0] == "0.0.0"
    assert session.info.firmware_version[0] == None
    assert session.info.model_name[0] == None
    assert session.info.platform[0] == "D12"
    assert session.info.ruleset_version[0] == None
    assert session.info.imu_sensor_type[0] == "BMI270"
    assert session.info.sampling_rate_hz[0] == 50
    assert session.info.sensor_id[0] == "TEX1234"


def test_header_including_app_version():
    session = Session.from_file_path(
        HERE.joinpath("test_data", "txt_file", "04-01-2024_noWalking_includingAppVersionAndDummyFirmversion.txt")
    )
    assert session.info.app_version[0] == "2.1.0.222"
    assert session.info.firmware_version[0] == "99.31.310.201"
    assert session.info.model_name[0] == "Pure 312 1AX"
    assert session.info.platform[0] == "D12"
    assert session.info.ruleset_version[0] == "44.81.243.7"

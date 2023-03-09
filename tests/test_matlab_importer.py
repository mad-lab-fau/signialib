from pathlib import Path

import numpy as np
import pandas as pd

from signialib import Session

HERE = Path(__file__).parent


def test_matlab_from_folder_path():
    session = Session.from_folder_path(HERE.joinpath("test_data", "mat_files"))
    assert len(session.datasets) == 2


def test_matlab_from_file_paths():
    base = HERE.joinpath("test_data", "mat_files")
    session = Session.from_file_paths([base.joinpath("data_left.mat"), base.joinpath("data_right.mat")])
    assert len(session.datasets) == 2


def test_matlab_from_file_path():
    session = Session.from_file_path(HERE.joinpath("test_data", "mat_files", "data_right.mat"))
    assert len(session.datasets) == 1


def test_matlab_acc_and_gyro():
    session = Session.from_file_path(HERE.joinpath("test_data", "mat_files", "data_right.mat"))
    assert session.datasets[0].acc and session.datasets[0].gyro


def test_matlab_header():
    session = Session.from_file_path(HERE.joinpath("test_data", "mat_files", "data_left.mat"))
    assert session.info.sampling_rate_hz[0] == 200.0
    assert session.info.sensor_position[0] == "ha_left"
    assert session.info.sensor_id[0] == "002"
    assert session.info.gyro_range_dps[0] == 1000.0
    assert session.info.has_position_info[0]
    assert session.info.version_firmware[0] == "DummyDeviceFirmwareXX1"
    assert session.info.utc_start[0] == 1622537266
    assert session.info.utc_stop[0] == 1622537496
    assert len(session.info._all_header_fields) == 17


def test_matlab_data_loader():
    data_csv = pd.read_csv(HERE.joinpath("test_data", "mat_left_data.csv"), index_col=0)
    session = Session.from_file_path(HERE.joinpath("test_data", "mat_files", "data_left.mat"))
    data = session.get_dataset_by_position("ha_left").data_as_df()
    pd.testing.assert_frame_equal(data, data_csv)


def test_matlab_index():
    session = Session.from_file_path(HERE.joinpath("test_data", "mat_files", "data_left.mat"))
    data = session.get_dataset_by_position("ha_left").data_as_df()
    assert (data.index.to_numpy() == np.arange(0, len(data))).all()


def test_matlab_no_labels():
    session = Session.from_file_path(HERE.joinpath("test_data", "mat_files", "data_left.mat"))
    assert session.labels is None

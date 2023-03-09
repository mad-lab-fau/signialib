from pathlib import Path

import pandas as pd

from signialib import Session

HERE = Path(__file__).parent


def test_alignment():
    session = Session.from_folder_path(HERE.joinpath("test_data", "mat_files"))
    session = session.align_to_syncregion()
    assert session.datasets[0].acc.data.shape[0] == session.datasets[1].gyro.data.shape[0]
    assert session.counter[0].shape == session.counter[1].shape
    assert session.size[0] == session.size[1]
    assert len(session.acc[0]) == len(session.counter[0])
    assert len(session.acc[1]) == len(session.counter[1])


def test_resampling():
    session = Session.from_folder_path(HERE.joinpath("test_data", "mat_files"))
    session = session.resample(50)
    data = session.get_dataset_by_position("ha_left").data_as_df()
    pd.testing.assert_frame_equal(
        data, pd.read_csv(HERE.joinpath("test_data", "mat_left_data_resampled50.csv"), index_col=0)
    )
    assert session.info.sampling_rate_hz[0] == 50.0
    assert len(session.acc[0]) == len(session.counter[0])
    assert session.size == (len(session.acc[0]), len(session.gyro[1]))


def test_skip_calibration():
    session = Session.from_folder_path(HERE.joinpath("test_data", "mat_files"))
    session = session.align_calib_resample(resample_rate_hz=50, skip_calibration=True)
    assert session.size == (len(session.acc[0]), len(session.gyro[1]))
    assert len(session.acc[0]) == len(session.counter[0])


# todo: calibrate_imu, align_calib_resample

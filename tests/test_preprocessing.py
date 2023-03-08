from pathlib import Path
import pandas as pd

from signialib import Session

HERE = Path(__file__).parent


def test_alignment():
    session = Session.from_folder_path(HERE.joinpath("test_data", "mat_files"))
    session = session.align_to_syncregion()
    assert(session.datasets[0].acc.data.shape[0] == session.datasets[1].acc.data.shape[0])


def test_resampling():
    session = Session.from_folder_path(HERE.joinpath("test_data", "mat_files"))
    session = session.resample(50)
    data = session.get_dataset_by_position("ha_left").data_as_df()
    pd.testing.assert_frame_equal(data, pd.read_csv(HERE.joinpath("test_data", "mat_left_data_resampled50.csv"), index_col=0))
    assert(session.info.sampling_rate_hz[0] == 50.0)
    assert(len(session.acc[0]) == len(session.counter[0]))
    assert(session.size == (len(session.acc[0]), len(session.acc[1])))


# todo: calibrate_imu, align_calib_resample

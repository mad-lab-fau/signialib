from pathlib import Path

from signialib import Session

HERE = Path(__file__).parent


def test_txt_from_file_path_bma400():
    Session.from_file_path(HERE.joinpath("test_data", "txt_file", "07-03-2023_18-29-52_test_BMA400.txt"))


def test_txt_from_file_path_bmi270():
    Session.from_file_path(HERE.joinpath("test_data", "txt_file", "08-03-2023_11-12-51_testWithLabels.txt"))


def test_txt_from_file_path_bma400_withstartinglabel():
    Session.from_file_path(HERE.joinpath("test_data", "txt_file", "03-04-2023_11-03-21_BMA400_WalkingLabel.txt"))


def test_txt_from_file_bmi270_withlabels():
    session = Session.from_file_path(HERE.joinpath("test_data", "txt_file", "08-03-2023_11-12-51_testWithLabels.txt"))
    assert session.labels is not None
    assert session.labels.shape[0] == 5


def test_txt_from_file_bmi270_withstartinglabel():
    session = Session.from_file_path(
        HERE.joinpath("test_data", "txt_file", "08-03-2023_11-12-51_testWithLabels_2.0.txt")
    )
    assert session.labels is not None
    assert session.labels.shape[0] == 6

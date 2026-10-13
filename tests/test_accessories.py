import os
import time
import pytest
import pandas as pd

import accessories


def test_returns_only_matching_file(tmp_path):
    f = tmp_path / "model.zip"
    f.touch()

    result = accessories.find_latest_file(tmp_path, pattern="*.zip")

    assert result == f


def test_returns_most_recently_modified_file(tmp_path):
    older = tmp_path / "a.zip"
    newer = tmp_path / "b.zip"
    older.touch()
    time.sleep(0.01)  # ensure distinct mtimes
    newer.touch()

    result = accessories.find_latest_file(tmp_path, pattern="*.zip")

    assert result == newer


def test_pattern_filters_out_non_matching_files(tmp_path):
    (tmp_path / "notes.txt").touch()
    zip_file = tmp_path / "model.zip"
    zip_file.touch()

    result = accessories.find_latest_file(tmp_path, pattern="*.zip")

    assert result == zip_file


def test_raises_file_not_found_error_when_no_matches(tmp_path):
    (tmp_path / "notes.txt").touch()  # exists, but doesn't match pattern

    with pytest.raises(FileNotFoundError, match=r"No files matching '\*\.zip'"):
        accessories.find_latest_file(tmp_path, pattern="*.zip")


def test_raises_file_not_found_error_when_folder_is_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        accessories.find_latest_file(tmp_path, pattern="*")


def test_raises_file_not_found_error_when_folder_does_not_exist(tmp_path):
    missing = tmp_path / "does_not_exist"

    # Path.glob on a nonexistent dir yields nothing rather than raising,
    # so this should hit the same empty-match path as the other cases.
    with pytest.raises(FileNotFoundError):
        accessories.find_latest_file(missing, pattern="*")


def make_history(desired, actual, drums):
    """drums: list of 8-element lists, one per timestep."""
    n = len(desired)
    data = {"desired_power": desired, "actual_power": actual}
    for i in range(8):
        data[f"drum_{i+1}"] = [row[i] for row in drums]
    return pd.DataFrame(data)


def test_zero_error_zero_control_effort():
    desired = [50, 50, 50]
    actual = [50, 50, 50]
    drums = [[0] * 8, [0] * 8, [0] * 8]  # constant drum angles
    history = make_history(desired, actual, drums)

    mae, cum_err, ce, mce = accessories.metrics(history)

    assert mae == 0
    assert cum_err == 0
    assert ce == 0
    assert mce == 0


def test_known_error_values():
    desired = [50, 60, 40]
    actual = [45, 65, 40]  # errors: 5, -5, 0
    drums = [[0] * 8, [0] * 8, [0] * 8]
    history = make_history(desired, actual, drums)

    mae, cum_err, ce, mce = accessories.metrics(history)

    assert mae == pytest.approx((5 + 5 + 0) / 3)
    assert cum_err == pytest.approx(10)


def test_known_control_effort():
    desired = [50, 50, 50]
    actual = [50, 50, 50]
    drums = [
        [0] * 8,
        [1] * 8,  # +1 per drum
        [0] * 8,  # -1 per drum
    ]
    history = make_history(desired, actual, drums)

    mae, cum_err, ce, mce = accessories.metrics(history)

    # 2 diffs * 8 drums * abs(1) = 16
    assert ce == pytest.approx(16)
    assert mce == pytest.approx(16 / 3)


def test_single_row_no_control_effort():
    # np.diff on 1 row returns empty array
    history = make_history([50], [50], [[0] * 8])
    mae, cum_err, ce, mce = accessories.metrics(history)
    assert ce == 0
    assert mce == 0


def test_empty_history_raises_value_error():
    history = make_history([], [], [])
    with pytest.raises(ValueError):
        accessories.metrics(history)

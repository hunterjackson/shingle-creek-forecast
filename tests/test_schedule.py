import pandas as pd
import pytest

from home_display.shingle import schedule


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2023-01-15", 55.0),
        ("2023-03-15", 55.0),
        ("2023-05-31", 52.0),
        ("2023-06-01", 53.5),
        ("2023-08-15", 53.5),
        ("2023-09-01", 53.5),
        ("2023-10-01", 54.0),
        ("2023-11-01", 55.0),
        ("2023-12-31", 55.0),
    ],
)
def test_toho_target_key_dates(date, expected):
    assert schedule.toho_target(pd.DatetimeIndex([date]))[0] == pytest.approx(expected, abs=0.02)


def test_toho_target_is_linear_during_spring_recession():
    # Mar 15 (55.0) -> May 31 (52.0): 77 days; Apr 23 is 39 days in
    v = schedule.toho_target(pd.DatetimeIndex(["2023-04-23"]))[0]
    assert v == pytest.approx(55.0 - 3.0 * 39 / 77, abs=0.02)


def test_toho_target_series_aligned_to_index():
    idx = pd.date_range("2020-01-01", periods=400, name="date")

    s = schedule.toho_target_series(idx)

    assert s.index.equals(idx)
    assert s.name == "toho_target"
    assert s.between(52.0, 55.0).all()

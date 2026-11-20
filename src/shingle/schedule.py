"""Lake Tohopekaliga regulation schedule (target stage, ft NGVD29).

Read from SFWMD (2005) Kissimmee Basin water-availability memo, fig. 3
"Current Regulation Schedule" (+/-0.01 ft, +/-1 day). Actual operations
deviate (e.g. fish & wildlife recessions), so the model uses both the
target and the observed departure from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (month, day, stage) breakpoints, linearly interpolated within a year
_BREAKS = [
    (1, 1, 55.0),
    (3, 15, 55.0),
    (5, 31, 52.0),
    (6, 1, 53.5),
    (9, 1, 53.5),
    (10, 1, 54.0),
    (11, 1, 55.0),
    (12, 31, 55.0),
]


def toho_target(index: pd.DatetimeIndex) -> np.ndarray:
    out = np.empty(len(index))
    for year in np.unique(index.year):
        mask = index.year == year
        xs = [pd.Timestamp(year=year, month=m, day=d).day_of_year for m, d, _ in _BREAKS]
        ys = [s for *_, s in _BREAKS]
        out[mask] = np.interp(index[mask].day_of_year, xs, ys)
    return out


def toho_target_series(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(toho_target(index), index=index, name="toho_target")

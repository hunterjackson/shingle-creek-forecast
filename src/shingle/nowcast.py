"""Today's figure updated from the hours of the target day already observed (docs/project_review.md, idea 2).

The day-1 forecast is issued at ~13:30Z on t+1, when 8 hours (05Z..12Z = midnight..8 am EST) of the target day are
in; the job then reruns every 3 h. With n hours seen, the day-1 mean is estimated as

    (n * mean seen + (24 - n) * (latest + a_n * (implied - latest))) / 24

where implied = (24 * p50 - 8 * mean of 05..12Z) / 16 is the issued forecast's level for the hours it had not seen.
a_n and the band shrink factor (ratio of the 10-90 % residual spread to the frozen forecast's) are fitted at
training time on the forecast-mode out-of-fold day-1 forecasts. The issued band is scaled by the shrink factor,
so it stays rain-aware. Out of fold: -12 % at 2:30 pm EST, -42 % at 5:30 pm, -74 % at 8:30 pm
(experiments/improve/i2_report.txt). The logged (scored) forecast stays the issued one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIRST, ISSUE_HOURS = 5, 8  # target day starts 05Z (midnight EST); 8 hours are in at issue
NS = tuple(range(ISSUE_HOURS + 1, 24))


def hours_table(hourly: pd.Series) -> pd.DataFrame:
    """Columns k = 5..28: the hourly value labelled k hours after 00Z of t+1, rows labelled by issue day t."""
    s = hourly.dropna()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s.resample("1h").mean()
    days = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), name="date")
    return pd.DataFrame({k: pd.Series(s.reindex(days + pd.Timedelta(hours=k)).to_numpy(), index=days - pd.Timedelta(days=1))
                         for k in range(FIRST, FIRST + 24)})


def _parts(seen: np.ndarray):
    seen = np.asarray(seen, float)
    vals = seen[~np.isnan(seen)]
    return len(seen), np.nanmean(seen[:ISSUE_HOURS]), vals.mean(), vals[-1]


def point(nc: dict, p50: float, seen) -> float:
    n, s8, sofar, last = _parts(seen)
    a = float(np.interp(n, nc["n"], nc["a"]))
    implied = (24 * p50 - ISSUE_HOURS * s8) / (24 - ISSUE_HOURS)
    return (n * sofar + (24 - n) * (last + a * (implied - last))) / 24


def apply(nc: dict, p10: float, p50: float, p90: float, seen) -> tuple[float, float, float] | None:
    """Updated (p10, p50, p90) for day 1 given the hourly values from 05Z on (NaN for gaps); None if nothing new."""
    if len(seen) <= ISSUE_HOURS:
        return None
    est = point(nc, p50, seen)
    k = float(np.interp(len(seen), nc["n"], nc["shrink"]))
    return est + k * (p10 - p50), est, est + k * (p90 - p50)


def fit(p50: pd.Series, hourly: pd.Series, stage: pd.Series, ns=NS) -> dict:
    """a_n and band shrink per hours-seen n, from frozen day-1 forecasts p50 (issue-day index) and the hourly record."""
    G = hours_table(hourly)
    y = stage.shift(-1)
    out = {"n": [], "a": [], "shrink": []}
    for n in ns:
        hrs = list(range(FIRST, FIRST + n))
        ix = p50.dropna().index.intersection(G.index)
        g = G.loc[ix, hrs]
        ok = y.reindex(ix).notna() & g.notna().sum(axis=1).ge(n - 2) & g.iloc[:, :ISSUE_HOURS].notna().any(axis=1)
        ix = ix[ok.to_numpy()]
        if len(ix) < 50:
            continue
        g, yy, p = G.loc[ix, hrs], y[ix], p50[ix]
        s8, sofar, last = g.iloc[:, :ISSUE_HOURS].mean(axis=1), g.mean(axis=1), g.ffill(axis=1).iloc[:, -1]
        x = (24 - n) * ((24 * p - ISSUE_HOURS * s8) / (24 - ISSUE_HOURS) - last)
        lhs = 24 * yy - n * sofar - (24 - n) * last
        a = float((x * lhs).sum() / (x ** 2).sum())
        r_new = yy - (n * sofar + (24 - n) * last + a * x) / 24
        r_old = yy - p
        spread = lambda r: float(r.quantile(0.9) - r.quantile(0.1))
        out["n"].append(n)
        out["a"].append(a)
        out["shrink"].append(min(1.0, spread(r_new) / spread(r_old)))
    return out


def seen_hours(hourly: pd.Series, issue_day: pd.Timestamp, now: pd.Timestamp) -> np.ndarray:
    """Hourly values of the target day (05Z of issue_day+1 on) for the hours complete at `now` (NaN where missing)."""
    s = hourly.copy()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s.resample("1h").mean()
    start = pd.Timestamp(issue_day).normalize() + pd.Timedelta(days=1, hours=FIRST)
    now = pd.Timestamp(now).tz_convert("UTC").tz_localize(None)
    end = min(now.floor("1h") - pd.Timedelta(hours=1), start + pd.Timedelta(hours=23))
    if end < start:
        return np.array([])
    return s.reindex(pd.date_range(start, end, freq="1h")).to_numpy()

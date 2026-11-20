"""Open-Meteo forecast client (live 16-day and archived historical forecasts).

Requests are made at each gridMET cell centre; Open-Meteo snaps each to its
model's nearest grid point, which is how coarse forecast grids are mapped onto
our per-cell rain features.
"""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TIMEZONE = "America/New_York"

# Open-Meteo daily variable -> our short name (matches gridmet.VARS keys)
DAILY = {"precipitation_sum": "pr", "et0_fao_evapotranspiration": "pet"}


def parse(payload: dict | list, cell_ids: list[str]) -> dict[str, pd.DataFrame]:
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(cell_ids):
        raise ValueError(f"{len(locations)} locations returned for {len(cell_ids)} cells")

    index = pd.DatetimeIndex(pd.to_datetime(locations[0]["daily"]["time"]), name="date")
    out = {}
    for om_var, short in DAILY.items():
        data = {
            cid: np.array([np.nan if v is None else v for v in loc["daily"][om_var]], dtype=float)
            for cid, loc in zip(cell_ids, locations)
        }
        out[short] = pd.DataFrame(data, index=index)
    return out


def _base_params(cells: pd.DataFrame) -> dict:
    return {
        "latitude": ",".join(f"{v:g}" for v in cells["lat"]),
        "longitude": ",".join(f"{v:g}" for v in cells["lon"]),
        "daily": ",".join(DAILY),
        "timezone": TIMEZONE,
    }


def _get(url: str, params: dict, client: httpx.Client | None) -> dict | list:
    client = client or httpx.Client(timeout=120)
    resp = client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def fetch_forecast(cells: pd.DataFrame, days: int = 16, client: httpx.Client | None = None) -> dict[str, pd.DataFrame]:
    params = _base_params(cells) | {"forecast_days": days}
    return parse(_get(FORECAST_URL, params, client), list(cells.index))


def fetch_historical_forecast(
    cells: pd.DataFrame, start: str, end: str, client: httpx.Client | None = None
) -> dict[str, pd.DataFrame]:
    params = _base_params(cells) | {"start_date": start, "end_date": end}
    return parse(_get(HISTORICAL_FORECAST_URL, params, client), list(cells.index))


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _lead_var(lead: int) -> str:
    return "precipitation" if lead == 0 else f"precipitation_previous_day{lead}"


def parse_previous_runs(payload: dict | list, cell_ids: list[str], leads=(0, 1, 2, 3, 4, 5, 6, 7)) -> dict[int, pd.DataFrame]:
    """Daily (local-day) rain sums of the forecast issued `lead` days before, per cell.

    lead 0 is Open-Meteo's analysis-like 'precipitation' series. Available ~2024+.
    A day whose 24 hourly values are all null is NaN.
    """
    locations = payload if isinstance(payload, list) else [payload]
    if len(locations) != len(cell_ids):
        raise ValueError(f"{len(locations)} locations returned for {len(cell_ids)} cells")
    times = pd.to_datetime(locations[0]["hourly"]["time"])
    day = pd.DatetimeIndex(times.normalize(), name="date")
    out = {}
    for lead in leads:
        cols = {}
        for cid, loc in zip(cell_ids, locations):
            vals = pd.Series([np.nan if v is None else v for v in loc["hourly"][_lead_var(lead)]], index=day, dtype=float)
            cols[cid] = vals.groupby(level=0).sum(min_count=1)
        out[lead] = pd.DataFrame(cols)
        out[lead].index.name = "date"
    return out


def fetch_previous_runs(
    cells: pd.DataFrame, start: str, end: str, leads=(0, 1, 2, 3, 4, 5, 6, 7), client: httpx.Client | None = None
) -> dict[int, pd.DataFrame]:
    params = {
        "latitude": ",".join(f"{v:g}" for v in cells["lat"]),
        "longitude": ",".join(f"{v:g}" for v in cells["lon"]),
        "hourly": ",".join(_lead_var(l) for l in leads),
        "start_date": start,
        "end_date": end,
        "timezone": TIMEZONE,
    }
    return parse_previous_runs(_get(PREVIOUS_RUNS_URL, params, client), list(cells.index), leads)

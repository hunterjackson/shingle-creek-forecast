"""Operational glue: update inputs, train an artifact, issue the morning forecast JSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from home_display.shingle import basin, config, fcst, fetch, forecast_log, grid, live, output, train
from home_display.shingle.sources import gefs, gridmet

ISSUE_READY_UTC = pd.Timedelta(hours=13)  # 12Z readings + MRMS 12Z QPE (~1 h latency) are in by ~13Z
MODEL_PATH = config.MODELS_DIR / "shingle.pkl"
OUT_PATH = config.OUT_DIR / "shingle.json"
LOG_PATH = config.OUT_DIR / "forecast_log.csv"


def latest_issue_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Latest issue day t whose forecast (made the morning of t+1, after ~13Z) can be run at `now`."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    return (now - ISSUE_READY_UTC).tz_localize(None).normalize() - pd.Timedelta(days=1)


def gefs_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = gefs.daily_leads(fetch.load_gefs(data_dir, "mean"))
    return raw, gridmet.cells_from_columns(raw[1].columns)


def forecast_rain(art: train.Artifact, cells: pd.DataFrame, raw: pd.DataFrame, src: pd.DataFrame, issue_day) -> pd.Series:
    """Bias-scaled basin-areal GEFS rain (mm) for each forecast date."""
    day = pd.Timestamp(issue_day)
    w = grid.bilinear_weights(cells[["lat", "lon"]], src)
    out = {}
    for k in sorted(raw.columns.get_level_values(0).unique()):
        if day not in raw.index:
            out[day + pd.Timedelta(days=int(k))] = np.nan
            continue
        mapped = grid.apply_weights(raw[k].loc[[day]], w)
        out[day + pd.Timedelta(days=int(k))] = float(fcst._areal(mapped, art.gr4j.weights).iloc[0]) * art.scale.get(int(k), 1.0)
    return pd.Series(out)


def _jsonable(d: dict) -> dict:
    def conv(v):
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v

    return {k: conv(v) for k, v in d.items()}


def forecast_doc(art: train.Artifact, hist: dict, raw: pd.DataFrame, src: pd.DataFrame, radar_cells, issue_day, out_path,
                 current: tuple[float, pd.Timestamp], radar_scale: float | None = None) -> dict:
    day = pd.Timestamp(issue_day).normalize()
    scale = radar_scale if radar_scale is not None else art.meta.get("radar_scale", 1.0)
    h = live.prepare_history(hist, day, radar_cells, scale, art.pet_clim)
    fc = train.predict(art, h, raw, src, day)
    rain = forecast_rain(art, hist["cells"], raw, src, day)
    meta = {"train_end": art.train_end, "live_fill": _jsonable(h["live_fill"]), "gefs_issued": bool(day in raw.index)}
    return output.write_json(out_path, fc, current_stage=current[0], current_time=current[1], issue_day=day, rain_mm=rain, meta=meta)


def radar_cell_means(data_dir: Path, cells: pd.DataFrame) -> pd.DataFrame | None:
    try:
        px = fetch.load_mrms(data_dir, hour=12)
    except ValueError:  # no files
        return None
    return grid.cell_means(px, cells, half_width=basin.GRIDMET_RES / 2)


def fit_radar_scale(gm: pd.DataFrame, radar: pd.DataFrame, cell: str = config.RAIN_CELL) -> float:
    pair = pd.concat([gm[cell], radar[cell]], axis=1, keys=["gm", "r"]).dropna()
    return float(pair["gm"].sum() / pair["r"].sum()) if len(pair) > 365 and pair["r"].sum() > 0 else 1.0


def update_inputs(data_dir: Path, today: pd.Timestamp | None = None) -> None:
    today = (today or pd.Timestamp.now(tz="UTC").tz_localize(None)).normalize()
    fetch.fetch_history(data_dir)
    fetch.fetch_iv_history(data_dir)
    fetch.fetch_mrms_history(data_dir, start=str((today - pd.Timedelta(days=14)).date()), hour=12)
    fetch.fetch_gefs_history(data_dir, start=str((today - pd.Timedelta(days=5)).date()), kinds=("mean",))


def train_from_disk(data_dir: Path, model_path: Path, train_end: str) -> train.Artifact:
    hist = fetch.load_history(data_dir)
    raw, src = gefs_raw(data_dir)
    art = train.train(hist, raw, src, train_end=train_end)
    radar = radar_cell_means(data_dir, hist["cells"])
    if radar is not None:
        art.meta["radar_scale"] = fit_radar_scale(hist["pr"].loc[:train_end], radar.loc[:train_end])
    train.save(art, model_path)
    return art


def predict_from_disk(data_dir: Path, model_path: Path, out_path: Path, update: bool = True, now=None, log_path: Path = LOG_PATH) -> dict:
    if update:
        update_inputs(data_dir)
    art = train.load(model_path)
    hist = fetch.load_history(data_dir)
    raw, src = gefs_raw(data_dir)
    radar = radar_cell_means(data_dir, hist["cells"])
    iv = hist["iv"]["gauge"]
    last = iv["stage_ft"].dropna()
    current = (float(last.iloc[-1]), last.index[-1]) if len(last) else (float("nan"), pd.Timestamp.now(tz="UTC"))
    issue = latest_issue_day(now)
    doc = forecast_doc(art, hist, raw, src, radar, issue, out_path, current)
    toho = hist["lakes"]["toho_stage"].dropna()
    nwm_h1 = nwm_stage_h1(art.model.b.rating, issue, float(toho.iloc[-1]) if len(toho) else float("nan"))
    forecast_log.append(log_path, doc, nwm_h1)
    return doc


def score_from_disk(data_dir: Path, log_path: Path = LOG_PATH) -> pd.DataFrame:
    hist = fetch.load_history(data_dir)
    return forecast_log.score(log_path, hist["gauge"]["stage_ft"])


def nwm_stage_h1(rating, issue_day, toho: float, fetch=None) -> float | None:
    """NWM short-range mean flow for local day t+1 -> stage through our rating (None if unavailable)."""
    from home_display.shingle.sources import nwm

    day1 = pd.Timestamp(issue_day).normalize() + pd.Timedelta(days=1)
    try:
        df = fetch() if fetch else nwm.fetch_streamflow(config.NWM_REACH, "short_range")
        q = nwm.daily_mean(df)
    except Exception:  # NWPS outages must not break the morning forecast
        return None
    if day1 not in q.index or pd.isna(toho):
        return None
    return float(rating.predict_stage([float(q.loc[day1])], [float(toho)])[0])

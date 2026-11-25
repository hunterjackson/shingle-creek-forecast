"""Fit the selected model and everything prediction needs into one artifact; predict from it.

train(): GR4J calibrated on the training period -> simulated states as features;
spec.build_model() fit on observed rain; per-lead GEFS scale fitted on the GEFS
era inside training; conformal bands calibrated on *forecast-mode* predictions
(issued GEFS rain), optionally out-of-fold by year so they are honest.

With oof_years, horizons >= LEARNED_BANDS_FROM_H get learned-width bands (shingle.bands): the width model is fit on
out-of-fold perfect-rain residuals of the training years before fcst_start (blocks of scale_block_years), the bins
on the forecast-mode OOF residuals. Day 1 keeps |p50 - stage| bins (learned bands lowered its flood coverage).

Days 1..WET_BANDS_MAX_H then wrap that band in a bands.WetBand: on days with >= bands.WET_MM of issued basin rain
over t+1..t+h, conformal bins on the learned scale calibrated on such forecast-mode days only (docs/divergence_test_plan.md T2).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from shingle import bands, config, dataset, fcst, gr4j_driver, live, spec
from shingle.evaluate import BinnedConformal
from shingle.models.base import HORIZONS, targets

SCALED_FROM_H = 4
LEARNED_BANDS_FROM_H = 2
WET_BANDS_MAX_H = 3


@dataclass
class Artifact:
    model: object
    gr4j: gr4j_driver.GR4JDriver
    scale: dict[int, float]
    pet_clim: pd.DataFrame
    horizons: tuple
    train_end: str
    fcst_start: str
    meta: dict = field(default_factory=dict)


def _with_gr4j(hist: dict, drv) -> dict:
    data = dataset.build_data(hist)
    data["gr4j"] = drv.simulate(data)
    return data


def _forecast_data(data: dict, hist: dict, raw: pd.DataFrame, src: pd.DataFrame, drv, scale: dict, horizons) -> dict:
    tgt = hist["cells"][["lat", "lon"]]
    return dict(data, fcst=fcst.build_mixed(raw, src, tgt, drv, data, scale, SCALED_FROM_H, horizons))


def _set_bands(model, preds: pd.DataFrame, data: dict, index: pd.DatetimeIndex) -> None:
    y = targets(data["stage"], model.horizons).loc[index]
    model.bands = {}
    for h in model.horizons:
        p50 = preds.loc[index, f"h{h}"]
        model.bands[h] = BinnedConformal.fit(y[f"h{h}"] - p50, model.band_scale(data, index, p50, h), model.n_bins, model.coverage)


def perfect_oof(data: dict, tr: pd.DatetimeIndex, held_index: pd.DatetimeIndex, horizons, block_years: int = 3, gap_days: int = 7) -> pd.DataFrame:
    """Out-of-fold point predictions (observed future rain) for held_index, in blocks of calendar years; each block's
    model is fit on the rest of tr minus the gap_days before the block."""
    years = sorted(set(held_index.year))
    parts = []
    for i in range(0, len(years), block_years):
        blk = years[i : i + block_years]
        held = held_index[held_index.year.isin(blk)]
        start = pd.Timestamp(f"{blk[0]}-01-01")
        fit_ix = tr[~tr.year.isin(blk) & ~((tr < start) & (tr >= start - pd.Timedelta(days=gap_days)))]
        m = spec.build_model(horizons)
        m.a.fit(data, fit_ix)
        m.b.fit(data, fit_ix)
        parts.append(m.point(data, held))
    return pd.concat(parts).sort_index()


def _set_learned_bands(model, data, tr, pre, d_fc, fc_idx, preds_fc, horizons, block_years, gap_days, learned) -> dict:
    """Fit |residual| scale models for `horizons`; horizons in `learned` get LearnedBand bands. Returns the scale models."""
    po = perfect_oof(data, tr, pre, horizons, block_years, gap_days)
    y_pre = targets(data["stage"], horizons).loc[po.index]
    y_fc = targets(d_fc["stage"], horizons).loc[fc_idx]
    sms = {}
    for h in horizons:
        sms[h] = bands.LearnedBand.fit_scale(data, po.index, po[f"h{h}"], y_pre[f"h{h}"] - po[f"h{h}"], h)
        if h in learned:
            p50 = preds_fc.loc[fc_idx, f"h{h}"]
            model.bands[h] = bands.LearnedBand.fit(sms[h], d_fc, fc_idx, p50, y_fc[f"h{h}"] - p50, h, model.coverage)
    return sms


def _set_wet_bands(model, sms: dict, d_fc, fc_idx, preds_fc, weights: dict) -> None:
    y_fc = targets(d_fc["stage"], model.horizons).loc[fc_idx]
    for h, sm in sms.items():
        if h > WET_BANDS_MAX_H:
            continue
        p50 = preds_fc.loc[fc_idx, f"h{h}"]
        r = y_fc[f"h{h}"] - p50
        X = bands.scale_features(d_fc, fc_idx, p50, h)
        scale = pd.Series(sm.predict(X[sm.feature_name_]), index=fc_idx)
        wet = bands.forecast_rain_sum(d_fc, fc_idx, h, weights).fillna(0.0) >= bands.WET_MM
        ok = wet & r.notna() & scale.notna()
        if ok.sum() < bands.WET_MIN_COUNT:
            continue
        model.bands[h] = bands.WetBand.fit(r[ok], scale[ok], model.coverage).attach(model.bands[h], sm, h, weights)


def train(
    hist: dict, raw: pd.DataFrame, src: pd.DataFrame, train_end: str, fcst_start: str = "2020-10-01",
    horizons=HORIZONS, gr4j_maxiter: int = 100, oof_years: bool = True, gap_days: int = 7, scale_block_years: int = 3,
) -> Artifact:
    horizons = tuple(horizons)
    base_data = dataset.build_data(hist)
    idx = base_data["stage"].dropna().index
    tr = idx[idx <= pd.Timestamp(train_end) - pd.Timedelta(days=gap_days)]

    weights = hist["cells"]["basin_frac"][hist["cells"]["basin_frac"] > 0].to_dict()
    drv = gr4j_driver.GR4JDriver(weights, maxiter=gr4j_maxiter).fit(base_data, tr)
    data = _with_gr4j(hist, drv)
    model = spec.build_model(horizons).fit(data, tr)

    tgt = hist["cells"][["lat", "lon"]]
    plain = fcst.build(raw, src, tgt, drv, data)
    issued = plain["pr"].dropna(how="all").index
    fc_idx = tr[(tr >= pd.Timestamp(fcst_start)) & tr.isin(issued)]
    scale = fcst.fit_scale(plain["pr"], data["pr"], config.RAIN_CELL, fc_idx)
    d_fc = _forecast_data(data, hist, raw, src, drv, scale, horizons)

    if oof_years:
        parts = []
        for yr in sorted(set(fc_idx.year)):
            held = fc_idx[fc_idx.year == yr]
            m = spec.build_model(horizons)
            m.a.fit(data, tr[tr.year != yr])
            m.b.fit(data, tr[tr.year != yr])
            parts.append(m.point(d_fc, held))
        preds_fc = pd.concat(parts)
        _set_bands(model, preds_fc, d_fc, fc_idx)
        learned = [h for h in horizons if h >= LEARNED_BANDS_FROM_H]
        scaled = [h for h in horizons if h >= LEARNED_BANDS_FROM_H or h <= WET_BANDS_MAX_H]
        pre = tr[tr < pd.Timestamp(fcst_start)]
        if scaled and len(pre):
            sms = _set_learned_bands(model, data, tr, pre, d_fc, fc_idx, preds_fc, scaled, scale_block_years, gap_days, learned)
            _set_wet_bands(model, sms, d_fc, fc_idx, preds_fc, weights)
    else:
        model.calibrate(d_fc, fc_idx)

    return Artifact(model, drv, scale, live.pet_climatology(hist["pet"]), horizons, str(train_end), str(fcst_start),
                    meta={"n_train": len(tr), "n_calib": len(fc_idx), "oof_years": oof_years})


def predict(art: Artifact, hist: dict, raw: pd.DataFrame, src: pd.DataFrame, issue_day) -> pd.DataFrame:
    """Forecast issued for day t (made the morning of t+1): one row per horizon."""
    day = pd.Timestamp(issue_day).normalize()
    data = _with_gr4j(hist, art.gr4j)
    d_fc = _forecast_data(data, hist, raw, src, art.gr4j, art.scale, art.horizons)
    fc = art.model.predict(d_fc, pd.DatetimeIndex([day], name="date")).iloc[0]
    return pd.DataFrame(
        [{"date": day + pd.Timedelta(days=h), "p10": fc[f"h{h}_p10"], "p50": fc[f"h{h}_p50"], "p90": fc[f"h{h}_p90"]} for h in art.horizons]
    )


def save(art: Artifact, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(art))


def load(path) -> Artifact:
    return pickle.loads(Path(path).read_bytes())


def backtest(art: Artifact, hist: dict, raw: pd.DataFrame, src: pd.DataFrame, index: pd.DatetimeIndex, future: str = "forecast") -> pd.DataFrame:
    """Forecasts for many issue days at once from archived inputs.

    future='forecast' uses the issued GEFS rain (honest), 'perfect' the observed future rain.
    """
    data = _with_gr4j(hist, art.gr4j)
    d = _forecast_data(data, hist, raw, src, art.gr4j, art.scale, art.horizons) if future == "forecast" else data
    return art.model.predict(d, pd.DatetimeIndex(index))

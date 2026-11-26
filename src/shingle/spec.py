"""The selected forecast model, in one place (choices documented in docs/shingle_findings.md).

Blend (0.5/0.5 per horizon) of
  * ARX: LightGBM on the h-day stage change (state, lake, rain, GR4J, morning reading h<=3)
  * RatingHybrid: LightGBM flow + Toho forecasts -> floodplain CombinedRating, residual carry
Day 1 is replaced, wherever the morning reading exists, by MorningAnchor: LightGBM (L1) on day-1 mean minus the
issue reading, stage features + morning readings + the midnight-8 am mean (docs/project_review.md, idea 1).
Morning (issue-time) readings only for h <= 3 and allowed missing (exist from 2007).
"""

from __future__ import annotations

from shingle import config
from shingle.models import anchor, arx, blend, direct, hybrid
from shingle.models.base import HORIZONS

LGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20}
RATING_OPTS = {"floodplain": True, "d_max": 1.2, "g_min": 0.3}
AM_MAX_H = 3
AM_FLOW = ["log_q_am_d", "log_q_up_am_d"]
AM_STAGE = AM_FLOW + ["stage_am_d"]
SLOPES = ["stage_slope6", "stage_slope12", "up_slope6", "up_slope12"]
DAY1_EXTRA = ["stage_sofar_d"] + SLOPES  # rate of rise: -2.4 % at day 1 (experiments/improve/i4_report.txt)
GR4J = ["gr4j_logq", "gr4j_dH", "gr4j_s", "gr4j_r"]


def _rain(cell: str) -> list[str]:
    return [f"pr_past{w}__{cell}" for w in (1, 3, 7, 30)] + [f"pr_nextH__{cell}", f"wb_past30__{cell}", f"wb_past90__{cell}"]


def feature_sets(cell: str = config.RAIN_CELL) -> dict[str, list[str]]:
    rain = _rain(cell)
    return {
        "stage": ["stage", "dstage_1", "dstage_3", "log_q", "log_q_up", "toho", "toho_dev", "toho_target_dH", "log_s61", "doy_sin", "doy_cos"] + rain + GR4J,
        "flow": ["log_q", "log_q_d1", "log_q_up", "doy_sin", "doy_cos"] + rain + GR4J,
        "lake": ["toho", "dtoho_1", "dtoho_3", "toho_dev", "toho_target_dH", "log_s61", "log_s59", "etoho", "log_q", "doy_sin", "doy_cos",
                 f"pr_nextH__{cell}", f"pr_past3__{cell}"],
    }


def _future(f: str) -> bool:
    return "next" in f or f == "gr4j_dH"


def build_model(horizons=HORIZONS, future_rain: bool = True, cell: str = config.RAIN_CELL) -> blend.Blend:
    fs = feature_sets(cell)
    if not future_rain:
        fs = {k: [f for f in v if not _future(f)] for k, v in fs.items()}
    hs = tuple(horizons)
    # rate of rise at every horizon: forecast mode days 3-7 -2 to -4.5 % (experiments/improve/i4_fc_report.txt)
    stage_f = {h: fs["stage"] + (AM_STAGE if h <= AM_MAX_H else []) + SLOPES for h in hs}
    flow_f = {h: fs["flow"] + (AM_FLOW if h <= AM_MAX_H else []) for h in hs}
    a = arx.ARX(stage_f, hs, backend="lgbm", allow_missing=AM_STAGE + SLOPES, lgbm_params=LGBM_PARAMS)
    b = hybrid.RatingHybrid(
        direct.DirectRegressor("log_q", flow_f, hs, backend="lgbm", allow_missing=AM_FLOW, lgbm_params=LGBM_PARAMS),
        direct.DirectRegressor("toho", fs["lake"], hs, backend="lgbm", lgbm_params=LGBM_PARAMS),
        hs,
        rating_opts=RATING_OPTS,
    )
    day1 = anchor.MorningAnchor(fs["stage"] + AM_STAGE + DAY1_EXTRA, lgbm_params=LGBM_PARAMS) if 1 in hs else None
    return blend.Blend(a, b, weights=0.5, horizons=hs, day1=day1)

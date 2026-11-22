"""Score a set of candidate models on one evaluation period and write a report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from home_display.shingle import config, evaluate
from home_display.shingle.models.base import targets


def score_models(
    models: dict,
    data: dict,
    train_index: pd.DatetimeIndex,
    eval_index: pd.DatetimeIndex,
    horizons=(1, 3, 7),
    high: float = config.FLOOD_STAGES["action"],
    eval_data: dict | None = None,
) -> dict:
    """Fit each model on train_index; predict eval_index with leave-one-year-out conformal bands.

    eval_data: inputs available at forecast time (e.g. data with issued rain forecasts in
    data['fcst']); used for prediction and band calibration. Defaults to `data`.
    """
    ed = data if eval_data is None else eval_data
    y = targets(data["stage"], horizons)
    low = float(data["stage"].loc[train_index].quantile(0.1))
    rows, ev_rows, forecasts = [], [], {}
    for name, model in models.items():
        model.fit(data, train_index)
        fc = evaluate.cross_conformal_predict(model, ed, eval_index)
        forecasts[name] = fc
        for h in horizons:
            yy = y[f"h{h}"].loc[eval_index]
            pm = evaluate.point_metrics(yy, fc[f"h{h}_p50"], high=high, low=low)
            sk = evaluate.threshold_skill(yy, fc[f"h{h}_p50"], high)
            rows.append(
                {
                    "model": name,
                    "h": h,
                    **{k: pm[k] for k in ("mae", "rmse", "bias", "mae_high", "n_high", "mae_low", "n_low")},
                    "cov80": evaluate.coverage(yy, fc[f"h{h}_p10"], fc[f"h{h}_p90"]),
                    "pod": sk["pod"],
                    "far": sk["far"],
                }
            )
            # align forecast issued at t with valid date t+h for event scoring
            valid = fc[f"h{h}_p50"].copy()
            valid.index = valid.index + pd.Timedelta(days=h)
            ev = evaluate.event_peaks(data["stage"].loc[eval_index], valid.reindex(eval_index), high)
            for r in ev.to_dict("records"):
                ev_rows.append({"model": name, "h": h, **r})
    return {
        "metrics": pd.DataFrame(rows),
        "events": pd.DataFrame(ev_rows, columns=["model", "h", "start", "end", "obs_peak", "obs_peak_date", "pred_peak", "pred_peak_date", "peak_error", "timing_error_days"]),
        "forecasts": forecasts,
        "low_threshold": low,
    }


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(res: dict, path: Path, title: str, notes: list[str] | None = None) -> None:
    m = res["metrics"]
    parts = [f"# {title}", ""]
    for n in notes or []:
        parts.append(f"- {n}")
    parts += ["", f"Low-water threshold (train 10th pct): {res['low_threshold']:.2f} ft", ""]
    for metric in ("mae", "mae_high", "mae_low", "cov80", "pod", "far"):
        piv = m.pivot(index="model", columns="h", values=metric).reset_index()
        piv.columns = ["model"] + [f"h{c}" for c in piv.columns[1:]]
        parts += [f"## {metric}", "", _md_table(piv), ""]
    if not res["events"].empty:
        ev = res["events"][["model", "h", "obs_peak", "pred_peak", "peak_error", "timing_error_days"]]
        parts += ["## flood events (obs ≥ action stage)", "", _md_table(ev), ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))

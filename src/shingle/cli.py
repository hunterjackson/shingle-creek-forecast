"""`shingle` command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from shingle import config, fetch, pipeline, webdata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shingle", description="Shingle Creek at Campbell stage forecast")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download/update historical inputs into the data dir")
    p_fetch.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_fetch.add_argument("--refresh", action="store_true", help="ignore cache and refetch everything")

    p_update = sub.add_parser("update", help="refresh all model inputs (history caches, IV, radar, GEFS) without predicting")
    p_update.add_argument("--data-dir", type=Path, default=config.DATA_DIR)

    p_train = sub.add_parser("train", help="fit the forecast model artifact")
    p_train.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_train.add_argument("--model", type=Path, default=pipeline.MODEL_PATH)
    p_train.add_argument("--train-end", default=None, help="last training day (default: 8 days ago)")

    p_pred = sub.add_parser("predict", help="update inputs and write the 7-day forecast JSON")
    p_pred.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_pred.add_argument("--model", type=Path, default=pipeline.MODEL_PATH)
    p_pred.add_argument("--out", type=Path, default=pipeline.OUT_PATH)
    p_pred.add_argument("--no-update", action="store_true", help="use cached inputs only")
    p_pred.add_argument("--log", type=Path, default=pipeline.LOG_PATH, help="forecast log CSV to append to")

    p_site = sub.add_parser("site", help="write the static site's JSON (forecast, observed, scorecard)")
    p_site.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_site.add_argument("--forecast", type=Path, default=pipeline.OUT_PATH)
    p_site.add_argument("--log", type=Path, default=pipeline.LOG_PATH)
    p_site.add_argument("--out-dir", type=Path, default=Path("site/data"))

    p_score = sub.add_parser("score", help="verify logged forecasts (ours vs NWM) against observed stage")
    p_score.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_score.add_argument("--log", type=Path, default=pipeline.LOG_PATH)

    p_ee = sub.add_parser("ee-check", help="compare gridMET from Earth Engine (the backup) with the cached THREDDS copy")
    p_ee.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p_ee.add_argument("--days", type=int, default=20)

    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        asyncio.run(fetch.fetch_history(args.data_dir, refresh=args.refresh))
    elif args.cmd == "update":
        pipeline.update_inputs(args.data_dir)
    elif args.cmd == "train":
        import pandas as pd

        end = args.train_end or str((pd.Timestamp.now().normalize() - pd.Timedelta(days=8)).date())
        pipeline.train_from_disk(args.data_dir, args.model, end)
    elif args.cmd == "predict":
        pipeline.predict_from_disk(args.data_dir, args.model, args.out, update=not args.no_update, log_path=args.log)
    elif args.cmd == "site":
        webdata.build(args.data_dir, args.forecast, args.log, args.out_dir)
    elif args.cmd == "ee-check":
        import pandas as pd

        from shingle.sources import gridmet, gridmet_ee

        bad = False
        for var in gridmet.VARS:
            cached = pd.read_parquet(args.data_dir / f"gridmet_{var}.parquet")
            start = str((cached.index.max() - pd.Timedelta(days=args.days)).date())
            end = str(pd.Timestamp.now().normalize().date())
            ee_df = asyncio.run(gridmet_ee.fetch(var, gridmet.cells_from_columns(cached.columns), start, end))
            r = gridmet_ee.compare(cached, ee_df)
            print(var, r)
            bad |= not (r["overlap_days"] > 0 and r["missing_cells"] == 0 and r["max_abs_diff"] <= 0.1)
        raise SystemExit(1 if bad else 0)
    elif args.cmd == "score":
        print(pipeline.score_from_disk(args.data_dir, args.log).round(3).to_string())

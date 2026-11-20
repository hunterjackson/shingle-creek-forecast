"""Pull and cache all historical inputs under a data directory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from home_display.shingle import basin as basin_mod
from home_display.shingle import cache, config
from home_display.shingle.sources import gridmet, nldi, usgs


def fetch_history(data_dir: Path, today: str | None = None, start: str = config.HISTORY_START, refresh: bool = False) -> None:
    data_dir = Path(data_dir)

    for site in (config.SITE, config.UPSTREAM_SITE):
        cache.update_daily(
            data_dir / f"usgs_{site}.parquet",
            lambda s, e, site=site: usgs.fetch_dv(site, s, e),
            default_start=start,
            today=today,
            refresh=refresh,
        )

    for var in gridmet.VARS:
        cache.update_daily(
            data_dir / f"gridmet_{var}.parquet",
            lambda s, e, var=var: gridmet.fetch(var, config.BBOX, s, e),
            default_start=start,
            today=today,
            refresh=refresh,
        )

    geojson = cache.cached_json(
        data_dir / "basin.geojson", lambda: nldi.fetch_basin_geojson(config.SITE), refresh=refresh
    )
    pr_cols = pd.read_parquet(data_dir / "gridmet_pr.parquet").columns
    cells = basin_mod.annotate_cells(gridmet.cells_from_columns(pr_cols), nldi.parse_basin(geojson), config.OUTLET)
    cells.to_parquet(data_dir / "cells.parquet")


def load_history(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    return {
        "gauge": pd.read_parquet(data_dir / f"usgs_{config.SITE}.parquet"),
        "upstream": pd.read_parquet(data_dir / f"usgs_{config.UPSTREAM_SITE}.parquet"),
        "pr": pd.read_parquet(data_dir / "gridmet_pr.parquet"),
        "pet": pd.read_parquet(data_dir / "gridmet_pet.parquet"),
        "cells": pd.read_parquet(data_dir / "cells.parquet"),
        "basin": nldi.parse_basin(json.loads((data_dir / "basin.geojson").read_text())),
    }

import numpy as np
import pandas as pd
import pytest

from home_display.shingle import grid


def pix(lats, lons):
    return [f"{la:.3f}_{lo:.3f}" for la in lats for lo in lons]


def test_aggregate_blocks_averages_pixels_into_coarser_cells():
    lats = [28.505, 28.495, 28.485, 28.475]
    lons = [-81.495, -81.485, -81.475, -81.465]
    cols = pix(lats, lons)
    df = pd.DataFrame([np.arange(16, dtype=float)], columns=cols, index=pd.DatetimeIndex(["2020-01-01"], name="date"))

    out = grid.aggregate_blocks(df, factor=2)

    assert out.shape == (1, 4)
    assert sorted(out.iloc[0].tolist()) == [2.5, 4.5, 10.5, 12.5]


def test_aggregate_blocks_ignores_nan_pixels():
    cols = pix([28.505, 28.495], [-81.495, -81.485])
    df = pd.DataFrame([[1.0, np.nan, 3.0, np.nan]], columns=cols)

    out = grid.aggregate_blocks(df, factor=2)

    assert out.iloc[0, 0] == pytest.approx(2.0)


def test_block_ids_are_block_centres():
    cols = pix([28.505, 28.495], [-81.495, -81.485])
    df = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], columns=cols)

    out = grid.aggregate_blocks(df, factor=2)

    lat, lon = map(float, out.columns[0].split("_"))
    assert lat == pytest.approx(28.500, abs=1e-6)
    assert lon == pytest.approx(-81.490, abs=1e-6)


def test_nearest_mapping():
    targets = pd.DataFrame({"lat": [28.50, 28.30], "lon": [-81.50, -81.40]}, index=["t1", "t2"])
    sources = pd.DataFrame({"lat": [28.49, 28.31, 28.0], "lon": [-81.51, -81.41, -81.0]}, index=["s1", "s2", "s3"])

    m = grid.nearest(targets, sources)

    assert m.to_dict() == {"t1": "s1", "t2": "s2"}


def test_remap_columns_builds_frame_for_target_cells():
    idx = pd.date_range("2020-01-01", periods=2, name="date")
    src = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]}, index=idx)

    out = grid.remap(src, pd.Series({"t1": "s2", "t2": "s2", "t3": "s1"}))

    assert list(out.columns) == ["t1", "t2", "t3"]
    assert out["t1"].tolist() == [3.0, 4.0]

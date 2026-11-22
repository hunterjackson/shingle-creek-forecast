import numpy as np
import pandas as pd

from shingle import radar


def test_build_regions_clusters_correlated_cells_using_train_period_only():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-07-01", "2023-12-31", name="date")
    a, b = rng.gamma(0.3, 10, len(idx)), rng.gamma(0.3, 10, len(idx))
    df = pd.DataFrame(
        {
            "28.300_-81.400": a, "28.300_-81.380": a + rng.normal(0, 0.3, len(idx)),
            "28.500_-81.400": b, "28.500_-81.380": b + rng.normal(0, 0.3, len(idx)),
        },
        index=idx,
    )
    # after train_end the two groups become identical - must be ignored
    df.loc["2022-01-01":, "28.500_-81.400"] = df.loc["2022-01-01":, "28.300_-81.400"]

    reg = radar.build_regions(df, train_end="2021-12-31", threshold=0.85)

    assert reg["28.300_-81.400"] == reg["28.300_-81.380"]
    assert reg["28.500_-81.400"] == reg["28.500_-81.380"]
    assert reg["28.300_-81.400"] != reg["28.500_-81.400"]


def test_regions_round_trip_json(tmp_path):
    reg = pd.Series({"c1": 1, "c2": 1, "c3": 2})
    path = tmp_path / "regions.json"

    radar.save_regions(reg, path, meta={"threshold": 0.85})
    back, meta = radar.load_regions(path)

    assert back.to_dict() == reg.to_dict()
    assert meta["threshold"] == 0.85


def test_region_centroids():
    reg = pd.Series({"28.300_-81.400": 1, "28.320_-81.420": 1, "28.500_-81.400": 2})

    cen = radar.region_centroids(reg)

    assert list(cen.index) == ["r1", "r2"]
    assert cen.loc["r1", "lat"] == 28.31

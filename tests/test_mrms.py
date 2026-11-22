import gzip

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
import xarray as xr

from shingle.sources import mrms
from shingle.sources.gridmet import BBox

BBOX = BBox(north=28.62, south=28.18, west=-81.60, east=-81.30)


def test_urls_for_local_day_use_next_day_05z_and_product_order():
    urls = mrms.candidate_urls(pd.Timestamp("2024-10-09"))

    assert urls == [
        "https://mtarchive.geol.iastate.edu/2024/10/10/mrms/ncep/MultiSensor_QPE_24H_Pass2/"
        "MultiSensor_QPE_24H_Pass2_00.00_20241010-050000.grib2.gz",
        "https://mtarchive.geol.iastate.edu/2024/10/10/mrms/ncep/GaugeCorr_QPE_24H/"
        "GaugeCorr_QPE_24H_00.00_20241010-050000.grib2.gz",
    ]


def test_urls_before_multisensor_era_only_gaugecorr():
    urls = mrms.candidate_urls(pd.Timestamp("2019-06-14"))

    assert len(urls) == 1 and "GaugeCorr_QPE_24H_00.00_20190615-050000" in urls[0]


def test_urls_before_archive_start_raise():
    with pytest.raises(ValueError):
        mrms.candidate_urls(pd.Timestamp("2017-01-01"))


def fake_grid(value=5.0):
    lat = np.round(np.arange(28.705, 28.095, -0.01), 3)
    lon = np.round(np.arange(278.305, 278.795, 0.01), 3)  # 0-360 longitudes like MRMS
    data = np.full((lat.size, lon.size), value)
    data[0, 0] = -3.0  # MRMS "no coverage" flag
    return xr.DataArray(data, coords={"latitude": lat, "longitude": lon}, dims=("latitude", "longitude"))


def test_crop_to_bbox_flattens_to_cell_series_and_masks_negative():
    da = fake_grid()
    da.loc[dict(latitude=28.505, longitude=278.505)] = -1.0

    s = mrms.crop(da, BBOX)

    assert s.index.name == "cell_id"
    assert all(-81.60 <= float(c.split("_")[1]) <= -81.30 for c in s.index)
    assert all(28.18 <= float(c.split("_")[0]) <= 28.62 for c in s.index)
    assert np.isnan(s["28.505_-81.495"])
    assert s.drop("28.505_-81.495").eq(5.0).all()


@respx.mock
def test_fetch_day_falls_back_to_second_product(monkeypatch):
    urls = mrms.candidate_urls(pd.Timestamp("2024-10-09"))
    respx.get(urls[0]).mock(return_value=httpx.Response(404))
    respx.get(urls[1]).mock(return_value=httpx.Response(200, content=gzip.compress(b"GRIB")))
    monkeypatch.setattr(mrms, "decode_grib", lambda raw: fake_grid(7.0))

    s = mrms.fetch_day(pd.Timestamp("2024-10-09"), BBOX, client=httpx.Client())

    assert s.name == pd.Timestamp("2024-10-09")
    assert s.dropna().eq(7.0).all()


@respx.mock
def test_fetch_day_all_missing_returns_none():
    for u in mrms.candidate_urls(pd.Timestamp("2024-10-09")):
        respx.get(u).mock(return_value=httpx.Response(404))

    assert mrms.fetch_day(pd.Timestamp("2024-10-09"), BBOX, client=httpx.Client()) is None


def test_fetch_range_builds_wide_frame_and_skips_missing(monkeypatch):
    def fake_day(day, bbox, client=None, hour=5):
        if day == pd.Timestamp("2024-01-02"):
            return None
        s = mrms.crop(fake_grid(float(day.day)), bbox)
        s.name = day
        return s

    monkeypatch.setattr(mrms, "fetch_day", fake_day)

    df = mrms.fetch("2024-01-01", "2024-01-03", BBOX, workers=2)

    assert df.index.name == "date"
    assert list(df.index) == list(pd.date_range("2024-01-01", "2024-01-03"))
    assert df.loc["2024-01-02"].isna().all()
    assert df.loc["2024-01-03"].dropna().eq(3.0).all()


@pytest.mark.live
def test_live_decode_real_file():
    s = mrms.fetch_day(pd.Timestamp("2024-10-09"), BBOX)
    assert s is not None and s.size == 44 * 30
    assert 100 < s.mean() < 200  # Milton


def test_urls_for_gridmet_day_end_at_12z_next_day():
    urls = mrms.candidate_urls(pd.Timestamp("2024-10-09"), hour=12)

    assert urls[0].endswith("MultiSensor_QPE_24H_Pass2_00.00_20241010-120000.grib2.gz")


def test_fetch_passes_hour_through(monkeypatch):
    seen = []

    def fake_day(day, bbox, client=None, hour=5):
        seen.append(hour)
        s = mrms.crop(fake_grid(1.0), bbox)
        s.name = day
        return s

    monkeypatch.setattr(mrms, "fetch_day", fake_day)

    mrms.fetch("2024-01-01", "2024-01-02", BBOX, workers=1, hour=12)

    assert seen == [12, 12]

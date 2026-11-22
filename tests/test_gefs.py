from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from shingle.sources import gefs
from shingle.sources.gridmet import BBox

FIXTURES = Path(__file__).parent / "fixtures"
RUN = pd.Timestamp("2022-09-26 00:00")
BBOX = BBox(north=28.62, south=28.18, west=-81.60, east=-81.30)


def test_url_for_mean_and_spread():
    assert gefs.url(RUN, 24) == (
        "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20220926/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f024"
    )
    assert gefs.url(RUN, 6, kind="spread").endswith("/gespr.t00z.pgrb2a.0p50.f006")
    with pytest.raises(ValueError):
        gefs.url(pd.Timestamp("2019-06-01"), 24)  # before GEFSv12 archive


def test_apcp_range_from_idx():
    text = (FIXTURES / "gefs_geavg_f024.idx").read_text()

    start, end, step = gefs.apcp_range(text)

    assert (start, end) == (11364191, 11364191 + 87745)
    assert step == "18-24"


def test_decode_crop_returns_cells_around_bbox_in_mm():
    raw = (FIXTURES / "gefs_apcp_f024.grib2").read_bytes()

    s = gefs.decode_crop(raw, BBOX, margin=0.5)

    # 0.5 deg grid points covering 27.68..29.12 N, -82.10..-80.80 E
    assert set(s.index) == {f"{la:.4f}_{lo:.4f}" for la in (28.0, 28.5, 29.0) for lo in (-82.0, -81.5, -81.0)}
    assert (s >= 0).all() and s.max() < 500
    assert s.name == "18-24"


@respx.mock
def test_fetch_run_uses_byte_ranges_and_stacks_buckets():
    text = (FIXTURES / "gefs_geavg_f024.idx").read_text()
    raw = (FIXTURES / "gefs_apcp_f024.grib2").read_bytes()
    respx.get(url__regex=r".*\.idx$").mock(return_value=httpx.Response(200, text=text))
    data_route = respx.get(url__regex=r".*0p50\.f0\d\d$").mock(return_value=httpx.Response(206, content=raw))

    df = gefs.fetch_run(RUN, fhours=(18, 24), bbox=BBOX)

    assert list(df.index) == [18, 24]
    assert df.index.name == "fhour"
    assert data_route.calls.last.request.headers["Range"] == f"bytes=11364191-{11364191 + 87745 - 1}"
    assert df.shape[1] == 9


def test_fetch_run_returns_none_when_run_missing():
    with respx.mock:
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        assert gefs.fetch_run(RUN, fhours=(6,), bbox=BBOX) is None


def test_daily_leads_sums_12z_windows_per_issue_day():
    # 00Z run on 2022-09-26 serves issue day 2022-09-25 (forecast made ~07 EST on the 26th)
    fh = list(range(6, 186, 6))
    run = pd.Timestamp("2022-09-26")
    buckets = pd.DataFrame({"c": [float(f) for f in fh]}, index=pd.MultiIndex.from_product([[run], fh], names=["run", "fhour"]))

    out = gefs.daily_leads(buckets, leads=(1, 2, 7))

    assert out.index.tolist() == [pd.Timestamp("2022-09-25")]
    # lead 1 = day 09-26 12Z -> 09-27 12Z = buckets ending f018, f024, f030, f036
    assert out.loc["2022-09-25", (1, "c")] == 18 + 24 + 30 + 36
    assert out.loc["2022-09-25", (2, "c")] == 42 + 48 + 54 + 60
    assert out.loc["2022-09-25", (7, "c")] == 162 + 168 + 174 + 180


def test_daily_leads_nan_when_bucket_missing():
    fh = [18, 24, 30]  # f036 missing
    buckets = pd.DataFrame({"c": 1.0}, index=pd.MultiIndex.from_product([[pd.Timestamp("2022-09-26")], fh], names=["run", "fhour"]))

    out = gefs.daily_leads(buckets, leads=(1,))

    assert np.isnan(out.iloc[0][(1, "c")])


@respx.mock
def test_fetch_run_retries_transient_errors():
    text = (FIXTURES / "gefs_geavg_f024.idx").read_text()
    raw = (FIXTURES / "gefs_apcp_f024.grib2").read_bytes()
    respx.get(url__regex=r".*\.idx$").mock(side_effect=[httpx.ConnectTimeout("handshake"), httpx.Response(200, text=text)])
    respx.get(url__regex=r".*0p50\.f0\d\d$").mock(side_effect=[httpx.ReadTimeout("slow"), httpx.Response(206, content=raw)])

    df = gefs.fetch_run(RUN, fhours=(24,), bbox=BBOX, retry_wait=0)

    assert list(df.index) == [24]


def test_grid_lons_handle_wrapped_grids():
    # GEFS: 0 .. 359.5 ; ECMWF open data: starts at 180 and wraps to 179.6
    g = gefs.grid_lons(0.0, 720, 0.5)
    assert g[0] == 0.0 and g[-1] == -0.5 and g[558] == pytest.approx(-81.0)
    e = gefs.grid_lons(180.0, 900, 0.4)
    assert e[0] == -180.0 and e[-1] == pytest.approx(179.6)
    assert np.argmin(np.abs(e - (-81.2))) == round((180 - 81.2) / 0.4)

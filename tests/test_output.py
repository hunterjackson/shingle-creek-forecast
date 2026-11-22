import json

import pandas as pd
import pytest

from home_display.shingle import output


@pytest.mark.parametrize("stage,cat", [(55.0, "normal"), (58.99, "normal"), (59.0, "action"), (60.2, "minor"), (61.0, "moderate"), (64.2, "major"), (float("nan"), "unknown")])
def test_flood_category(stage, cat):
    assert output.flood_category(stage) == cat


def test_forecast_json_schema(tmp_path):
    fc = pd.DataFrame({"date": pd.date_range("2024-10-10", periods=2), "p10": [55.0, 56.0], "p50": [56.0, 59.5], "p90": [57.0, 61.2]})
    rain = pd.Series([12.3, 40.0], index=fc["date"])
    path = tmp_path / "shingle.json"

    doc = output.write_json(
        path, fc, current_stage=55.4, current_time=pd.Timestamp("2024-10-09 12:00", tz="UTC"),
        issue_day=pd.Timestamp("2024-10-09"), rain_mm=rain, meta={"train_end": "2023-12-31"},
    )

    on_disk = json.loads(path.read_text())
    assert on_disk == doc
    assert doc["site"]["usgs"] == "02264495" and doc["site"]["nws"] == "SHIF1"
    assert doc["current"] == {"stage_ft": 55.4, "time": "2024-10-09T12:00:00+00:00", "category": "normal"}
    assert doc["flood_stages"] == {"action": 59.0, "minor": 60.0, "moderate": 61.0, "major": 62.0}
    assert doc["forecast"][1] == {"date": "2024-10-11", "p10": 56.0, "p50": 59.5, "p90": 61.2, "category": "action",
                                  "p90_category": "moderate", "rain_mm": 40.0}
    assert doc["issue_day"] == "2024-10-09" and doc["model"] == {"train_end": "2023-12-31"}

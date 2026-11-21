import pandas as pd

from home_display.shingle import config


def test_split_dates_ordered_and_match_decision():
    assert config.TRAIN_END == "2021-12-31"
    assert config.VAL_END == "2023-12-31"
    assert pd.Timestamp(config.TRAIN_END) < pd.Timestamp(config.VAL_END)
    assert config.MRMS_START == "2018-06-30"

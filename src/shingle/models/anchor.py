"""Day-1 model anchored on the morning reading (docs/project_review.md, idea 1).

At issue (13:30Z on t+1) a third of the day-1 target day (EST t+1) is already observed. This LightGBM model
(L1 loss) predicts the day-1 mean minus the issue reading (stage_am), trained only on days that have the
intraday record (2007+). Out-of-fold this beats the blend at day 1 by ~40 %; training the same tree on all rows,
with the day-t mean standing in before 2007, loses most of the gain (experiments/improve/i1_report.txt).
Days without a morning reading get NaN, and the blend's day 1 is used instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shingle import features

# readings that may be missing (a gap in an hour, upstream gauge down); LightGBM handles NaN
OPTIONAL = ("log_q_am_d", "log_q_up_am_d", "stage_slope6", "stage_slope12", "up_slope6", "up_slope12")


class MorningAnchor:
    name = "morning_anchor"
    horizons = (1,)

    def __init__(self, features: list[str], lgbm_params: dict | None = None, seed: int = 0):
        self.features = list(features)
        self.lgbm_params = dict(lgbm_params or {})
        self.seed = seed
        self.model = None
        self.n_train = 0

    def _X(self, data: dict, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
        X = features.build(data, self.features, horizon=1).loc[index]
        ok = X[[c for c in self.features if c not in OPTIONAL]].notna().all(axis=1)
        if "stage_am" in data:
            ok &= data["stage_am"].reindex(index).notna()
        else:
            ok &= False
        return X, ok

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "MorningAnchor":
        import lightgbm as lgb

        self.model, self.n_train = None, 0
        if "stage_am" not in data:
            return self
        X, ok = self._X(data, train_index)
        dy = (data["stage"].shift(-1) - data["stage_am"]).loc[train_index]
        ok &= dy.notna()
        if ok.sum() == 0:
            return self
        params = dict(verbose=-1, random_state=self.seed)
        params.update(self.lgbm_params)
        params["objective"] = "l1"
        self.model = lgb.LGBMRegressor(**params).fit(X[ok].to_numpy(), dy[ok].to_numpy())
        self.n_train = int(ok.sum())
        return self

    def point(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        out = pd.Series(np.nan, index=index)
        if self.model is not None and "stage_am" in data:
            X, ok = self._X(data, index)
            if ok.any():
                out[ok] = data["stage_am"].reindex(index)[ok] + self.model.predict(X[ok].to_numpy())
        return pd.DataFrame({"h1": out})

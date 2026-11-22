"""Direct multi-horizon regressor for one target variable.

For each horizon h, regress the change target(t+h) - target(t) (in a
transformed space) on named features, then add it back to target(t).
Targets: 'stage' (ft), 'log_q' (log1p cfs; predicted back in cfs), 'toho' (ft).
Backends: 'ridge' (standardized RidgeCV) or 'lgbm' (LightGBM, median).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shingle import features
from shingle.models.base import HORIZONS

_TARGETS = {
    "stage": (lambda d: d["stage"], lambda x: x),
    "log_q": (lambda d: np.log1p(d["q"]), np.expm1),
    "toho": (lambda d: d["toho"], lambda x: x),
}


def _make_backend(backend: str, seed: int = 0, lgbm_params: dict | None = None):
    if backend == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if backend == "lgbm":
        import lightgbm as lgb

        params = dict(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
        )
        params.update(lgbm_params or {})
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"unknown backend {backend!r}")


class DirectRegressor:
    def __init__(
        self,
        target: str,
        features: list[str] | dict[int, list[str]],
        horizons=HORIZONS,
        backend: str = "ridge",
        offset: str | None = None,
        allow_missing: list[str] | tuple = (),
        lgbm_params: dict | None = None,
        monotone: dict[str, int] | None = None,
    ):
        """offset: optional feature name (e.g. 'gr4j_dH') giving a physical model's predicted
        change; the regressor then learns only the correction, and the offset is added back,
        so out-of-range physical signals pass through un-capped."""
        if target not in _TARGETS:
            raise ValueError(f"unknown target {target!r}")
        if allow_missing and backend != "lgbm":
            raise ValueError("allow_missing needs the lgbm backend (native NaN handling)")
        # features that may be NaN (e.g. intraday readings that only exist from 2007);
        # LightGBM learns a default split direction for missing values
        self.allow_missing = set(allow_missing)
        self.n_train: dict[int, int] = {}
        self.lgbm_params = dict(lgbm_params or {})
        # feature -> +1/-1 monotone constraint (lgbm); e.g. more forecast rain never lowers flow
        self.monotone = dict(monotone or {})
        self.target = target
        self.horizons = tuple(horizons)
        self._features = features
        self.backend = backend
        self.offset = offset
        self.models: dict[int, object] = {}

    def feature_names(self, h: int) -> list[str]:
        return list(self._features[h] if isinstance(self._features, dict) else self._features)

    def _complete(self, X: pd.DataFrame, h: int) -> pd.Series:
        required = [c for c in X.columns if c not in self.allow_missing]
        return X[required].notna().all(axis=1)

    def _offset(self, data: dict, h: int) -> pd.Series:
        return features.build(data, [self.offset], horizon=h)[self.offset]

    def _level(self, data: dict) -> pd.Series:
        return _TARGETS[self.target][0](data)

    def fit(self, data: dict, train_index: pd.DatetimeIndex) -> "DirectRegressor":
        level = self._level(data)
        for h in self.horizons:
            X = features.build(data, self.feature_names(h), horizon=h).loc[train_index]
            dy = (level.shift(-h) - level).loc[train_index]
            if self.offset:
                dy = dy - self._offset(data, h).loc[train_index]
            ok = self._complete(X, h) & dy.notna()
            params = dict(self.lgbm_params)
            if self.monotone and self.backend == "lgbm":
                params["monotone_constraints"] = [int(self.monotone.get(c, 0)) for c in X.columns]
            model = _make_backend(self.backend, lgbm_params=params) if self.backend == "lgbm" else _make_backend(self.backend)
            model.fit(X[ok].to_numpy(), dy[ok].to_numpy())
            self.models[h] = model
            self.n_train[h] = int(ok.sum())
        return self

    def predict(self, data: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Predicted target level at t+h (original units) for each issue date t."""
        level = self._level(data).loc[index]
        inverse = _TARGETS[self.target][1]
        out = {}
        for h in self.horizons:
            X = features.build(data, self.feature_names(h), horizon=h).loc[index]
            ok = self._complete(X, h)
            pred = pd.Series(np.nan, index=index)
            if ok.any():
                pred[ok] = level[ok] + self.models[h].predict(X[ok].to_numpy())
            if self.offset:
                pred = pred + self._offset(data, h).loc[index]
            out[f"h{h}"] = inverse(pred)
        return pd.DataFrame(out, index=index)

"""Model test plan (docs/model_test_plan.md): shared data, configurable model factory, parallel OOF runner.

A config dict selects one variation of the production spec:
  lgbm:   dict of LightGBM overrides (applied to all three learners)
  seed:   LightGBM random_state
  backend:'lgbm' | 'catboost'
  arx:    'lgbm' | 'linear_tree' | 'mono' | 'ridge_resid'
  lake:   'lgbm' | 'ridge' | 'avg'
  sim:    key in SIMS (rainfall-runoff simulation used for gr4j_* features)
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SP = os.path.dirname(os.path.abspath(__file__)) + "/"  # outputs (oof/, sims/, data.pkl) land here; gitignored
os.chdir(REPO)
from shingle import config, dataset, features, fetch, gr4j_driver, spec  # noqa: E402
from shingle.models import arx, direct, hybrid  # noqa: E402
from shingle.models import blend as blend_mod  # noqa: E402

H = (1, 3, 5, 7)
END = config.VAL_END  # 2023-12-31; the test period is never touched here
LOW = 52.48
CELL = config.RAIN_CELL
SIM_DIR = SP + "sims/"
DATA_PKL = SP + "data.pkl"

_DATA = None


def load() -> dict:
    global _DATA
    if _DATA is None:
        if os.path.exists(DATA_PKL):
            _DATA = pickle.load(open(DATA_PKL, "rb"))
        else:
            hist = fetch.load_history("data")
            d = dataset.build_data(hist)
            d["gr4j"] = pd.read_parquet("data/gr4j_sim_train2021.parquet")
            d["_cells"] = hist["cells"]
            pickle.dump(d, open(DATA_PKL, "wb"))
            _DATA = d
    return _DATA


def index() -> pd.DatetimeIndex:
    d = load()
    return d["stage"].dropna().loc[:END].index


def sim(name: str) -> pd.DataFrame:
    if name == "gr4j":
        return load()["gr4j"]
    return pd.read_parquet(SIM_DIR + f"{name}.parquet")


# ---------------- learners ----------------

class CatBoostLike:
    """sklearn-ish CatBoost wrapper used in place of LightGBM."""

    def __init__(self, seed=0, **kw):
        from catboost import CatBoostRegressor

        p = dict(iterations=600, learning_rate=0.05, depth=6, loss_function="RMSE", verbose=0, random_seed=seed, thread_count=1)
        p.update(kw)
        self.m = CatBoostRegressor(**p)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


def _patch_backend(backend: str, seed: int):
    if backend == "catboost":
        def make(b, seed_=0, lgbm_params=None):
            if b == "ridge":
                return _orig_make(b)
            return CatBoostLike(seed=seed)
        direct._make_backend = make
    else:
        direct._make_backend = _orig_make


_orig_make = direct._make_backend


class RidgeResid:
    """Stage model: ridge on the h-day change (complete-case features), LightGBM on the ridge residual."""

    def __init__(self, feats: dict, lin_feats: dict, horizons, lgbm_params):
        self.lin = direct.DirectRegressor("stage", lin_feats, horizons, backend="ridge")
        self.feats, self.horizons, self.lp = feats, tuple(horizons), lgbm_params
        self.res = {}

    def fit(self, data, ix):
        self.lin.fit(data, ix)
        base = self.lin.predict(data, ix)
        import lightgbm as lgb

        for h in self.horizons:
            X = features.build(data, self.feats[h], horizon=h).loc[ix]
            r = data["stage"].shift(-h).loc[ix] - base[f"h{h}"]
            ok = r.notna()
            p = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, random_state=0, verbose=-1)
            p.update(self.lp)
            self.res[h] = lgb.LGBMRegressor(**p).fit(X[ok].to_numpy(), r[ok].to_numpy())
        return self

    def point(self, data, ix):
        base = self.lin.predict(data, ix)
        out = {}
        for h in self.horizons:
            X = features.build(data, self.feats[h], horizon=h).loc[ix]
            out[f"h{h}"] = base[f"h{h}"] + self.res[h].predict(X.to_numpy())
        return pd.DataFrame(out, index=ix)


class AvgLake:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def fit(self, data, ix):
        self.a.fit(data, ix)
        self.b.fit(data, ix)
        return self

    def predict(self, data, ix):
        pa, pb = self.a.predict(data, ix), self.b.predict(data, ix)
        return pa.where(pb.isna(), (pa + pb) / 2).fillna(pb)


def build(cfg: dict):
    """Return (stage_model, hybrid_model) per cfg; production = build({})."""
    lp = dict(spec.LGBM_PARAMS)
    lp.update(cfg.get("lgbm", {}))
    lp["random_state"] = cfg.get("seed", 0)
    lp.setdefault("n_jobs", 1)
    _patch_backend(cfg.get("backend", "lgbm"), cfg.get("seed", 0))
    fs = spec.feature_sets()
    stage_f = {h: fs["stage"] + (spec.AM_STAGE if h <= spec.AM_MAX_H else []) for h in H}
    flow_f = {h: fs["flow"] + (spec.AM_FLOW if h <= spec.AM_MAX_H else []) for h in H}
    kind = cfg.get("arx", "lgbm")
    if kind == "ridge_resid":
        lin = {h: [f for f in fs["stage"]] for h in H}
        a = RidgeResid(stage_f, lin, H, lp)
    else:
        alp = dict(lp)
        mono = None
        if kind == "linear_tree":
            alp.update(linear_tree=True, linear_lambda=cfg.get("linear_lambda", 1.0))
        if kind == "mono":
            mono = {f: 1 for f in fs["stage"] if "next" in f or f == "gr4j_dH"}
        a = arx.ARX(stage_f, H, backend="lgbm", allow_missing=spec.AM_STAGE, lgbm_params=alp, monotone=mono)
    flow = direct.DirectRegressor("log_q", flow_f, H, backend="lgbm", allow_missing=spec.AM_FLOW, lgbm_params=lp)
    lk = cfg.get("lake", "lgbm")
    lake_lgbm = direct.DirectRegressor("toho", fs["lake"], H, backend="lgbm", lgbm_params=lp)
    lake_ridge = direct.DirectRegressor("toho", fs["lake"], H, backend="ridge")
    lake = {"lgbm": lake_lgbm, "ridge": lake_ridge, "avg": AvgLake(lake_lgbm, lake_ridge)}[lk]
    b = hybrid.RatingHybrid(flow, lake, H, rating_opts=spec.RATING_OPTS)
    return a, b


# ---------------- OOF ----------------

def folds(ix: pd.DatetimeIndex, block_years=3, gap_days=7):
    years = sorted(set(ix.year))
    out = []
    for i in range(0, len(years), block_years):
        blk = years[i : i + block_years]
        held = ix[ix.year.isin(blk)]
        start = pd.Timestamp(f"{blk[0]}-01-01")
        near = (ix < start) & (ix >= start - pd.Timedelta(days=gap_days))
        out.append((blk, ix[~ix.year.isin(blk) & ~near], held))
    return out


def _run_fold(args):
    cfg, k = args
    d = load()
    if cfg.get("sim", "gr4j") != "gr4j":
        d = dict(d, gr4j=sim(cfg["sim"]))
    _, tr, held = folds(index())[k]
    a, b = build(cfg)
    a.fit(d, tr)
    b.fit(d, tr)
    pa, pb = a.point(d, held), b.point(d, held)
    return pd.concat({"a": pa, "b": pb}, axis=1)


def oof(name: str, cfg: dict, workers: int = 8, force: bool = False) -> pd.DataFrame:
    path = SP + f"oof/{name}.parquet"
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    os.makedirs(SP + "oof", exist_ok=True)
    load()
    t = time.time()
    n = len(folds(index()))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        parts = list(ex.map(_run_fold, [(cfg, k) for k in range(n)]))
    df = pd.concat(parts).sort_index()
    df.columns = [f"{x}_{y}" for x, y in df.columns]
    df.to_parquet(path)
    print(f"[oof] {name} {time.time() - t:.0f}s", flush=True)
    return df


def blend(df: pd.DataFrame, w=0.5) -> pd.DataFrame:
    w = w if isinstance(w, dict) else {h: w for h in H}
    return pd.DataFrame({f"h{h}": w[h] * df[f"a_h{h}"] + (1 - w[h]) * df[f"b_h{h}"] for h in H}, index=df.index)


# ---------------- metrics ----------------

def targets(ix) -> pd.DataFrame:
    s = load()["stage"]
    return pd.DataFrame({f"h{h}": s.shift(-h).reindex(ix) for h in H}, index=ix)


def metrics(p: pd.DataFrame) -> dict:
    y = targets(p.index)
    out = {}
    for h in H:
        e = (p[f"h{h}"] - y[f"h{h}"])
        ok = e.notna()
        e, yy = e[ok], y[f"h{h}"][ok]
        out[f"mae{h}"] = e.abs().mean()
        out[f"hi{h}"] = e[yy >= 59].abs().mean()
        out[f"lo{h}"] = e[yy <= LOW].abs().mean()
    return out


def crest_errors(p: pd.DataFrame, h: int) -> pd.DataFrame:
    from shingle import evaluate

    s = load()["stage"].loc[: END]
    pred_t = p[f"h{h}"].copy()
    pred_t.index = pred_t.index + pd.Timedelta(days=h)  # by target date
    obs = s.reindex(pred_t.index)
    ok = obs.notna() & pred_t.notna()
    return evaluate.event_peaks(obs[ok], pred_t[ok], 59.0)


def boot_diff(pc: pd.DataFrame, pp: pd.DataFrame, n=2000, block=120, seed=0, subset=None) -> pd.DataFrame:
    """Paired moving-block bootstrap of MAE(candidate) - MAE(production) per horizon (ft and %)."""
    rng = np.random.default_rng(seed)
    y = targets(pp.index)
    rows = []
    for h in H:
        ec = (pc[f"h{h}"] - y[f"h{h}"]).abs()
        ep = (pp[f"h{h}"] - y[f"h{h}"]).abs()
        ok = ec.notna() & ep.notna()
        if subset == "hi":
            ok &= y[f"h{h}"] >= 59
        if subset == "lo":
            ok &= y[f"h{h}"] <= LOW
        dc, dp = ec[ok].to_numpy(), ep[ok].to_numpy()
        m = len(dc)
        if m == 0:
            continue
        b = min(block, m)
        nb = int(np.ceil(m / b))
        diffs = np.empty(n)
        for i in range(n):
            st = rng.integers(0, m - b + 1, nb)
            ii = (st[:, None] + np.arange(b)).ravel()[:m]
            diffs[i] = dc[ii].mean() - dp[ii].mean()
        base = dp.mean()
        rows.append({"h": h, "n": m, "prod": base, "cand": dc.mean(), "diff": dc.mean() - base, "pct": 100 * (dc.mean() - base) / base,
                     "lo90": np.quantile(diffs, 0.05), "hi90": np.quantile(diffs, 0.95)})
    return pd.DataFrame(rows).set_index("h")

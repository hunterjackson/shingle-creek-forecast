import sys, json; sys.path.insert(0, ".")
import numpy as np, mt
rng = np.random.default_rng(42)
cfgs = {
    "B_linear_tree": {"arx": "linear_tree"},
    "B_mono": {"arx": "mono"},
    "B_ridge_resid": {"arx": "ridge_resid"},
    "F_lake_ridge": {"lake": "ridge"},
    "F_lake_avg": {"lake": "avg"},
}
nest = {0.02: 750, 0.05: 300, 0.1: 150}
for i in range(10):
    lr = float(rng.choice([0.02, 0.05, 0.1]))
    cfgs[f"A_rs{i}"] = {"lgbm": {"learning_rate": lr, "n_estimators": nest[lr], "num_leaves": int(rng.choice([15, 31, 63, 127])),
                        "min_child_samples": int(rng.choice([10, 20, 40, 80])), "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
                        "subsample": float(rng.choice([0.7, 0.8, 1.0])), "subsample_freq": 1, "reg_lambda": float(rng.choice([0.0, 1.0, 10.0]))}}
json.dump(cfgs, open(mt.SP + "s1_cfgs.json", "w"), indent=1)
for name, cfg in cfgs.items():
    try:
        df = mt.oof(name, cfg)
        print(name, {k: round(v, 4) for k, v in mt.metrics(mt.blend(df)).items()}, flush=True)
    except Exception as e:
        print("FAIL", name, repr(e), flush=True)

import sys; sys.path.insert(0, ".")
import mt
d = mt.load(); print("gr4j sim range", d["gr4j"].dropna().index.min(), d["gr4j"].dropna().index.max(), "am", [k for k in d if k.endswith("_am")])
print("folds", len(mt.folds(mt.index())), "rows", len(mt.index()))
for s in range(5):
    df = mt.oof(f"prod_s{s}", {"seed": s})
    print(s, {k: round(v, 4) for k, v in mt.metrics(mt.blend(df)).items()}, flush=True)

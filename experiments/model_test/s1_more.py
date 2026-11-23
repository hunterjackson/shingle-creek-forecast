import sys, json; sys.path.insert(0, ".")
import mt
names = [n for n in json.load(open(mt.SP + "sims_info.json")) if n != "gr4j"]
cfgs = {f"C1_{n}": {"sim": n} for n in names}
cfgs["A_catboost"] = {"backend": "catboost"}
for name, cfg in cfgs.items():
    try:
        df = mt.oof(name, cfg)
        print(name, {k: round(v, 4) for k, v in mt.metrics(mt.blend(df)).items()}, flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAIL", name, repr(e), flush=True)

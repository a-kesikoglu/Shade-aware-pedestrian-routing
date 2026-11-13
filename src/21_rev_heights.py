"""Building height checks: reference sample inside and outside the study area, storey height conversion,
height-dependent error model, class-specific bias scenario and a table of median gains per error model.
Outputs: data/processed/rev_heights.json, data/processed/rev_height_mc.pkl, outputs/tables/rev_uncertainty.csv"""
import importlib.util
import json
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd
from config import *

from mcworker import run as _run
K = 30
FACTORS = [2.8, 3.0, 3.3, 3.5]
GCLS = [0, 10, 20, 400]          # 3D-GloBFP height classes used for the height-dependent model


def stats_(h_ref, h_glo):
    e = h_glo - h_ref
    r = np.log(h_ref / h_glo)
    return {"n": int(len(e)), "bias_m": float(e.mean()), "rmse_m": float(np.sqrt((e ** 2).mean())),
            "r": float(np.corrcoef(h_ref, h_glo)[0, 1]) if len(e) > 2 else None,
            "median_abs_rel_err": float((np.abs(e) / h_ref).median()) if len(e) else None,
            "ratio_med": float(np.exp(np.median(r)))}


def part_a():
    v = pd.read_csv(PROC / "height_validation_detail.csv")
    lev = np.where(v.tag == "levels", v.h_osm / FLOOR_HEIGHT, np.nan)
    out = {"all": stats_(v.h_osm, v.h_glo), "in_aoi": stats_(v.h_osm[v.in_aoi], v.h_glo[v.in_aoi]),
           "margin": stats_(v.h_osm[~v.in_aoi], v.h_glo[~v.in_aoi]),
           "n_levels": int((v.tag == "levels").sum()), "n_height": int((v.tag == "height").sum()),
           "n_levels_in_aoi": int(((v.tag == "levels") & v.in_aoi).sum())}
    out["storey_factor"] = {}
    for f in FACTORS:
        h = np.where(v.tag == "levels", lev * f, v.h_osm)
        out["storey_factor"][f"{f:.1f}"] = stats_(pd.Series(h), v.h_glo)
    gc = pd.cut(v.h_glo, GCLS, labels=["<10", "10-20", ">=20"])
    out["by_glo_class"] = {str(c): stats_(g.h_osm, g.h_glo) for c, g in v.groupby(gc, observed=True)}
    hcl = pd.cut(v.h_osm, GCLS, labels=["<10", "10-20", ">=20"])
    out["by_ref_class"] = {str(c): stats_(g.h_osm, g.h_glo) for c, g in v.groupby(hcl, observed=True)}
    return out, v


def uncertainty_table():
    base = pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"]
    ref = {t: 100 * np.median([1 - r["B10"]["dsun"] / r["SP"]["dsun"] if r["SP"]["dsun"] > 0 else 0
                               for r in base[t]]) for t in PERIODS}
    ind = pickle.load(open(PROC / "height_mc.pkl", "rb"))
    dep = pickle.load(open(PROC / "height_mc_dependent.pkl", "rb"))
    models = {"independent": ind["mc"], "source polygon": dep["grouped"], "250 m block": dep["block"]}
    hd = PROC / "rev_height_mc.pkl"
    if hd.exists():
        models["height dependent"] = pickle.load(open(hd, "rb"))["mc"]
    rows = []
    for name, reals in models.items():
        for t in PERIODS:
            meds = np.array([100 * np.median(x[t]["rows"][:, 4]) for x in reals])
            jac = np.concatenate([x[t]["rows"][:, 0] for x in reals])
            rows.append({"model": name, "time": t, "ref": ref[t], "min": meds.min(), "max": meds.max(),
                         "width": meds.max() - meds.min(), "max_shift": np.abs(meds - ref[t]).max(),
                         "jaccard_med": float(np.median(jac)), "n": len(reals)})
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "rev_uncertainty.csv", index=False)
    return tab


if __name__ == "__main__":
    A, v = part_a()
    _r = np.log(v.h_osm / v.h_glo).values
    _g = np.digitize(v.h_glo.values, GCLS[1:-1])
    A["hd_pool_n"] = {str(c): int((_g == c).sum()) for c in range(3)}
    A["hd_pool_factor"] = {str(c): float(np.exp(np.median(_r[_g == c]))) for c in range(3)}
    if "mc" in sys.argv[1:]:
        bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
        h0 = bld.h.values
        perturb = np.isin(bld.h_src.values, ["globfp", "globfp_footprint", "imputed"])
        r = np.log(v.h_osm / v.h_glo).values
        gref = np.digitize(v.h_glo.values, GCLS[1:-1])            # class of each reference building
        gb = np.digitize(h0, GCLS[1:-1])                           # class of each modelled building
        pools = {c: r[gref == c] for c in range(3)}
        med = {c: float(np.median(pools[c])) for c in range(3)}
        A["hd_pool_n"] = {str(c): int(len(pools[c])) for c in range(3)}
        A["hd_pool_factor"] = {str(c): float(np.exp(med[c])) for c in range(3)}
        rng = np.random.default_rng(SEED + 23)
        jobs = []
        for k in range(K):                                       # random part, centred within each class
            e = np.zeros(len(h0))
            for c in range(3):
                m = gb == c
                e[m] = rng.choice(pools[c] - med[c], size=m.sum(), replace=True)
            jobs.append((f"hd{k}", np.maximum(h0 * np.where(perturb, np.exp(e), 1.0), 3.0)))
        bias_c = np.array([med[c] for c in gb])                   # class-specific systematic correction
        jobs.append(("hd_bias", np.maximum(h0 * np.where(perturb, np.exp(bias_c), 1.0), 3.0)))
        for f in FACTORS:
            fac = A["storey_factor"][f"{f:.1f}"]["ratio_med"]
            jobs.append((f"bias_f{f:.1f}", np.maximum(h0 * np.where(perturb, fac, 1.0), 3.0)))
        with Pool(8) as pool:
            res = dict(pool.map(_run, jobs, chunksize=1))
        pickle.dump({"mc": [res[f"hd{k}"] for k in range(K)], "hd_bias": res["hd_bias"],
                     "bias_f": {f"{f:.1f}": res[f"bias_f{f:.1f}"] for f in FACTORS}},
                    open(PROC / "rev_height_mc.pkl", "wb"))
    tab = uncertainty_table()
    if (PROC / "rev_height_mc.pkl").exists():
        R = pickle.load(open(PROC / "rev_height_mc.pkl", "rb"))
        A["hd_bias"] = {t: {"net_sunlit": R["hd_bias"][t]["net_sunlit"],
                            "B10_dDsun_med": float(100 * np.median(R["hd_bias"][t]["rows"][:, 4]))} for t in PERIODS}
        A["bias_f"] = {f: {t: {"net_sunlit": x[t]["net_sunlit"],
                               "B10_dDsun_med": float(100 * np.median(x[t]["rows"][:, 4]))} for t in PERIODS}
                       for f, x in R["bias_f"].items()}
    A["uncertainty"] = {f"{r.model}_{r.time}": r._asdict() for r in tab.itertuples(index=False)}
    json.dump(A, open(PROC / "rev_heights.json", "w"), indent=1, default=float)
    print(json.dumps({k: A[k] for k in A if k != "uncertainty"}, indent=1, default=float))
    print(tab.round(2).to_string())

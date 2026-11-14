"""B10 routes over a grid of CTEI weights (step 0.1 on the simplex) for the four periods.
Output: outputs/tables/rev_weight_grid.csv"""
import pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
from config import *

_G = {}


def _init():
    from impedance import load_segments
    from routing import Net
    seg = load_segments()
    _G.update(seg=seg, L=seg.length.values, net=Net(seg.u.values, seg.v.values, seg.length.values),
              OD=pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"],
              base=pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"])


def job(w):
    from impedance import thermal_cost
    from routing import pareto_paths, budget
    if not _G:
        _init()
    seg, L, net, OD, base = (_G[k] for k in ["seg", "L", "net", "OD", "base"])
    rows = []
    for t in PERIODS:
        s = seg[f"s_{t.replace(':', '')}"].values
        c2 = thermal_cost(L, s, seg.ndvi_n.values, seg.lst_n.values, w)
        red, jac, same = [], [], []
        for k, (o, d, _) in enumerate(OD):
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
            b = sols[budget(sols, L0, 0.10)][2]
            sp = base[t][k]["SP"]; b0 = base[t][k]["B10"]["edges"]
            red.append(100 * (sp["dsun"] - (L[b] * s[b]).sum()) / sp["dsun"] if sp["dsun"] > 0 else 0.0)
            A, B = set(b), set(b0)
            jac.append(L[list(A & B)].sum() / L[list(A | B)].sum()); same.append(A == B)
        rows.append({"w1": w[0], "w2": w[1], "w3": w[2], "time": t, "dDsun_med": float(np.median(red)),
                     "dDsun_mean": float(np.mean(red)), "J_mean": float(np.mean(jac)), "identical": float(np.mean(same))})
    print(w, flush=True)
    return rows


if __name__ == "__main__":
    W = [(round(a / 10, 1), round(b / 10, 1), round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]
    W = [w for w in W if w[0] + w[1] + w[2] > 0]
    with Pool(6) as p:
        res = p.map(job, W, chunksize=1)
    pd.DataFrame([r for x in res for r in x]).to_csv(TAB / "rev_weight_grid.csv", index=False)

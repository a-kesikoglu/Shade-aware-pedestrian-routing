"""O-D sampling, exact Pareto routing for four diurnal periods, satellite-only baseline,
weight and penalty sensitivity."""
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from config import *
from impedance import load_segments, thermal_cost
from routing import Net, pareto_paths, knee, budget, run_metrics

seg = load_segments()
net = Net(seg.u.values, seg.v.values, seg.length.values)
L = seg.length.values
NDV, LSTN = seg.ndvi_n.values, seg.lst_n.values
S = {t: seg[f"s_{t.replace(':', '')}"].values for t in PERIODS}

# ---------------------------------------------------------------- O-D sampling (floor-area weighted)
nodes = pd.read_csv(PROC / "network_nodes.csv").set_index("node").loc[net.nodes]
bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
gfa = bld.geometry.area.values * np.maximum(1, np.round(bld.h.values / FLOOR_HEIGHT))
bc = bld.geometry.centroid
kt = cKDTree(np.c_[bc.x, bc.y])
nb = kt.query_ball_point(nodes[["x", "y"]].values, r=50)
wn = np.array([gfa[i].sum() for i in nb]) + 1.0
p = wn / wn.sum()


def sample_od(n, seed, dmin=500, dmax=2500):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        o, d = rng.choice(net.N, 2, p=p)
        if o == d:
            continue
        l0 = net.sssp(L, o)[d]
        if dmin <= l0 <= dmax:
            out.append((int(o), int(d), float(l0)))
    return out


OD = sample_od(500, SEED)
OD_FLOW = sample_od(5000, SEED + 1)
pickle.dump({"OD": OD, "OD_FLOW": OD_FLOW, "node_w": wn}, open(PROC / "od_sets.pkl", "wb"))


def describe(es, sunlit, c2):
    es = np.asarray(es, int)
    l = L[es]
    ln = l.sum()
    dsun = float((l * sunlit[es]).sum())
    return {"len": float(ln), "cost": float(c2[es].sum()), "dsun": dsun, "shade_share": 1 - dsun / ln,
            "lst": float((l * seg.lst.values[es]).sum() / ln), "ndvi": float((l * seg.ndvi.values[es]).sum() / ln),
            "run": run_metrics(es, L, sunlit)}


def jaccard(a, b):
    a, b = set(a), set(b)
    return float(L[list(a & b)].sum() / L[list(a | b)].sum()) if a | b else 1.0


def route_set(o, d, c2, sunlit):
    sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
    picks = {"SP": 0, "B5": budget(sols, L0, 0.05), "B10": budget(sols, L0, 0.10),
             "KP": knee(sols), "MTEP": len(sols) - 1}
    r = {k: describe(sols[i][2], sunlit, c2) | {"edges": sols[i][2]} for k, i in picks.items()}
    r["n_pareto"] = len(sols)
    r["front"] = [(s[0], s[1]) for s in sols]
    return r


# ---------------------------------------------------------------- main experiment
res = {}
for t in PERIODS:
    c2 = thermal_cost(L, S[t], NDV, LSTN, WEIGHTS["balanced"])
    res[t] = [route_set(o, d, c2, S[t]) for o, d, _ in OD]
    print(t, "done")

# satellite-only impedance (no shade term, no penalty), time invariant
c_sat = thermal_cost(L, np.zeros_like(L), NDV, LSTN, (0, 0.5, 0.5), beta=0)
sat = []
for o, d, _ in OD:
    sols, L0 = pareto_paths(net, o, d, L, c_sat, detour=1.5)
    sat.append({"B10": sols[budget(sols, L0, 0.10)][2], "KP": sols[knee(sols)][2]})

# weight and penalty sensitivity
sens = {}
for name, w in WEIGHTS.items():
    for t in PERIODS:
        c2 = thermal_cost(L, S[t], NDV, LSTN, w)
        sens[(name, BETA, t)] = []
        for o, d, _ in OD:
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
            sens[(name, BETA, t)].append({"B10": sols[budget(sols, L0, 0.10)][2], "KP": sols[knee(sols)][2],
                                          "KP_len": sols[knee(sols)][0], "SP": sols[0][2]})
for beta in [0.0, 10.0]:
    for t in PERIODS:
        c2 = thermal_cost(L, S[t], NDV, LSTN, WEIGHTS["balanced"], beta=beta)
        sens[("balanced", beta, t)] = []
        for o, d, _ in OD:
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
            sens[("balanced", beta, t)].append({"B10": sols[budget(sols, L0, 0.10)][2],
                                                "KP": sols[knee(sols)][2], "KP_len": sols[knee(sols)][0],
                                                "SP": sols[0][2]})
    print("beta", beta)

pickle.dump({"res": res, "sat": sat, "sens": sens}, open(PROC / "routing_results.pkl", "wb"))
print("saved")

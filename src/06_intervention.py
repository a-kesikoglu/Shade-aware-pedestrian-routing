"""Thermal betweenness and priority shade-corridor scenarios."""
import json
import pickle
import numpy as np
import pandas as pd
from config import *
from impedance import load_segments, thermal_cost, robust_minmax
from routing import Net, pareto_paths, budget

seg = load_segments()
net = Net(seg.u.values, seg.v.values, seg.length.values)
L = seg.length.values
ods = pickle.load(open(PROC / "od_sets.pkl", "rb"))
OD, OD_FLOW = ods["OD"], ods["OD_FLOW"]
HRS = HOURLY
SH = np.stack([seg[f"s_{t.replace(':', '')}"].values for t in HRS])     # hours x edges
s_day = SH.mean(0)
a, b = np.nanpercentile(seg.ndvi, [2, 98])


# trip-weighted edge betweenness from 5000 floor-area weighted shortest paths
def sp_edges(o, d):
    _, pred = net.sssp(L, o, pred=True)
    return net.path_edges(pred, o, d)


flow = np.zeros(len(L))
for o, d, _ in OD_FLOW:
    flow[sp_edges(o, d)] += 1
seg["flow"] = flow
seg["s_day"] = s_day
seg["tb"] = flow * s_day
SP_EVAL = [sp_edges(o, d) for o, d, _ in OD]


def select(score, frac, rng=None):
    if rng is not None:
        order = rng.permutation(len(L))
    else:
        tie = np.random.default_rng(0).random(len(L)) * 1e-9
        order = np.argsort(-(score + tie))
    cum = np.cumsum(L[order])
    return order[: np.searchsorted(cum, frac * L.sum()) + 1]


def evaluate(treated):
    s = SH.copy()
    s[:, treated] = np.minimum(s[:, treated], 0.25)
    nd = seg.ndvi.values.copy()
    nd[treated] = nd[treated] + 0.25
    ndn = np.clip((nd - a) / (b - a), 0, 1)
    cost_sp, dsun_sp = 0.0, 0.0
    for h in range(len(HRS)):
        c2 = thermal_cost(L, s[h], ndn, seg.lst_n.values, WEIGHTS["balanced"])
        for es in SP_EVAL:
            cost_sp += c2[es].sum(); dsun_sp += (L[es] * s[h, es]).sum()
    # cool routing after intervention at 15:00 (B10)
    h15 = HRS.index("15:00")
    c2 = thermal_cost(L, s[h15], ndn, seg.lst_n.values, WEIGHTS["balanced"])
    cb, db = 0.0, 0.0
    for o, d, _ in OD:
        sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
        es = sols[budget(sols, L0, 0.10)][2]
        cb += c2[es].sum(); db += (L[es] * s[h15, es]).sum()
    return {"cost_sp_day": cost_sp, "dsun_sp_day": dsun_sp, "cost_b10_15": cb, "dsun_b10_15": db,
            "km": float(L[treated].sum() / 1000), "n_edges": int(len(treated))}


base = evaluate(np.array([], int))
rows = []
scores = {"TB": seg.tb.values, "flow": flow, "sun": s_day}
for frac in [0.01, 0.03, 0.05, 0.10]:
    for name, sc in scores.items():
        r = evaluate(select(sc, frac)); r.update(strategy=name, frac=frac); rows.append(r)
    rr = [evaluate(select(None, frac, np.random.default_rng(100 + k))) for k in range(20)]
    r = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
    r.update({f"{k}_sd": float(np.std([x[k] for x in rr], ddof=1)) for k in ["cost_sp_day", "dsun_sp_day"]})
    r.update(strategy="random", frac=frac); rows.append(r)
    print(frac, "done")
df = pd.DataFrame(rows)
for k in ["cost_sp_day", "dsun_sp_day", "cost_b10_15", "dsun_b10_15"]:
    df[f"red_{k}"] = 100 * (base[k] - df[k]) / base[k]
df.to_csv(TAB / "intervention_scenarios.csv", index=False)
seg[["eid", "flow", "s_day", "tb", "geometry"]].to_file(PROC / "thermal_betweenness.gpkg", driver="GPKG")
json.dump(base, open(PROC / "intervention_base.json", "w"), indent=1)
print(df[["strategy", "frac", "km", "red_cost_sp_day", "red_dsun_sp_day", "red_cost_b10_15", "red_dsun_b10_15"]].round(2))

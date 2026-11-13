"""Additional routing analyses: supported and unsupported Pareto solutions, direct minimisation of the
sunlit distance within the 10% budget, shortest path identity of the satellite baseline, zero baselines,
paired test table in metres, route overlap by period and absolute differences."""
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from config import *
from impedance import load_segments, thermal_cost
from routing import Net, pareto_paths, budget, knee

seg = load_segments()
L = seg.length.values
net = Net(seg.u.values, seg.v.values, L)
R = pickle.load(open(PROC / "routing_results.pkl", "rb"))
res = R["res"]
OD = pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"]
S = {t: seg[f"s_{t.replace(':', '')}"].values for t in PERIODS}
NDV, LSTN = seg.ndvi_n.values, seg.lst_n.values
out = {}
CLASSES = ["B5", "B10", "KP", "MTEP"]


def jac(a, b):
    a, b = set(a), set(b)
    u = list(a | b)
    return float(L[list(a & b)].sum() / L[u].sum()) if u else 1.0


def supported_flags(front, tol=1e-9):
    """True for points on the lower-left convex hull of a front sorted by f1 (f2 decreasing)."""
    f = np.asarray(front, float)
    n = len(f)
    if n <= 2:
        return np.ones(n, bool)
    g = (f - f.min(0)) / np.where(np.ptp(f, 0) > 0, np.ptp(f, 0), 1)
    flags = np.zeros(n, bool)
    hull = [0]
    for k in range(1, n):
        while len(hull) >= 2:
            a, b = g[hull[-2]], g[hull[-1]]
            cross = (b[0] - a[0]) * (g[k][1] - a[1]) - (b[1] - a[1]) * (g[k][0] - a[0])
            if cross <= tol:      # b is not below the segment a-k
                hull.pop()
            else:
                break
        hull.append(k)
    # points on hull edges (collinear) count as supported
    for i in range(len(hull) - 1):
        a, b = g[hull[i]], g[hull[i + 1]]
        for k in range(hull[i], hull[i + 1] + 1):
            cross = (b[0] - a[0]) * (g[k][1] - a[1]) - (b[1] - a[1]) * (g[k][0] - a[0])
            if abs(cross) <= 1e-7:
                flags[k] = True
    return flags


def index_of(front, x):
    f = np.asarray(front)
    return int(np.argmin(np.abs(f[:, 0] - x["len"]) + np.abs(f[:, 1] - x["cost"])))


# ---------------------------------------------------------------- 1 supported / unsupported solutions
uns = {}
for t in PERIODS:
    n_sol = n_uns = 0
    pairs_with = 0
    cls = {c: 0 for c in CLASSES}
    for r in res[t]:
        fl = supported_flags(r["front"])
        n_sol += len(fl); n_uns += int((~fl).sum()); pairs_with += int((~fl).any())
        for c in CLASSES:
            cls[c] += int(not fl[index_of(r["front"], r[c])])
    uns[t] = {"solutions": n_sol, "unsupported": n_uns, "unsupported_share": n_uns / n_sol,
              "pairs_with_unsupported_share": pairs_with / len(res[t]),
              **{f"{c}_unsupported_share": cls[c] / len(res[t]) for c in CLASSES}}
out["unsupported"] = uns

# ---------------------------------------------------------------- 2 direct minimisation of D_sun within 10%
dmin = {}
dmin_routes = {}
for t in PERIODS:
    c_sun = L * S[t]                       # w = (1, 0, 0), beta = 0: cost equals the sunlit distance
    rows = []
    dmin_routes[t] = []
    for k, (o, d, _) in enumerate(OD):
        sols, L0 = pareto_paths(net, o, d, L, c_sun, detour=1.5)
        b = sols[budget(sols, L0, 0.10)]
        sp = res[t][k]["SP"]
        es = np.asarray(b[2], int)
        ds = float((L[es] * S[t][es]).sum())
        dmin_routes[t].append(b[2])
        red = lambda x: 100 * (sp["dsun"] - x) / sp["dsun"] if sp["dsun"] > 0 else 0.0
        rows.append({"od": k, "dsun_min": ds, "dDsun_min": red(ds), "dL_min": 100 * (L[es].sum() / sp["len"] - 1),
                     "dDsun_B10": red(res[t][k]["B10"]["dsun"]), "dDsun_KP": red(res[t][k]["KP"]["dsun"]),
                     "B10_minus_min_m": res[t][k]["B10"]["dsun"] - ds,
                     "J_B10_min": jac(es, res[t][k]["B10"]["edges"]),
                     "sp_dsun": sp["dsun"]})
    d = pd.DataFrame(rows)
    pot = d.sp_dsun - d.dsun_min
    ach = d.sp_dsun - (d.sp_dsun * (1 - d.dDsun_B10 / 100))
    m = pot > 1e-6
    dmin[t] = {"dDsun_min_med": float(d.dDsun_min.median()), "dDsun_min_mean": float(d.dDsun_min.mean()),
               "dDsun_B10_med": float(d.dDsun_B10.median()), "dL_min_med": float(d.dL_min.median()),
               "dDsun_min_p25": float(d.dDsun_min.quantile(.25)), "dDsun_min_p75": float(d.dDsun_min.quantile(.75)),
               "B10_share_of_potential_total": float(ach[m].sum() / pot[m].sum()),
               "B10_share_of_potential_med": float(np.median(ach[m] / pot[m])),
               "B10_equals_min_share": float((d.B10_minus_min_m.abs() <= 1e-6).mean()),
               "B10_minus_min_m_med": float(d.B10_minus_min_m.median()),
               "B10_minus_min_m_p90": float(d.B10_minus_min_m.quantile(.9)),
               "J_B10_min_med": float(d.J_B10_min.median()),
               "pairs_with_potential_share": float(m.mean())}
    print(t, dmin[t], flush=True)
out["dsun_min"] = dmin
pickle.dump(dmin_routes, open(PROC / "rev_dsun_min_routes.pkl", "wb"))

# ---------------------------------------------------------------- 3 satellite baseline: same SP?
c_sat = thermal_cost(L, np.zeros_like(L), NDV, LSTN, (0, 0.5, 0.5), beta=0)
same_sp = []
for k, (o, d, _) in enumerate(OD):
    sols, L0 = pareto_paths(net, o, d, L, c_sat, detour=1.5)
    same_sp.append([set(sols[0][2]) == set(res[t][k]["SP"]["edges"]) for t in PERIODS])
same_sp = np.array(same_sp)
out["sat_sp"] = {"same_sp_share_by_period": {t: float(same_sp[:, i].mean()) for i, t in enumerate(PERIODS)},
                 "shade_sp_same_across_periods_share": float(np.mean(
                     [len({frozenset(res[t][k]["SP"]["edges"]) for t in PERIODS}) == 1 for k in range(len(OD))]))}

# ---------------------------------------------------------------- 4 zero baselines
out["zero_baselines"] = {t: {"sp_dsun_zero": int(sum(r["SP"]["dsun"] <= 0 for r in res[t])),
                             "sp_cost_zero": int(sum(r["SP"]["cost"] <= 0 for r in res[t])),
                             "sp_dsun_min": float(min(r["SP"]["dsun"] for r in res[t]))} for t in PERIODS}

# ---------------------------------------------------------------- 5 paired tests in metres
rows = []
for t in PERIODS:
    sp = np.array([r["SP"]["dsun"] for r in res[t]])
    for c in CLASSES:
        x = np.array([r[c]["dsun"] for r in res[t]])
        dif = sp - x
        nz = np.abs(dif) > 1e-9
        w = stats.wilcoxon(sp, x, zero_method="wilcox")
        wplus = float(stats.rankdata(np.abs(dif[nz]))[dif[nz] > 0].sum())
        z = stats.norm.isf(w.pvalue / 2)
        dl = np.array([r[c]["len"] - r["SP"]["len"] for r in res[t]])
        rows.append({"time": t, "route": c, "n_nonzero": int(nz.sum()), "W_plus": wplus, "W": float(w.statistic),
                     "z": float(z), "p": float(w.pvalue), "r": float(z / np.sqrt(nz.sum())),
                     "diff_med_m": float(np.median(dif)), "diff_mean_m": float(dif.mean()),
                     "diff_med_nz_m": float(np.median(dif[nz])),
                     "diff_p25_m": float(np.percentile(dif, 25)), "diff_p75_m": float(np.percentile(dif, 75)),
                     "extra_len_med_m": float(np.median(dl)), "extra_len_mean_m": float(dl.mean())})
tab = pd.DataFrame(rows).sort_values("p")
m = len(tab); run = 0.0; holm = []
for i, p in enumerate(tab.p):
    run = max(run, min(1.0, (m - i) * p)); holm.append(run)
tab["p_holm"] = holm
tab = tab.sort_values(["time", "route"], key=lambda s: s.map({c: i for i, c in enumerate(PERIODS + CLASSES)}))
tab.to_csv(TAB / "rev_paired_tests.csv", index=False)
out["paired_tests"] = tab.set_index(["time", "route"]).to_dict(orient="index")
out["paired_tests"] = {f"{k[0]}_{k[1]}": v for k, v in out["paired_tests"].items()}

# ---------------------------------------------------------------- 6 absolute values and B10 overlap with SP
ab = {}
for t in PERIODS:
    sp = np.array([r["SP"]["dsun"] for r in res[t]]); b = np.array([r["B10"]["dsun"] for r in res[t]])
    dl = np.array([r["B10"]["len"] - r["SP"]["len"] for r in res[t]])
    chg = np.array([set(r["B10"]["edges"]) != set(r["SP"]["edges"]) for r in res[t]])
    jj = np.array([jac(r["B10"]["edges"], r["SP"]["edges"]) for r in res[t]])
    ab[t] = {"sp_dsun_med": float(np.median(sp)), "b10_dsun_med": float(np.median(b)),
             "avoided_med_m": float(np.median(sp - b)), "avoided_mean_m": float(np.mean(sp - b)),
             "avoided_med_changed_m": float(np.median((sp - b)[chg])),
             "avoided_med_changed_min": float(np.median((sp - b)[chg]) / 1.3 / 60),
             "extra_med_changed_m": float(np.median(dl[chg])),
             "J_SP_B10_med": float(np.median(jj)), "J_SP_B10_mean": float(jj.mean()),
             "identical_share": float(1 - chg.mean())}
out["absolute"] = ab

# ---------------------------------------------------------------- 7 KP and B10 at 09:00
kp = {}
for t in PERIODS:
    lk = np.array([r["KP"]["len"] for r in res[t]]); lb = np.array([r["B10"]["len"] for r in res[t]])
    kp[t] = {"KP_len_med": float(np.median(lk)), "B10_len_med": float(np.median(lb)),
             "KP_minus_B10_len_med": float(np.median(lk - lb)), "KP_longer_share": float(np.mean(lk > lb + 1e-6))}
out["kp_b10"] = kp

json.dump(out, open(PROC / "rev_routing.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["unsupported", "sat_sp", "zero_baselines", "absolute", "kp_b10"]}, indent=1))

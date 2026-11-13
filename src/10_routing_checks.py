"""Verification and sensitivity of the biobjective routing.

1  brute-force enumeration on synthetic grids and on short real OD pairs (completeness of the Pareto set,
   including unsupported solutions)
2  label counts, truncation and query times for the main experiment
3  detour cap 1.3 / 1.5 / 2.0 and the sunlit distance reduction along the budget curve (all Pareto routes)
4  knee point edge cases
5  threshold sensitivity of the longest sunlit run for beta = 0, 4, 10
6  time-dependent evaluation of static routes (walking speed 1.3 m/s)
7  run times of shadow casting and route queries
Outputs: data/processed/routing_checks.json, outputs/tables/budget_curve_dsun.csv
"""
import json
import pickle
import platform
import subprocess
import time
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import rasterio
from config import *
from impedance import load_segments, thermal_cost
from routing import Net, pareto_paths, knee, budget, run_metrics
from shade import sun_positions, cast

rng = np.random.default_rng(SEED + 3)
OUT = {}


def nondominated(F, tol=1e-6):
    """Unique non-dominated (f1, f2) points of an array, sorted by f1."""
    F = np.array(sorted(map(tuple, F)))
    keep, best = [], np.inf
    for f1, f2 in F:
        if f2 < best - tol:
            if keep and abs(keep[-1][0] - f1) <= tol:   # same length, lower cost replaces
                keep[-1] = (f1, f2)
            else:
                keep.append((f1, f2))
            best = f2
    return np.array(keep)


def supported_mask(F):
    """Points on the lower-left convex hull of a front (supported efficient solutions)."""
    if len(F) <= 2:
        return np.ones(len(F), bool)
    hull = [0]
    for i in range(1, len(F)):
        while len(hull) >= 2:
            a, b = F[hull[-2]], F[hull[-1]]
            if (b[0] - a[0]) * (F[i][1] - a[1]) - (b[1] - a[1]) * (F[i][0] - a[0]) <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(i)
    m = np.zeros(len(F), bool); m[hull] = True
    return m


def same_front(A, B, tol=1e-6):
    if len(A) != len(B):
        return False
    return bool(np.all(np.abs(np.asarray(A) - np.asarray(B)) <= tol * np.maximum(1, np.abs(np.asarray(B)))))


# ---------------------------------------------------------------- 1a synthetic grids
syn = {"instances": 0, "match": 0, "n_points": 0, "n_unsupported": 0, "unsupported_found": 0}
for it in range(300):
    nr, nc = (4, 5) if it % 2 else (5, 5)
    G = nx.grid_2d_graph(nr, nc)
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    E = list(G.edges)
    u = np.array([idx[a] for a, _ in E]); v = np.array([idx[b] for _, b in E])
    c1 = rng.uniform(10, 30, len(E))
    c2 = c1 * rng.choice([0.05, 0.3, 0.6, 1.0], len(E)) * rng.uniform(0.8, 1.2, len(E))
    net = Net(u, v, c1)
    o, d = (0, len(nodes) - 1) if it % 3 else tuple(rng.choice(len(nodes), 2, replace=False))
    sols, L0 = pareto_paths(net, int(o), int(d), c1, c2, detour=1.5)
    alg = nondominated([(s[0], s[1]) for s in sols])
    F = []
    eid = {frozenset((idx[a], idx[b])): k for k, (a, b) in enumerate(E)}
    for p in nx.all_simple_paths(nx.relabel_nodes(G, idx), int(o), int(d)):
        es = [eid[frozenset(x)] for x in zip(p[:-1], p[1:])]
        f1 = c1[es].sum()
        if f1 <= 1.5 * L0 + 1e-6:
            F.append((f1, c2[es].sum()))
    bf = nondominated(F)
    ok = same_front(alg, bf)
    sup = supported_mask(bf)
    syn["instances"] += 1; syn["match"] += int(ok); syn["n_points"] += len(bf)
    syn["n_unsupported"] += int((~sup).sum())
    syn["unsupported_found"] += int(sum(any(np.all(np.abs(a - b) < 1e-6) for a in alg) for b in bf[~sup]))
OUT["synthetic"] = syn
print("synthetic", syn, flush=True)

# ---------------------------------------------------------------- 1b short real OD pairs, exhaustive DFS
seg = load_segments()
L = seg.length.values
net = Net(seg.u.values, seg.v.values, L)
S = {t: seg[f"s_{t.replace(':', '')}"].values for t in PERIODS}
c15 = thermal_cost(L, S["15:00"], seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])


def enumerate_paths(o, d, c1, c2, cap, h1, limit=3_000_000):
    """All simple paths o -> d with c1 <= cap by depth-first search, pruned with the exact distance-to-go."""
    F, steps = [], 0
    stack = [(o, 0.0, 0.0, iter(net.adj[o]))]
    onpath = {o}
    while stack:
        i, f1, f2, itr = stack[-1]
        nxt = next(itr, None)
        if nxt is None:
            stack.pop(); onpath.discard(i); continue
        j, e = nxt
        steps += 1
        if steps > limit:
            return None
        if j in onpath:
            continue
        g1 = f1 + c1[e]
        if g1 + h1[j] > cap:
            continue
        g2 = f2 + c2[e]
        if j == d:
            F.append((g1, g2)); continue
        onpath.add(j)
        stack.append((j, g1, g2, iter(net.adj[j])))
    return F


real = {"tested": 0, "skipped": 0, "match": 0, "n_points": 0, "n_unsupported": 0, "unsupported_found": 0}
tries = 0
while real["tested"] < 100 and tries < 5000:
    tries += 1
    o, d = rng.choice(net.N, 2, replace=False)
    h1 = net.sssp(L, d)
    if not 150 <= h1[o] <= 350:
        continue
    sols, L0 = pareto_paths(net, int(o), int(d), L, c15, detour=1.5)
    F = enumerate_paths(int(o), int(d), L, c15, 1.5 * L0 + 1e-6, h1)
    if F is None:
        real["skipped"] += 1; continue
    bf = nondominated(F); alg = nondominated([(s[0], s[1]) for s in sols])
    sup = supported_mask(bf)
    real["tested"] += 1; real["match"] += int(same_front(alg, bf)); real["n_points"] += len(bf)
    real["n_unsupported"] += int((~sup).sum())
    real["unsupported_found"] += int(sum(any(np.all(np.abs(a - b) < 1e-6) for a in alg) for b in bf[~sup]))
OUT["real_enumeration"] = real
print("real", real, flush=True)

# ---------------------------------------------------------------- 2-4 main runs, caps, budget curve, knees
OD = pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"]
base = pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"]
EPS = [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
curve_rows, cap_rows, kn = [], [], []
labels_max, trunc, qtime = 0, 0, []
for t in PERIODS:
    c2 = thermal_cost(L, S[t], seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])
    for k, (o, d, _) in enumerate(OD):
        per_cap = {}
        for cap in (1.3, 1.5, 2.0):
            t0 = time.perf_counter()
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=cap)
            dt = time.perf_counter() - t0
            if cap == 1.5:
                qtime.append(dt)
                labels_max = max(labels_max, pareto_paths.n_labels); trunc += int(pareto_paths.truncated)
                ds = np.array([(L[s[2]] * S[t][s[2]]).sum() for s in sols])
                f = np.array([[s[0], s[1]] for s in sols])
                for eps in EPS:
                    j = budget(sols, L0, eps)
                    curve_rows.append((t, k, eps, 100 * (1 - ds[j] / ds[0]) if ds[0] > 0 else 0.0,
                                       100 * (1 - f[j, 1] / f[0, 1])))
                n = len(sols)
                if n >= 3:
                    g = (f - f.min(0)) / np.where(np.ptp(f, 0) > 0, np.ptp(f, 0), 1)
                    a_, b_ = g[0], g[-1]; dv = b_ - a_
                    dist = np.abs(dv[0] * (a_[1] - g[:, 1]) - dv[1] * (a_[0] - g[:, 0])) / np.linalg.norm(dv)
                    ties = int((np.abs(dist - dist.max()) < 1e-9).sum() > 1)
                else:
                    ties = 0
                kn.append((t, n, ties, knee(sols)))
            picks = {"B5": budget(sols, L0, 0.05), "B10": budget(sols, L0, 0.10), "KP": knee(sols),
                     "MTEP": len(sols) - 1}
            per_cap[cap] = {r: (frozenset(sols[i][2]), (L[sols[i][2]] * S[t][sols[i][2]]).sum(),
                                sols[i][0] / L0 - 1) for r, i in picks.items()}
            per_cap[cap]["SPd"] = (L[sols[0][2]] * S[t][sols[0][2]]).sum()
            per_cap[cap]["n"] = len(sols)
        for cap in (1.3, 2.0):
            for r in ["B5", "B10", "KP", "MTEP"]:
                a, b = per_cap[cap][r], per_cap[1.5][r]
                spd = per_cap[cap]["SPd"]
                cap_rows.append((t, k, cap, r, float(a[0] == b[0]), 100 * (1 - a[1] / spd) if spd > 0 else 0.0,
                                 100 * a[2]))
    print(t, "caps done", flush=True)

cur = pd.DataFrame(curve_rows, columns=["time", "od", "eps", "dDsun", "dC"])
cs = cur.groupby(["time", "eps"]).agg(dDsun_med=("dDsun", "median"), dDsun_p25=("dDsun", lambda x: x.quantile(.25)),
                                      dDsun_p75=("dDsun", lambda x: x.quantile(.75)),
                                      dC_med=("dC", "median"), dC_p25=("dC", lambda x: x.quantile(.25)),
                                      dC_p75=("dC", lambda x: x.quantile(.75))).reset_index()
cs.to_csv(TAB / "budget_curve_dsun.csv", index=False)
cap = pd.DataFrame(cap_rows, columns=["time", "od", "cap", "route", "same", "dDsun", "dL"])
OUT["detour_cap"] = {f"{c:g}": {r: {"same_share": float(g.same.mean()), "dDsun_med": float(g.dDsun.median()),
                                    "dL_med": float(g.dL.median())}
                                for r, g in cap[cap.cap == c].groupby("route")} for c in (1.3, 2.0)}
OUT["detour_cap_by_time"] = {f"{c:g}": {t: {r: float(g.dDsun.median()) for r, g in cap[(cap.cap == c) & (cap.time == t)].groupby("route")}
                                       for t in PERIODS} for c in (1.3, 2.0)}
kn = pd.DataFrame(kn, columns=["time", "n", "ties", "kp"])
OUT["knee"] = {t: {"n1": float((g.n == 1).mean()), "n2": float((g.n == 2).mean()), "n3plus": float((g.n >= 3).mean()),
                   "ties": int(g.ties.sum()), "kp_is_sp_share": float((g.kp == 0).mean())} for t, g in kn.groupby("time")}
OUT["labels"] = {"max_labels": int(labels_max), "truncated": int(trunc)}
qt = np.array(qtime) * 1000
OUT["query_ms"] = {"median": float(np.median(qt)), "p95": float(np.percentile(qt, 95)), "max": float(qt.max())}

# ---------------------------------------------------------------- 5 longest sunlit run, thresholds and beta
sens = pickle.load(open(PROC / "routing_results.pkl", "rb"))["sens"]
rr = {}
for beta in (0.0, 4.0, 10.0):
    for thr in (0.4, 0.5, 0.6):
        dd, red, kp_r = [], [], []
        for t in PERIODS:
            for x in sens[("balanced", beta, t)]:
                a = run_metrics(x["SP"], L, S[t], thr); b = run_metrics(x["B10"], L, S[t], thr)
                dd.append(b - a); red.append(b < a - 1e-9)
                kp_r.append(run_metrics(x["KP"], L, S[t], thr) - a)
        rr[f"beta{beta:g}_thr{thr}"] = {"B10_minus_SP_med": float(np.median(dd)), "B10_minus_SP_mean": float(np.mean(dd)),
                                        "B10_shorter_share": float(np.mean(red)),
                                        "KP_minus_SP_mean": float(np.mean(kp_r))}
OUT["rmax"] = rr

# ---------------------------------------------------------------- 6 time-dependent evaluation
V = 1.3                                     # m/s, comfortable walking speed of adults
TIMES = sorted(set(HOURLY + PERIODS), key=lambda s: int(s[:2]) * 60 + int(s[3:]))
th = np.array([int(s[:2]) + int(s[3:]) / 60 for s in TIMES])
SH = np.stack([seg[f"s_{s.replace(':', '')}"].values for s in TIMES])


def dsun_dynamic(es, t0):
    es = np.asarray(es, int)
    mid = np.cumsum(L[es]) - L[es] / 2
    tt = np.clip(t0 + mid / V / 3600, th[0], th[-1])
    j = np.clip(np.searchsorted(th, tt) - 1, 0, len(th) - 2)
    w = (tt - th[j]) / (th[j + 1] - th[j])
    s = (1 - w) * SH[j, es] + w * SH[j + 1, es]
    return float((L[es] * s).sum())


td = {}
for t in PERIODS:
    t0 = int(t[:2]) + int(t[3:]) / 60
    err, red_s, red_d, keep = [], [], [], []
    for r in base[t]:
        sp, b = r["SP"], r["B10"]
        ds_sp, db = dsun_dynamic(sp["edges"], t0), dsun_dynamic(b["edges"], t0)
        err.append(100 * (ds_sp - sp["dsun"]) / sp["dsun"] if sp["dsun"] > 0 else 0)
        red_s.append(100 * (1 - b["dsun"] / sp["dsun"]) if sp["dsun"] > 0 else 0)
        red_d.append(100 * (1 - db / ds_sp) if ds_sp > 0 else 0)
        keep.append(db <= ds_sp + 1e-9)
    err = np.array(err)
    td[t] = {"sp_rel_err_med": float(np.median(err)), "sp_abs_rel_err_med": float(np.median(np.abs(err))),
             "sp_abs_rel_err_p90": float(np.percentile(np.abs(err), 90)),
             "B10_dDsun_static_med": float(np.median(red_s)), "B10_dDsun_dynamic_med": float(np.median(red_d)),
             "B10_not_worse_share": float(np.mean(keep)),
             "trip_minutes_med": float(np.median([r["SP"]["len"] for r in base[t]]) / V / 60),
             "trip_minutes_max": float(np.max([r["B10"]["len"] for r in base[t]]) / V / 60)}
OUT["time_dependent"] = td
print("time dependent", td, flush=True)

# ---------------------------------------------------------------- 7 run times
with rasterio.open(PROC / "bld_h_1m.tif") as s_:
    bh = s_.read(1)
ch = rasterio.open(PROC / "canopy_h_1m.tif").read(1)
sun = sun_positions(TIMES)
ct = {}
for r in sun.itertuples():
    reps = []
    for _ in range(3):
        t0 = time.perf_counter(); cast(bh, ch, r.azimuth, r.elevation); reps.append(time.perf_counter() - t0)
    ct[r.time] = float(np.median(reps))
OUT["cast_seconds"] = ct
try:
    cpu = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True).stdout.split("\n")[1].strip()
except Exception:
    cpu = platform.processor()
OUT["hardware"] = {"cpu": cpu, "os": platform.platform(), "python": platform.python_version(),
                   "ram_gb": 32, "parallel": "single process for timings"}
json.dump(OUT, open(PROC / "routing_checks.json", "w"), indent=1)
print(json.dumps({k: OUT[k] for k in ["detour_cap", "knee", "labels", "query_ms", "rmax", "cast_seconds"]}, indent=1))

"""Statistics and tables from the routing, uncertainty and intervention experiments."""
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from config import *
from impedance import load_segments, thermal_cost
from routing import run_metrics

seg = load_segments()
L = seg.length.values
R = pickle.load(open(PROC / "routing_results.pkl", "rb"))
res, sat, sens = R["res"], R["sat"], R["sens"]
OD = pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"]
S = {t: seg[f"s_{t.replace(':', '')}"].values for t in PERIODS}
out = {}
rng = np.random.default_rng(SEED)


def boot_ci(x, f=np.median, n=2000):
    x = np.asarray(x)
    bs = [f(rng.choice(x, len(x))) for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def jac(a, b):
    a, b = set(a), set(b)
    u = list(a | b)
    return float(L[list(a & b)].sum() / L[u].sum()) if u else 1.0


# ------------------------------------------------------------------ RQ1 network shade
times = sorted(set(HOURLY + PERIODS))
ori = pd.cut(seg.bearing, [-0.1, 22.5, 67.5, 112.5, 157.5, 180.1], labels=["N-S", "NE-SW", "E-W", "NW-SE", "N-S2"])
ori = ori.astype(str).replace({"N-S2": "N-S"})
seg["ori"] = ori
rq1 = []
for t in times:
    s = seg[f"s_{t.replace(':', '')}"]
    row = {"time": t, "all": float((L * s).sum() / L.sum()),
           "fully_sunlit_share_len": float(L[s > 0.95].sum() / L.sum())}
    for o in ["N-S", "NE-SW", "E-W", "NW-SE"]:
        m = (seg.ori == o).values
        row[o] = float((L[m] * s[m]).sum() / L[m].sum())
    hm = seg.h_adj.notna().values
    row["spearman_hadj"] = float(stats.spearmanr(seg.h_adj[hm], s[hm]).statistic)
    rq1.append(row)
rq1 = pd.DataFrame(rq1)
rq1.to_csv(TAB / "network_sunlit_by_hour.csv", index=False)
out["orientation_len_share"] = {o: float(L[(seg.ori == o).values].sum() / L.sum()) for o in ["N-S", "NE-SW", "E-W", "NW-SE"]}
out["rq1"] = rq1.set_index("time").round(4).to_dict(orient="index")
# daily (07-19) mean sunlit by orientation
SH = np.stack([seg[f"s_{t.replace(':', '')}"].values for t in HOURLY])
out["daily_mean_sunlit"] = float((L * SH.mean(0)).sum() / L.sum())
seg["canopy_any"] = seg.canopy_share > 0.1
out["canopy_edges_len_share"] = float(L[seg.canopy_any].sum() / L.sum())

hcls = pd.cut(seg.h_adj, [0, 10, 15, 20, 30, 200], labels=["<10", "10-15", "15-20", "20-30", ">30"])
out["sunlit_by_hclass"] = {t: {str(c): float(np.average(g[f"s_{t.replace(':', '')}"], weights=g.length))
                               for c, g in seg.groupby(hcls, observed=True)} for t in PERIODS}
out["bld_share20_by_hclass"] = {str(c): float(np.average(g.bld_share_20m, weights=g.length))
                                for c, g in seg.groupby(hcls, observed=True)}
out["hclass_len_share"] ={str(c): float(g.length.sum() / L.sum()) for c, g in seg.groupby(hcls, observed=True)}

# ------------------------------------------------------------------ RQ2 route comparison
rows = []
for t in PERIODS:
    for k, r in enumerate(res[t]):
        sp = r["SP"]
        for typ in ["SP", "B5", "B10", "KP", "MTEP"]:
            x = r[typ]
            rows.append({"time": t, "od": k, "type": typ, "len": x["len"], "dL": 100 * (x["len"] / sp["len"] - 1),
                         "shade": 100 * x["shade_share"], "dsun": x["dsun"], "lst": x["lst"], "ndvi": x["ndvi"],
                         "run": x["run"], "dC": 100 * (sp["cost"] - x["cost"]) / sp["cost"],
                         "dDsun": 100 * (sp["dsun"] - x["dsun"]) / sp["dsun"] if sp["dsun"] > 0 else 0.0,
                         "dLST": x["lst"] - sp["lst"], "same_as_sp": float(set(x["edges"]) == set(sp["edges"])),
                         "n_pareto": r["n_pareto"]})
rt = pd.DataFrame(rows)
rt.to_csv(TAB / "route_metrics_long.csv", index=False)


def q(x):
    return f"{np.median(x):.1f} ({np.percentile(x, 25):.1f}–{np.percentile(x, 75):.1f})"


t2 = []
for t in PERIODS:
    for typ in ["SP", "B5", "B10", "KP", "MTEP"]:
        d = rt[(rt.time == t) & (rt.type == typ)]
        t2.append({"Period": t, "Route": typ, "Length (m)": q(d.len), "ΔL (%)": q(d.dL),
                   "Shade (%)": q(d.shade), "Sunlit distance (m)": q(d.dsun),
                   "Longest sunlit run (m)": q(d.run), "Mean Ts (°C)": q(d.lst),
                   "ΔC (%)": q(d.dC), "Changed (%)": f"{100 * (1 - d.same_as_sp.mean()):.0f}"})
pd.DataFrame(t2).to_csv(TAB / "table2_route_comparison.csv", index=False)

rq2 = {}
for t in PERIODS:
    d = rt[rt.time == t]
    e = {}
    for typ in ["B5", "B10", "KP", "MTEP"]:
        x = d[d.type == typ]; sp = d[d.type == "SP"]
        w = stats.wilcoxon(sp.dsun.values, x.dsun.values, zero_method="wilcox")
        nz = (sp.dsun.values - x.dsun.values) != 0
        z = stats.norm.isf(w.pvalue / 2)
        e[typ] = {"dL_med": float(x.dL.median()), "dL_mean": float(x.dL.mean()),
                  "dC_med": float(x.dC.median()), "dC_mean": float(x.dC.mean()), "dC_ci": boot_ci(x.dC),
                  "dDsun_med": float(x.dDsun.median()), "dDsun_mean": float(x.dDsun.mean()),
                  "dDsun_ci": boot_ci(x.dDsun), "dsun_med": float(x.dsun.median()),
                  "shade_med": float(x.shade.median()), "run_med": float(x.run.median()),
                  "dLST_med": float(x.dLST.median()), "dLST_mean": float(x.dLST.mean()),
                  "changed_share": float(1 - x.same_as_sp.mean()),
                  "wilcoxon_p": float(w.pvalue), "effect_r": float(z / np.sqrt(nz.sum())) if nz.sum() else 0.0}
    e["SP"] = {"len_med": float(d[d.type == "SP"].len.median()), "shade_med": float(d[d.type == "SP"].shade.median()),
               "dsun_med": float(d[d.type == "SP"].dsun.median()), "run_med": float(d[d.type == "SP"].run.median()),
               "lst_med": float(d[d.type == "SP"].lst.median())}
    e["n_pareto_med"] = float(d[d.type == "SP"].n_pareto.median())
    e["n_pareto_max"] = int(d[d.type == "SP"].n_pareto.max())
    e["single_solution_share"] = float((d[d.type == "SP"].n_pareto == 1).mean())
    rq2[t] = e
# Holm correction over the 16 Wilcoxon tests (4 route classes x 4 periods)
pv = sorted([(rq2[t][typ]["wilcoxon_p"], t, typ) for t in PERIODS for typ in ["B5", "B10", "KP", "MTEP"]])
m = len(pv); run = 0.0
for i, (pval, t, typ) in enumerate(pv):
    run = max(run, min(1.0, (m - i) * pval)); rq2[t][typ]["wilcoxon_p_holm"] = run
rq2["wilcoxon_p_holm_max"] = max(rq2[t][typ]["wilcoxon_p_holm"] for t in PERIODS for typ in ["B5", "B10", "KP", "MTEP"])
rq2["effect_r_range"] = [min(rq2[t][typ]["effect_r"] for t in PERIODS for typ in ["B5", "B10", "KP", "MTEP"]),
                         max(rq2[t][typ]["effect_r"] for t in PERIODS for typ in ["B5", "B10", "KP", "MTEP"])]
# spatial block bootstrap: resample 500 m origin cells to respect shared network edges
nodes_xy = pd.read_csv(PROC / "network_nodes.csv").set_index("node")
from routing import Net as _Net
_net = _Net(seg.u.values, seg.v.values, L)
oxy = nodes_xy.loc[_net.nodes[[o for o, _, _ in OD]]][["x", "y"]].values
import geopandas as _gpd
_x0, _y0 = _gpd.read_file(PROC / "aoi.gpkg").total_bounds[:2]     # grid anchored at the south-west corner of the study area
cell = (np.floor((oxy[:, 0] - _x0) / 500) * 1000 + np.floor((oxy[:, 1] - _y0) / 500)).astype(int)
cells = np.unique(cell); rq2["n_origin_cells"] = int(len(cells))
for t in PERIODS:
    for typ in ["B10", "KP"]:
        x = rt[(rt.time == t) & (rt.type == typ)].sort_values("od").dDsun.values
        bs = []
        for _ in range(2000):
            cs = rng.choice(cells, len(cells))
            idx = np.concatenate([np.where(cell == c)[0] for c in cs])
            bs.append(np.median(x[idx]))
        rq2[t][typ]["dDsun_ci_block"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    rq2[t]["B10"]["nonpositive_share"] = float((rt[(rt.time == t) & (rt.type == "B10")].dDsun <= 1e-9).mean())
    rq2[t]["B10"]["negative_share"] = float((rt[(rt.time == t) & (rt.type == "B10")].dDsun < -1e-9).mean())
kp = rt[rt.type == "KP"].pivot(index="od", columns="time", values="dC")
fr = stats.friedmanchisquare(*[kp[t] for t in PERIODS])
rq2["friedman_KP_dC"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}
b10 = rt[rt.type == "B10"].pivot(index="od", columns="time", values="dDsun")
fr = stats.friedmanchisquare(*[b10[t] for t in PERIODS])
rq2["friedman_B10_dDsun"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}
out["rq2"] = rq2
# illustrative O-D pair for Figure 4: four distinct routes at 15:00, rich front, SP 1.3-2.2 km
def _distinct(r):
    return len({frozenset(r[x]["edges"]) for x in ["SP", "KP", "B10", "MTEP"]}) == 4


cand = [k for k, (o, d, l) in enumerate(OD) if 1300 < l < 2200 and res["15:00"][k]["n_pareto"] >= 8
        and _distinct(res["15:00"][k])]
kx = max(cand, key=lambda i: (res["15:00"][i]["SP"]["dsun"] - res["15:00"][i]["B10"]["dsun"]) / res["15:00"][i]["SP"]["dsun"])
ex = {"index": int(kx), "n_pareto": {t: res[t][kx]["n_pareto"] for t in PERIODS}}
for t in PERIODS:
    sp = res[t][kx]["SP"]
    ex[t] = {typ: {"len": res[t][kx][typ]["len"], "dsun": res[t][kx][typ]["dsun"],
                   "dL": 100 * (res[t][kx][typ]["len"] / sp["len"] - 1),
                   "dC": 100 * (sp["cost"] - res[t][kx][typ]["cost"]) / sp["cost"]} for typ in ["SP", "KP", "B10", "MTEP"]}
out["example"] = ex
# representative pairs: B10 reduction at 15:00 closest to the 10th, 50th and 90th percentile of all pairs
d15 = np.array([100 * (1 - r["B10"]["dsun"] / r["SP"]["dsun"]) if r["SP"]["dsun"] > 0 else 0 for r in res["15:00"]])
chg = np.array([set(r["B10"]["edges"]) != set(r["SP"]["edges"]) for r in res["15:00"]])
pool = [k for k, (o, d, l) in enumerate(OD) if 1300 < l < 2200 and res["15:00"][k]["n_pareto"] >= 3 and chg[k]]
out["examples3"] = {}
for q_ in (10, 50, 90):
    tgt = np.percentile(d15[chg], q_)
    k = min(pool, key=lambda i: abs(d15[i] - tgt))
    r = res["15:00"][k]
    out["examples3"][f"p{q_}"] = {"index": int(k), "target": float(tgt), "dDsun": float(d15[k]),
                                  **{typ: {"len": r[typ]["len"], "dsun": r[typ]["dsun"],
                                           "dL": 100 * (r[typ]["len"] / r["SP"]["len"] - 1)} for typ in ["SP", "KP", "B10"]}}
out["od"] = {"n": len(OD), "sp_len_med": float(np.median([x[2] for x in OD])),
             "sp_len_iqr": [float(np.percentile([x[2] for x in OD], 25)), float(np.percentile([x[2] for x in OD], 75))]}

# trade-off curve: median exposure reduction vs detour budget
curve = []
for t in PERIODS:
    for eps in [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
        vals, dsv = [], []
        for r in res[t]:
            f = np.array(r["front"]); L0 = f[0, 0]; c0 = f[0, 1]
            ok = f[:, 0] <= (1 + eps) * L0 + 1e-6
            vals.append(100 * (c0 - f[ok, 1].min()) / c0)
        curve.append({"time": t, "eps": eps, "dC_med": float(np.median(vals)), "dC_mean": float(np.mean(vals)),
                      "dC_p75": float(np.percentile(vals, 75))})
pd.DataFrame(curve).to_csv(TAB / "budget_curve.csv", index=False)

# ------------------------------------------------------------------ RQ3 satellite-only baseline
rq3 = {}
for t in PERIODS:
    c2 = thermal_cost(L, S[t], seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])
    dd, dc, jj, same, dl, dlen = [], [], [], [], [], []
    for k, r in enumerate(res[t]):
        es = np.array(sat[k]["B10"], int)
        sp = r["SP"]
        ds = (L[es] * S[t][es]).sum()
        dd.append(100 * (sp["dsun"] - ds) / sp["dsun"] if sp["dsun"] > 0 else 0.0)
        dc.append(100 * (sp["cost"] - c2[es].sum()) / sp["cost"])
        jj.append(jac(es, r["B10"]["edges"]))
        same.append(set(es) == set(sp["edges"]))
        dl.append((L[es] * seg.lst.values[es]).sum() / L[es].sum() - sp["lst"])
        dlen.append(100 * (L[es].sum() / sp["len"] - 1))
    full = rt[(rt.time == t) & (rt.type == "B10")]
    w = stats.wilcoxon(full.dDsun.values, np.array(dd))
    rq3[t] = {"sat_dDsun_med": float(np.median(dd)), "sat_dDsun_mean": float(np.mean(dd)),
              "full_dDsun_med": float(full.dDsun.median()), "full_dDsun_mean": float(full.dDsun.mean()),
              "sat_dC_med": float(np.median(dc)), "full_dC_med": float(full.dC.median()),
              "jaccard_med": float(np.median(jj)), "jaccard_mean": float(np.mean(jj)),
              "sat_worse_than_sp_share": float(np.mean(np.array(dd) < -1e-9)),
              "sat_same_as_sp": float(np.mean(same)), "wilcoxon_p": float(w.pvalue),
              "sat_dLST_med": float(np.median(dl)), "sat_dLST_mean": float(np.mean(dl)),
              "sat_dLST_p10": float(np.percentile(dl, 10)), "sat_dL_med": float(np.median(dlen)),
              "full_dLST_mean": float(full.dLST.mean())}
    dif = full.sort_values("od").dDsun.values - np.array(dd)
    rq3[t].update({"paired_diff_med": float(np.median(dif)), "paired_diff_p25": float(np.percentile(dif, 25)),
                   "paired_diff_p75": float(np.percentile(dif, 75)), "shade_better_share": float(np.mean(dif > 1e-9)),
                   "equal_share": float(np.mean(np.abs(dif) <= 1e-9)), "sat_better_share": float(np.mean(dif < -1e-9)),
                   "full_worse_than_sp_share": float(np.mean(full.dDsun.values < -1e-9)),
                   "sat_dDsun_p25": float(np.percentile(dd, 25)), "sat_dDsun_p75": float(np.percentile(dd, 75))})
# temporal instability of full-model routes
tj = {}
for a_ in PERIODS:
    for b_ in PERIODS:
        if a_ < b_:
            tj[f"{a_}-{b_}"] = float(np.median([jac(res[a_][k]["B10"]["edges"], res[b_][k]["B10"]["edges"])
                                                for k in range(len(OD))]))
rq3["B10_jaccard_between_periods_med"] = tj
# share of O-D pairs whose B10 route is identical in all four periods
rq3["B10_identical_all_periods"] = float(np.mean([len({frozenset(res[t][k]["B10"]["edges"]) for t in PERIODS}) == 1
                                                  for k in range(len(OD))]))
out["rq3"] = rq3

# ------------------------------------------------------------------ weights / penalty sensitivity
t3 = []
for (name, beta, t), lst in sens.items():
    dd, jj, kd, rn = [], [], [], []
    for k, x in enumerate(lst):
        sp = x["SP"]; es = x["B10"]
        dsp = (L[sp] * S[t][sp]).sum(); db = (L[es] * S[t][es]).sum()
        dd.append(100 * (dsp - db) / dsp if dsp > 0 else 0)
        jj.append(jac(es, res[t][k]["B10"]["edges"]))
        kd.append(100 * (x["KP_len"] / L[sp].sum() - 1))
        rn.append(run_metrics(x["KP"], L, S[t]))
    t3.append({"weights": name, "beta": beta, "time": t, "w": WEIGHTS[name],
               "B10_dDsun_med": float(np.median(dd)), "B10_dDsun_mean": float(np.mean(dd)),
               "B10_jaccard_vs_balanced_med": float(np.median(jj)), "KP_dL_med": float(np.median(kd)),
               "KP_run_med": float(np.median(rn)), "KP_run_p90": float(np.percentile(rn, 90))})
    t3[-1].update({"_kd": kd, "_rn": rn, "_jj": jj})
pooled = {}
for r in t3:
    key = (r["weights"], r["beta"])
    pooled.setdefault(key, {"kd": [], "rn": [], "jj": []})
    pooled[key]["kd"] += r.pop("_kd"); pooled[key]["rn"] += r.pop("_rn"); pooled[key]["jj"] += r.pop("_jj")
out["t3_pooled"] = {f"{w}|{b:g}": {"KP_dL_med": float(np.median(v["kd"])), "KP_run_med": float(np.median(v["rn"])),
                                   "J_med": float(np.median(v["jj"])), "J_p25": float(np.percentile(v["jj"], 25)),
                                   "J_p75": float(np.percentile(v["jj"], 75)),
                                   "identical": float(np.mean(np.array(v["jj"]) >= 0.999999))}
                    for (w, b), v in pooled.items()}
t3 = pd.DataFrame(t3)
t3.to_csv(TAB / "table3_sensitivity.csv", index=False)

# ------------------------------------------------------------------ RQ4 height uncertainty
try:
    M = pickle.load(open(PROC / "height_mc.pkl", "rb"))
    rq4 = {"r_med_logratio": M["r_med"], "ratio_med": float(np.exp(M["r_med"])), "n_ref": M["n_ref"]}
    for t in PERIODS:
        base_s = float((L * S[t]).sum() / L.sum())
        ns = [m[t]["net_sunlit"] for m in M["mc"]]
        J = np.concatenate([m[t]["rows"][:, 0] for m in M["mc"]])
        red = [np.median(m[t]["rows"][:, 4]) * 100 for m in M["mc"]]
        # relative change of baseline-route sunlit distance under perturbed shade
        b = M["bias_scenario"][t]
        rq4[t] = {"base_net_sunlit": base_s, "mc_net_sunlit_mean": float(np.mean(ns)),
                  "mc_net_sunlit_range": [float(min(ns)), float(max(ns))],
                  "mc_jaccard_med": float(np.median(J)), "mc_jaccard_p10": float(np.percentile(J, 10)),
                  "mc_jaccard_ge09_share": float(np.mean(J >= 0.9)), "mc_identical_share": float(np.mean(J >= 0.999)),
                  "mc_B10_dDsun_med_range": [float(min(red)), float(max(red))],
                  "mc_B10_dDsun_med_mean": float(np.mean(red)),
                  "bias_net_sunlit": b["net_sunlit"], "bias_jaccard_med": float(np.median(b["rows"][:, 0])),
                  "bias_identical_share": float(np.mean(b["rows"][:, 0] >= 0.999)),
                  "bias_B10_dDsun_med": float(np.median(b["rows"][:, 4]) * 100),
                  # does the baseline cool route remain cooler than SP in perturbed fields?
                  "mc_baseline_route_still_better_share": float(np.mean(np.concatenate(
                      [m[t]["rows"][:, 3] <= m[t]["rows"][:, 1] + 1e-9 for m in M["mc"]])))}
    # convergence of the independent run: running median over realisations
    conv = {}
    for t in PERIODS:
        red = np.array([np.median(m[t]["rows"][:, 4]) * 100 for m in M["mc"]])
        jm = np.array([np.median(m[t]["rows"][:, 0]) for m in M["mc"]])
        run_red = [np.median(red[:n]) for n in range(1, len(red) + 1)]
        run_j = [np.median(jm[:n]) for n in range(1, len(jm) + 1)]
        conv[t] = {"red_final": float(run_red[-1]), "red_maxdev_after15": float(np.max(np.abs(np.array(run_red[14:]) - run_red[-1]))),
                   "j_maxdev_after15": float(np.max(np.abs(np.array(run_j[14:]) - run_j[-1]))),
                   "running_red": [float(x) for x in run_red]}
    rq4["convergence"] = conv
    try:
        D = pickle.load(open(PROC / "height_mc_dependent.pkl", "rb"))
        rq4["dependent"] = {"n_groups": D["n_groups"], "n_blocks": D["n_blocks"],
                            "multiplier_mean": D["multiplier_mean"], "multiplier_p5_p95": D["multiplier_p5_p95"]}
        for sc in ["grouped", "block"]:
            rq4["dependent"][sc] = {}
            for t in PERIODS:
                J = np.concatenate([m[t]["rows"][:, 0] for m in D[sc]])
                red = [np.median(m[t]["rows"][:, 4]) * 100 for m in D[sc]]
                ns = [m[t]["net_sunlit"] for m in D[sc]]
                rq4["dependent"][sc][t] = {"jaccard_med": float(np.median(J)), "jaccard_p10": float(np.percentile(J, 10)),
                                           "identical": float(np.mean(J >= 0.999)),
                                           "red_range": [float(min(red)), float(max(red))], "red_med": float(np.median(red)),
                                           "net_sunlit_range": [float(min(ns)), float(max(ns))]}
    except FileNotFoundError:
        pass
    for t in PERIODS:
        J = np.concatenate([m[t]["rows"][:, 0] for m in M["mc"]])
        rq4[t]["mc_jaccard_p25"] = float(np.percentile(J, 25)); rq4[t]["mc_jaccard_p75"] = float(np.percentile(J, 75))
        rq4[t]["mc_changed_share"] = float(np.mean(J < 0.999))
    out["rq4"] = rq4
except FileNotFoundError:
    pass

# ------------------------------------------------------------------ RQ5 intervention
iv = pd.read_csv(TAB / "intervention_scenarios.csv")
out["rq5"] = iv.round(3).to_dict(orient="records")
import geopandas as gpd
g =gpd.read_file(PROC / "thermal_betweenness.gpkg")
o = np.argsort(-g.flow.values); cum = np.cumsum(g.length.values[o]) / g.length.sum()
trip_km = g.flow.values * g.length.values
out["flow_concentration"] = {f"top{int(p*100)}pct_len_share_of_tripkm":
                             float(trip_km[o[: np.searchsorted(cum, p) + 1]].sum() / trip_km.sum())
                             for p in [0.01, 0.03, 0.05, 0.10]}
o2 = np.argsort(-g.tb.values); cum2 = np.cumsum(g.length.values[o2]) / g.length.sum()
top5 = o2[: np.searchsorted(cum2, 0.05) + 1]
out["tb_top5"] = {"s_day_mean": float(np.average(g.s_day.values[top5], weights=g.length.values[top5])),
                  "net_s_day_mean": float(np.average(g.s_day.values, weights=g.length.values)),
                  "highway_len_share": (seg.iloc[top5].groupby("highway").length.sum()
                                        / seg.iloc[top5].length.sum()).round(3).to_dict()}
for f in ["surface_stats.json", "network_stats.json", "satellite_meta.json", "intervention_base.json",
          "routing_checks.json", "shade_checks.json", "priority_checks.json", "height_checks.json"]:
    if (PROC / f).exists():
        out[f.replace(".json", "")] = json.load(open(PROC / f))
# lowest sun of the day (hourly 07:00-19:00)
lo = min(HOURLY, key=lambda t: out["network_stats"]["sun"][t]["elev"])
out["lowest_sun"] = {"time": lo, "elev": out["network_stats"]["sun"][lo]["elev"],
                     "fully_sunlit_share_len": out["rq1"][lo]["fully_sunlit_share_len"], "all": out["rq1"][lo]["all"]}
out["orientation_len_share"] = out["orientation_len_share"]
json.dump(out, open(ROOT / "outputs" / "results_summary.json", "w"), indent=1, default=float)
print("ok")


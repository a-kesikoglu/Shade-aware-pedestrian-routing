"""Compact manuscript tables (CSV) built from the experiment outputs.
Table3 route metrics, Table4 sensitivity of B10 routes, Table5 interventions, TableS* supplementary."""
import json
import numpy as np
import pandas as pd
from config import *

S = json.load(open(ROOT / "outputs" / "results_summary.json"))
rt = pd.read_csv(TAB / "route_metrics_long.csv")
f = lambda x, d=0: f"{np.median(x):,.{d}f}"
iqr = lambda x, d=1: f"{np.median(x):.{d}f} ({np.percentile(x, 25):.{d}f} to {np.percentile(x, 75):.{d}f})"
neg = lambda s: s.replace("-", "−")

# ---------------------------------------------------------------- Table 3: route metrics
rows = []
for t in PERIODS:
    for typ in ["SP", "B5", "B10", "KP", "MTEP"]:
        d = rt[(rt.time == t) & (rt.type == typ)]
        ci = S["rq2"][t][typ]["dDsun_ci"] if typ != "SP" else None
        rows.append({"Period": t, "Route": typ, "L (m)": f(d.len), "ΔL (%)": f(d.dL, 1),
                     "Dsun (m)": f(d.dsun), "Rmax (m)": f(d.run), "ΔC (%)": f(d.dC, 1),
                     "ΔDsun (%)": "–" if typ == "SP" else neg(iqr(d.dDsun)),
                     "95% CI": "–" if typ == "SP" else neg(f"{ci[0]:.1f} to {ci[1]:.1f}"),
                     "Changed (%)": "–" if typ == "SP" else f"{100 * (1 - d.same_as_sp.mean()):.0f}"})
pd.DataFrame(rows).to_csv(TAB / "Table3.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- Table 4: sensitivity of B10 routes
s = pd.read_csv(TAB / "table3_sensitivity.csv")
pool = S["t3_pooled"]
rows = []


def add(group, name, red, jac, ident):
    r = {"Group": group, "Variant": name}
    for t in PERIODS:
        r[t] = f"{red[t]:.1f}"
    r["J"] = "–" if jac is None else f"{jac:.2f}"
    r["Identical (%)"] = "–" if ident is None else f"{100 * ident:.0f}"
    rows.append(r)


lab = {"balanced": "Balanced (reference)", "shade": "Shade dominant (0.6, 0.2, 0.2)",
       "vegetation": "Vegetation dominant (0.2, 0.6, 0.2)", "lst": "LST dominant (0.2, 0.2, 0.6)",
       "shade_only": "Shade only (1, 0, 0)"}
for (name, beta), g in s.groupby(["weights", "beta"], sort=False):
    red = {t: g[g.time == t].B10_dDsun_med.iloc[0] for t in PERIODS}
    p = pool[f"{name}|{beta:g}"]
    grp = "Weights" if beta == BETA else "Penalty"
    nm = lab[name] if beta == BETA else f"β = {beta:g}"
    ref = name == "balanced" and beta == BETA
    add("Reference" if ref else grp, nm, red, None if ref else g.B10_jaccard_vs_balanced_med.mean(),
        None if ref else p["identical"])
V = S["shade_checks"]["variants"]
vl = {"trunk 0.10": ("Canopy", "Crown base 10% of height"), "trunk 0.40": ("Canopy", "Crown base 40% of height"),
      "tau_c 0.00": ("Canopy", "Transmissivity 0"), "tau_c 0.20": ("Canopy", "Transmissivity 0.20"),
      "canopy h >= 2.0 m": ("Canopy", "Minimum height 2.0 m"), "canopy h >= 3.0 m": ("Canopy", "Minimum height 3.0 m"),
      "no NDVI filter": ("Canopy", "No NDVI filter"), "NDVI >= 0.25": ("Canopy", "NDVI filter 0.25"),
      "buildings only": ("Canopy", "No canopy"), "terrain": ("Ground", "Terrain from GLO-30")}
for k, (gname, nm) in vl.items():
    v = V[k]
    add(gname, nm, {t: v[t]["dDsun_med"] for t in PERIODS}, np.mean([v[t]["jaccard_med"] for t in PERIODS]),
        np.mean([v[t]["identical"] for t in PERIODS]))
SC = S["shade_checks"]
for key, nm in [("facade_run_shadier", "Facade side, shadier side"), ("facade_run_mean", "Facade side, mean of sides"),
                ("tunnel_run", "Pedestrian underpasses open")]:
    v = SC[key]
    add("Corridor" if "facade" in key else "Network", nm, {t: v[t]["dDsun_med"] for t in PERIODS},
        np.mean([v[t]["jaccard_med"] for t in PERIODS]), np.mean([v[t]["identical"] for t in PERIODS]))
ext = SC["extent"]["by_time"]
add("Network", "Extent plus 300 m", {t: ext[t]["dDsun_ext_med"] for t in PERIODS}, None, None)
RV = {k: json.load(open(PROC / f"{k}.json")) for k in ["rev_routing", "rev_canopy", "rev_lst"]}
dm = RV["rev_routing"]["dsun_min"]
add("Weights", "Direct minimisation of Dsun (1, 0, 0), β = 0", {t: dm[t]["dDsun_min_med"] for t in PERIODS},
    np.mean([dm[t]["J_B10_min_med"] for t in PERIODS]), np.mean([dm[t]["B10_equals_min_share"] for t in PERIODS]))
cn = RV["rev_canopy"]["network"]
add("Canopy", "Ray test only under crowns", {t: cn[t]["B10_dDsun_med"] for t in PERIODS}, None,
    np.mean([cn[t]["B10_same_share"] for t in PERIODS]))
ls = RV["rev_lst"]["single_scene"]
add("Surface temperature", f"Single scene of {ls['date'][8:10].lstrip('0')} July 2025", {t: ls[t]["B10_dDsun_med"] for t in PERIODS}, None,
    np.mean([ls[t]["B10_same_share"] for t in PERIODS]))
_go = ["Reference", "Weights", "Penalty", "Canopy", "Ground", "Surface temperature", "Corridor", "Network"]
rows = sorted(rows, key=lambda r: _go.index(r["Group"]))
T4 = pd.DataFrame(rows)
T4.columns = ["Group", "Variant"] + [f"ΔDsun {t}" for t in PERIODS] + ["J", "Identical (%)"]
T4.to_csv(TAB / "Table4.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- Table 5: interventions
iv = pd.read_csv(TAB / "intervention_scenarios.csv")
ex = pd.read_csv(TAB / "intervention_extended.csv")
rnd = pd.read_csv(TAB / "rev_random100.csv")
lab = {"TB": "Thermal betweenness", "gain": "Direct gain ranking", "flow": "Pedestrian flow",
       "sun": "Daily sunlit fraction", "random": "Random"}
rows = []
for frac in sorted(iv.frac.unique()):
    prev_km, prev_red = None, None
    for st in ["TB", "gain", "flow", "sun", "random"]:
        if st == "gain":
            d = ex[(ex.frac == frac) & (ex.strategy == "gain")].iloc[0]
        else:
            d = iv[(iv.frac == frac) & (iv.strategy == st)].iloc[0]
        red = d.red_dsun_sp_day
        cell = f"{red:.1f}"
        if st == "random":
            r = rnd[rnd.frac == frac]
            red = r.red_dsun_sp_day.mean()
            cell = f"{red:.1f} ({r.red_dsun_sp_day.quantile(.025):.1f} to {r.red_dsun_sp_day.quantile(.975):.1f})"
            d = pd.Series({"red_cost_sp_day": r.red_cost_sp_day.mean(), "red_dsun_b10_15": r.red_dsun_b10_15.mean(),
                           "km": r.km.mean()})
        km_ = d.km
        rows.append({"Treated": f"{100 * frac:g}%", "Strategy": lab[st], "Length (km)": f"{km_:.1f}",
                     "ΔDday, SP (%)": cell, "ΔCday, SP (%)": f"{d.red_cost_sp_day:.1f}",
                     "ΔDsun, B10 15:00 (%)": f"{d.red_dsun_b10_15:.1f}",
                     "Mean per km (pp/km)": f"{red / km_:.2f}"})
T5 = pd.DataFrame(rows)
tb = iv[iv.strategy == "TB"].sort_values("frac")
km = np.r_[0, tb.km.values]; rd = np.r_[0, tb.red_dsun_sp_day.values]
inc = np.diff(rd) / np.diff(km)
T5["Added per km (pp/km)"] = "–"
for i, frac in enumerate(sorted(iv.frac.unique())):
    T5.loc[(T5.Treated == f"{100 * frac:g}%") & (T5.Strategy == lab["TB"]), "Added per km (pp/km)"] = f"{inc[i]:.2f}"
T5.to_csv(TAB / "Table5.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- paired tests (metres)
pt = pd.read_csv(TAB / "rev_paired_tests.csv")
TT = pd.DataFrame([{"Period": r.time, "Route": r.route, "n": f"{r.n_nonzero:.0f}", "W+": f"{r.W_plus:,.0f}",
                    "z": f"{r.z:.1f}", "r": f"{r.r:.2f}", "Holm p": "< 0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}",
                    "Median difference (m)": neg(f"{r.diff_med_m:.0f} ({r.diff_p25_m:.0f} to {r.diff_p75_m:.0f})"),
                    "Mean difference (m)": f"{r.diff_mean_m:.0f}", "Median extra length (m)": f"{r.extra_len_med_m:.0f}"}
                   for r in pt.itertuples()])
TT.to_csv(TAB / "TableTests.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- height uncertainty by error model and period
un = pd.read_csv(TAB / "rev_uncertainty.csv")
RH = json.load(open(PROC / "rev_heights.json"))
ref = un.groupby("time").ref.first()
rows = [{"Scenario": "Reference heights", **{t: f"{ref[t]:.1f}" for t in PERIODS}, "Max. shift (pp)": "–"}]
for m, g in un.groupby("model", sort=False):
    g = g.set_index("time")
    rows.append({"Scenario": f"Random, {m} (range of 30)",
                 **{t: f"{g.loc[t, 'min']:.1f} to {g.loc[t, 'max']:.1f}" for t in PERIODS},
                 "Max. shift (pp)": f"{g.max_shift.max():.1f}"})
bias = {"Bias, uniform factor (storeys 3.0 m)": S["rq4"]}
rows.append({"Scenario": f"Bias, uniform factor {RH['all']['ratio_med']:.2f}",
             **{t: f"{S['rq4'][t]['bias_B10_dDsun_med']:.1f}" for t in PERIODS},
             "Max. shift (pp)": f"{max(abs(S['rq4'][t]['bias_B10_dDsun_med'] - ref[t]) for t in PERIODS):.1f}"})
if "hd_bias" in RH:
    rows.append({"Scenario": "Bias, class-specific factors", **{t: f"{RH['hd_bias'][t]['B10_dDsun_med']:.1f}" for t in PERIODS},
                 "Max. shift (pp)": f"{max(abs(RH['hd_bias'][t]['B10_dDsun_med'] - ref[t]) for t in PERIODS):.1f}"})
    for f_, v in RH["bias_f"].items():
        if f_ == "3.0":
            continue
        fac = RH["storey_factor"][f_]["ratio_med"]
        rows.append({"Scenario": f"Bias, storeys of {f_} m (factor {fac:.2f})", **{t: f"{v[t]['B10_dDsun_med']:.1f}" for t in PERIODS},
                     "Max. shift (pp)": f"{max(abs(v[t]['B10_dDsun_med'] - ref[t]) for t in PERIODS):.1f}"})
TU = pd.DataFrame(rows)
TU.to_csv(TAB / "TableUnc.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- supplementary tables
meta = S["satellite_meta"]
pd.DataFrame([{"Scene": x["id"], "Platform": x["platform"].replace("landsat-", "Landsat "),
               "Acquisition (UTC)": x["datetime"][:16].replace("T", " "), "Scene cloud cover (%)": f"{x['cloud_cover']:.2f}"}
              for x in meta["landsat"]]).to_csv(TAB / "TableS1.csv", index=False, encoding="utf-8-sig")
FC = SC["facade_coverage"]
pd.DataFrame([{"Street class": {"all": "All segments"}.get(c, c.capitalize()),
               "Facades on both sides (%)": f"{100 * v['both']:.0f}", "Facade on one side (%)": f"{100 * v['one']:.0f}",
               "No facade within 30 m (%)": f"{100 * v['none']:.0f}"} for c, v in FC.items()])     .to_csv(TAB / "TableS2a.csv", index=False, encoding="utf-8-sig")
rows = []
for t in PERIODS:
    for c, v in SC["facade"][t].items():
        rows.append({"Period": t, "Street class": c.capitalize(), "Length share (%)": f"{100 * v['len_share_class']:.0f}",
                     "Both facades (%)": f"{100 * v['both_sides_share']:.0f}",
                     "Centreline s": f"{v['s_center']:.2f}", "Mean of sides s": f"{v['s_sides_mean']:.2f}",
                     "Shadier side s": f"{v['s_shadier_side']:.2f}",
                     "Sides differ > 0.5 (%)": f"{100 * v['sides_differ_gt05']:.0f}"})
pd.DataFrame(rows).to_csv(TAB / "TableS2.csv", index=False, encoding="utf-8-sig")
RC = S["routing_checks"]
rows = []
for k, v in RC["rmax"].items():
    b, th = k.replace("beta", "").split("_thr")
    rows.append({"β": b, "Threshold": th, "Median change B10 vs SP (m)": f"{v['B10_minus_SP_med']:.0f}",
                 "Mean change B10 vs SP (m)": neg(f"{v['B10_minus_SP_mean']:.0f}"),
                 "B10 shorter run (%)": f"{100 * v['B10_shorter_share']:.0f}",
                 "Mean change KP vs SP (m)": neg(f"{v['KP_minus_SP_mean']:.0f}")})
pd.DataFrame(rows).to_csv(TAB / "TableS3.csv", index=False, encoding="utf-8-sig")
rows = []
for cap, d in RC["detour_cap"].items():
    for r, v in d.items():
        rows.append({"Detour limit": f"{100 * (float(cap) - 1):.0f}%", "Route": r,
                     "Identical to 50% limit (%)": f"{100 * v['same_share']:.0f}",
                     "Median ΔDsun (%)": f"{v['dDsun_med']:.1f}", "Median ΔL (%)": f"{v['dL_med']:.1f}"})
pd.DataFrame(rows).to_csv(TAB / "TableS4.csv", index=False, encoding="utf-8-sig")
PC = S["priority_checks"]
rows = []
for k, v in PC["demand"].items():
    rows.append({"Demand model": {"uniform": "Uniform nodes", "poi": "Points of interest", "gfa_poi": "Floor area and points of interest"}[k],
                 "Top 5% overlap (J)": f"{v['top5_jaccard_vs_ref']:.2f}", "Busiest 5% share of trip distance (%)": f"{100 * v['top5pct_len_tripkm']:.0f}",
                 "TB 5% (%)": f"{v['red_TB5']:.1f}", "Reference selection (%)": f"{v['red_ref_selection5']:.1f}",
                 "Flow 5% (%)": f"{v['red_flow5']:.1f}", "Random 5% (%)": f"{v['red_random5']:.1f}",
                 **{f"B10 {t}": f"{v['B10_dDsun_med'][t]:.1f}" for t in PERIODS}})
pd.DataFrame(rows).to_csv(TAB / "TableS5.csv", index=False, encoding="utf-8-sig")
print(T4.to_string()); print(T5.to_string())

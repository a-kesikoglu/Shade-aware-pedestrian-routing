"""Robustness of thermal betweenness and of the shade intervention scenarios.

1  trip-distance concentration curve
2  strategies incl. a direct gain ranking and 100 random draws
3  physical intervention types (structural shade, mature trees, establishment phase)
4  temporal demand weights (equal, morning, evening)
5  OD bootstrap of the 5000 flow pairs and an independent flow sample: overlap of priority sets
6  alternative demand models (uniform nodes, OSM points of interest)
7  biased building heights
Outputs: data/processed/priority_checks.json, outputs/tables/concentration_curve.csv,
         outputs/tables/intervention_extended.csv
"""
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from scipy.spatial import cKDTree
from config import *
from impedance import load_segments, thermal_cost
from routing import Net, pareto_paths, budget
from shade import sun_positions, cast
from sampling import corridor_points, edge_mean, edge_sunlit

seg = load_segments()
L = seg.length.values
net = Net(seg.u.values, seg.v.values, L)
ods = pickle.load(open(PROC / "od_sets.pkl", "rb"))
OD, OD_FLOW, WN = ods["OD"], ods["OD_FLOW"], ods["node_w"]
HRS = HOURLY
SH = np.stack([seg[f"s_{t.replace(':', '')}"].values for t in HRS])
a_nd, b_nd = np.nanpercentile(seg.ndvi, [2, 98])
OUT = {}


def sp_edges(o, d):
    _, pred = net.sssp(L, o, pred=True)
    return net.path_edges(pred, o, d)


def count_edges(paths):
    c = np.zeros(len(L))
    for p in paths:
        c[p] += 1
    return c


FLOW_P = [sp_edges(o, d) for o, d, _ in OD_FLOW]
EVAL_P = [sp_edges(o, d) for o, d, _ in OD]
FLOW = count_edges(FLOW_P)
EVAL_C = count_edges(EVAL_P)


def select(score, frac, rng=None):
    order = rng.permutation(len(L)) if rng is not None else \
        np.argsort(-(score + np.random.default_rng(0).random(len(L)) * 1e-9))
    cum = np.cumsum(L[order])
    return order[: np.searchsorted(cum, frac * L.sum()) + 1]


def day_metrics(treated, SH_=SH, ec=EVAL_C, s_cap=0.25, nd_add=0.25, omega=None):
    omega = np.ones(len(HRS)) if omega is None else np.asarray(omega, float)
    s = SH_.copy()
    s[:, treated] = np.minimum(s[:, treated], s_cap)
    nd = seg.ndvi.values.copy(); nd[treated] += nd_add
    ndn = np.clip((nd - a_nd) / (b_nd - a_nd), 0, 1)
    dsun = float((omega[:, None] * s * (ec * L)[None, :]).sum())
    cost = 0.0
    for h in range(len(HRS)):
        cost += omega[h] * float((ec * thermal_cost(L, s[h], ndn, seg.lst_n.values, WEIGHTS["balanced"])).sum())
    return dsun, cost


def b10_15(treated, s_cap=0.25, nd_add=0.25):
    h = HRS.index("15:00")
    s = SH[h].copy(); s[treated] = np.minimum(s[treated], s_cap)
    nd = seg.ndvi.values.copy(); nd[treated] += nd_add
    ndn = np.clip((nd - a_nd) / (b_nd - a_nd), 0, 1)
    c2 = thermal_cost(L, s, ndn, seg.lst_n.values, WEIGHTS["balanced"])
    db = 0.0
    for o, d, _ in OD:
        sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
        es = sols[budget(sols, L0, 0.10)][2]; db += (L[es] * s[es]).sum()
    return db


def jac_len(A, B):
    A, B = set(A), set(B)
    return float(L[list(A & B)].sum() / L[list(A | B)].sum())


BASE_D, BASE_C = day_metrics(np.array([], int))
BASE_B = b10_15(np.array([], int))
s_day = SH.mean(0)
TB = FLOW * s_day
GAIN = FLOW * np.maximum(SH - 0.25, 0).mean(0)

# ---------------------------------------------------------------- 1 concentration curve
o = np.argsort(-FLOW); cum_len = np.cumsum(L[o]) / L.sum(); cum_trip = np.cumsum((FLOW * L)[o]) / (FLOW * L).sum()
step = max(1, len(o) // 2000)
pd.DataFrame({"len_share": np.r_[0, cum_len[::step], 1], "tripkm_share": np.r_[0, cum_trip[::step], 1]}) \
    .to_csv(TAB / "concentration_curve.csv", index=False)
OUT["zero_flow_len_share"] = float(L[FLOW == 0].sum() / L.sum())

# ---------------------------------------------------------------- 2 strategies
rows = []
rng = np.random.default_rng(500)
for frac in [0.01, 0.03, 0.05, 0.10]:
    for name, sc in {"TB": TB, "gain": GAIN, "flow": FLOW, "sun": s_day}.items():
        tr = select(sc, frac)
        d, c = day_metrics(tr)
        rows.append({"strategy": name, "frac": frac, "km": L[tr].sum() / 1000, "red_dsun_sp_day": 100 * (1 - d / BASE_D),
                     "red_cost_sp_day": 100 * (1 - c / BASE_C),
                     "red_dsun_b10_15": 100 * (1 - b10_15(tr) / BASE_B) if name in ("TB", "gain") else np.nan})
    sel = [select(None, frac, rng) for _ in range(100)]
    rr = [day_metrics(x)[0] for x in sel]
    red = 100 * (1 - np.array(rr) / BASE_D)
    rows.append({"strategy": "random100", "frac": frac, "km": float(np.mean([L[x].sum() for x in sel]) / 1000),
                 "km_target": frac * L.sum() / 1000, "red_dsun_sp_day": float(red.mean()),
                 "red_p2_5": float(np.percentile(red, 2.5)), "red_p97_5": float(np.percentile(red, 97.5)),
                 "red_min": float(red.min()), "red_max": float(red.max())})
    print(frac, "strategies", flush=True)
ext = pd.DataFrame(rows)
ext.to_csv(TAB / "intervention_extended.csv", index=False)
OUT["top5_overlap_TB_gain"] = jac_len(select(TB, .05), select(GAIN, .05))
OUT["top5_overlap_TB_flow"] = jac_len(select(TB, .05), select(FLOW, .05))

# ---------------------------------------------------------------- 3 intervention types (TB 5%)
tr5 = select(TB, 0.05)
types = {"structural": dict(s_cap=0.25, nd_add=0.0), "mature trees": dict(s_cap=0.25, nd_add=0.25),
         "establishment": dict(s_cap=0.60, nd_add=0.10)}
OUT["types"] = {}
for k, kw in types.items():
    d, c = day_metrics(tr5, **kw)
    OUT["types"][k] = {"red_dsun_sp_day": 100 * (1 - d / BASE_D), "red_cost_sp_day": 100 * (1 - c / BASE_C),
                       "red_dsun_b10_15": 100 * (1 - b10_15(tr5, **kw) / BASE_B)}
print("types", OUT["types"], flush=True)

# ---------------------------------------------------------------- 4 temporal demand weights
hr = np.array([int(t[:2]) for t in HRS])
prof = {"equal": np.ones(len(HRS)), "morning": np.where(hr <= 10, 2.0, 1.0), "evening": np.where(hr >= 16, 2.0, 1.0)}
OUT["temporal"] = {}
for k, w in prof.items():
    b0, _ = day_metrics(np.array([], int), omega=w)
    r = {}
    for name, sc in {"TB": TB, "flow": FLOW}.items():
        r[name] = 100 * (1 - day_metrics(select(sc, .05), omega=w)[0] / b0)
    rr = [100 * (1 - day_metrics(select(None, .05, np.random.default_rng(900 + j)), omega=w)[0] / b0) for j in range(20)]
    r["random_mean"] = float(np.mean(rr))
    OUT["temporal"][k] = r

# ---------------------------------------------------------------- 5 OD bootstrap and independent flow sample
rng = np.random.default_rng(77)
ref = {f: select(TB, f) for f in (0.01, 0.03, 0.05)}
boot = {f: [] for f in ref}; diff = []
for b in range(200):
    idx = rng.integers(0, len(FLOW_P), len(FLOW_P))
    fl = count_edges([FLOW_P[i] for i in idx])
    tb = fl * s_day
    for f in ref:
        boot[f].append(jac_len(ref[f], select(tb, f)))
    diff.append(100 * (day_metrics(select(fl, .05))[0] - day_metrics(select(tb, .05))[0]) / BASE_D)
OUT["bootstrap"] = {f"{f:g}": {"jaccard_med": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                               "p95": float(np.percentile(v, 95))} for f, v in boot.items()}
OUT["bootstrap_TB_minus_flow_pp"] = {"med": float(np.median(diff)), "p2_5": float(np.percentile(diff, 2.5)),
                                     "p97_5": float(np.percentile(diff, 97.5)), "share_pos": float(np.mean(np.array(diff) > 0))}
print("bootstrap", OUT["bootstrap"], OUT["bootstrap_TB_minus_flow_pp"], flush=True)


def sample_od(p, n, seed, dmin=500, dmax=2500):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        o, d = rng.choice(net.N, 2, p=p)
        if o == d:
            continue
        l0 = net.sssp(L, o)[d]
        if dmin <= l0 <= dmax:
            out.append((int(o), int(d), float(l0)))
    return out


p_ref = WN / WN.sum()
indep = sample_od(p_ref, 5000, SEED + 2)
fl2 = count_edges([sp_edges(o, d) for o, d, _ in indep])
OUT["independent_flow_sample"] = {f"{f:g}": jac_len(ref[f], select(fl2 * s_day, f)) for f in ref}

# ---------------------------------------------------------------- 6 alternative demand models
nodes = pd.read_csv(PROC / "network_nodes.csv").set_index("node").loc[net.nodes]
xy = nodes[["x", "y"]].values
bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
bc = bld.geometry.centroid
nb = cKDTree(xy).query_ball_point(np.c_[bc.x, bc.y], r=50)
OUT["gfa_nodes_per_building"] = {"mean": float(np.mean([len(x) for x in nb])),
                                 "median": float(np.median([len(x) for x in nb])),
                                 "share_zero": float(np.mean([len(x) == 0 for x in nb]))}
OUT["zero_weight_nodes_share"] = float(np.mean(WN <= 1.0))
poi = gpd.read_file(RAW / "osm_pois.gpkg")
pw = np.array([len(x) for x in cKDTree(np.c_[poi.geometry.x, poi.geometry.y]).query_ball_point(xy, r=50)]) + 0.1
schemes = {"uniform": np.ones(net.N) / net.N, "poi": pw / pw.sum(),
           "gfa_poi": (p_ref / p_ref.max() + pw / pw.max()) / (p_ref / p_ref.max() + pw / pw.max()).sum()}
S4 = {t: seg[f"s_{t.replace(':', '')}"].values for t in PERIODS}
OUT["n_pois"] = int(len(poi))
OUT["demand"] = {}
for name, p in schemes.items():
    ev = sample_od(p, 500, SEED + 20); fl_od = sample_od(p, 5000, SEED + 21)
    fl = count_edges([sp_edges(o, d) for o, d, _ in fl_od])
    ec = count_edges([sp_edges(o, d) for o, d, _ in ev])
    tb = fl * s_day
    b0 = day_metrics(np.array([], int), ec=ec)[0]
    red_tb = 100 * (1 - day_metrics(select(tb, .05), ec=ec)[0] / b0)
    red_fl = 100 * (1 - day_metrics(select(fl, .05), ec=ec)[0] / b0)
    red_rd = np.mean([100 * (1 - day_metrics(select(None, .05, np.random.default_rng(300 + j)), ec=ec)[0] / b0)
                      for j in range(20)])
    red_ref_sel = 100 * (1 - day_metrics(ref[0.05], ec=ec)[0] / b0)
    b10 = {}
    for t in PERIODS:
        c2 = thermal_cost(L, S4[t], seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])
        red = []
        for o, d, _ in ev:
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
            sp, bb = sols[0][2], sols[budget(sols, L0, 0.10)][2]
            dsp = (L[sp] * S4[t][sp]).sum()
            red.append(100 * (1 - (L[bb] * S4[t][bb]).sum() / dsp) if dsp > 0 else 0)
        b10[t] = float(np.median(red))
    o2 = np.argsort(-fl); cl = np.cumsum(L[o2]) / L.sum(); ct = np.cumsum((fl * L)[o2]) / (fl * L).sum()
    OUT["demand"][name] = {"top5_jaccard_vs_ref": jac_len(ref[0.05], select(tb, .05)), "red_TB5": red_tb,
                           "red_flow5": red_fl, "red_random5": float(red_rd), "red_ref_selection5": red_ref_sel,
                           "top5pct_len_tripkm": float(ct[np.searchsorted(cl, .05)]),
                           "sp_len_med": float(np.median([x[2] for x in ev])), "B10_dDsun_med": b10}
    print("demand", name, OUT["demand"][name], flush=True)

# ---------------------------------------------------------------- 7 biased heights
bldh = bld.h.values
pert = bld.h_src.isin(["globfp", "globfp_footprint", "imputed"]).values
v = pd.read_csv(PROC / "height_validation.csv")
ratio = float(np.exp(np.median(np.log(v.h_osm / v.h_glo))))
hb = np.maximum(bldh * np.where(pert, ratio, 1.0), 3.0)
with rasterio.open(PROC / "bld_h_1m.tif") as s_:
    TF = s_.transform; H, W = s_.shape
order = bld.sort_values("h")
bid = features.rasterize(((g, i + 1) for i, g in zip(order.index, order.geometry)), out_shape=(H, W),
                         transform=TF, dtype="int32", fill=0)
bh = np.r_[0.0, hb].astype("float32")[bid]
ch = rasterio.open(PROC / "canopy_h_1m.tif").read(1)
bmask = rasterio.open(PROC / "bld_mask_1m.tif").read(1) > 0
Ec, Rc, Cc = corridor_points(seg.geometry.values, bmask, TF)
sun = sun_positions(HRS)
SHb = np.stack([edge_sunlit(cast(bh, ch, r.azimuth, r.elevation), (Ec, Rc, Cc), seg.covered.values) for r in sun.itertuples()])
tbb = FLOW * SHb.mean(0)
b0 = day_metrics(np.array([], int), SH_=SHb)[0]
OUT["bias_heights"] = {"ratio": ratio, "top5_jaccard_vs_ref": jac_len(ref[0.05], select(tbb, .05)),
                       "red_TB5": 100 * (1 - day_metrics(select(tbb, .05), SH_=SHb)[0] / b0),
                       "red_ref_selection5": 100 * (1 - day_metrics(ref[0.05], SH_=SHb)[0] / b0),
                       "red_flow5": 100 * (1 - day_metrics(select(FLOW, .05), SH_=SHb)[0] / b0),
                       "net_s_day": float((L * SHb.mean(0)).sum() / L.sum())}
print("bias", OUT["bias_heights"], flush=True)
json.dump(OUT, open(PROC / "priority_checks.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))

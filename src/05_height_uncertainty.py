"""Monte Carlo propagation of building-height uncertainty to edge shade and cool routes."""
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from config import *
from shade import sun_positions, cast
from impedance import load_segments, thermal_cost
from routing import Net, pareto_paths, budget
from sampling import corridor_points, edge_mean, edge_sunlit

K = 30
seg = load_segments()
net = Net(seg.u.values, seg.v.values, seg.length.values)
L = seg.length.values
od = pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"]
base = pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"]

with rasterio.open(PROC / "bld_h_1m.tif") as s:
    TF = s.transform; H, W = s.shape
canopy_h = rasterio.open(PROC / "canopy_h_1m.tif").read(1)
bmask = rasterio.open(PROC / "bld_mask_1m.tif").read(1) > 0
bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
order = bld.sort_values("h")
bid = features.rasterize(((g, i + 1) for i, g in zip(order.index, order.geometry)), out_shape=(H, W),
                         transform=TF, dtype="int32", fill=0)
h0 = bld.h.values
perturb = bld.h_src.isin(["globfp", "globfp_footprint", "imputed"]).values

v = pd.read_csv(PROC / "height_validation.csv")
r = np.log(v.h_osm / v.h_glo).values          # log ratio reference / GloBFP
r_med = float(np.median(r))
r_c = r - r_med

n = len(seg)
Ec, Rc, Cc = corridor_points(seg.geometry.values, bmask, TF)

sun = sun_positions(PERIODS).set_index("time")
rng = np.random.default_rng(SEED + 7)


def run(hvec, tag):
    lut = np.r_[0.0, hvec].astype("float32")
    bh = lut[bid]
    out = {}
    for t in PERIODS:
        tr = cast(bh, canopy_h, sun.loc[t, "azimuth"], sun.loc[t, "elevation"])
        s = edge_sunlit(tr, (Ec, Rc, Cc), seg.covered.values)
        c2 = thermal_cost(L, s, seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])
        rows = []
        for k, (o, d, _) in enumerate(od):
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
            b10 = sols[budget(sols, L0, 0.10)][2]
            sp = sols[0][2]
            b0 = base[t][k]["B10"]["edges"]
            a, b = set(b10), set(b0)
            jac = L[list(a & b)].sum() / L[list(a | b)].sum()
            d_sp = (L[sp] * s[sp]).sum(); d_b = (L[b10] * s[b10]).sum()
            # baseline route evaluated in the perturbed shade field
            d_b0 = (L[b0] * s[b0]).sum()
            rows.append((jac, d_sp, d_b, d_b0, (d_sp - d_b) / d_sp if d_sp > 0 else 0.0))
        out[t] = {"net_sunlit": float((L * s).sum() / L.sum()), "rows": np.array(rows)}
    print(tag, {t: round(out[t]["net_sunlit"], 4) for t in PERIODS})
    return out


results = {"r_med": r_med, "n_ref": int(len(r)), "bias_scenario": None, "mc": []}
hb = h0 * np.where(perturb, np.exp(r_med), 1.0)
results["bias_scenario"] = run(np.maximum(hb, 3.0), "bias")
for k in range(K):
    e = rng.choice(r_c, size=len(h0), replace=True)
    hk = h0 * np.where(perturb, np.exp(e), 1.0)
    results["mc"].append(run(np.maximum(hk, 3.0), f"mc{k}"))
pickle.dump(results, open(PROC / "height_mc.pkl", "wb"))


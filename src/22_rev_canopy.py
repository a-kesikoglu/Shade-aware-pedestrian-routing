"""Canopy shadow rule: (a) simple geometry test of a cylindrical crown against the exact ray test,
(b) network effect of replacing the rule that shades every pixel under a crown by the ray test alone.
Outputs: data/processed/rev_canopy.json"""
import json
import pickle
import numpy as np
import rasterio
from config import *
from shade import cast, sun_positions
from impedance import load_segments, thermal_cost
from routing import Net, pareto_paths, budget
from sampling import corridor_points, edge_sunlit

out = {"geometry": []}
# ---------------------------------------------------------------- (a) single cylindrical crown
R, HC, BASE = 4.0, 10.0, 0.25          # crown radius (m), canopy height (m), crown base as share of height
n = 81
yy, xx = np.mgrid[0:n, 0:n] - n // 2
canopy = np.where(xx ** 2 + yy ** 2 <= R ** 2, HC, 0).astype("float32")
zero = np.zeros_like(canopy)
for el in [15, 20, 30, 45, 60, 75]:
    for az in [90, 135, 180, 225, 270]:
        u = np.array([np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))])
        # exact ray test for pixel centres: ray from (x, y, 0) towards the sun, crown = cylinder between base and top
        px, py = xx.astype(float), -yy.astype(float)          # y north = up
        d = np.linspace(0, 120, 4801)
        X = px[..., None] + d * u[0]; Y = py[..., None] + d * u[1]; Z = d * np.tan(np.deg2rad(el))
        inside = (X ** 2 + Y ** 2 <= R ** 2) & (Z >= BASE * HC) & (Z < HC)
        exact = inside.any(-1)
        under = canopy > 0
        a = cast(zero, canopy, az, el, under_crown=True) < 1
        b = cast(zero, canopy, az, el, under_crown=False) < 1
        out["geometry"].append({
            "elev": el, "azimuth": az,
            "under_crown_px": int(under.sum()),
            "under_crown_exact_shaded_share": float(exact[under].mean()),
            "rule_overshade_px": int((a & ~exact).sum()), "ray_overshade_px": int((b & ~exact).sum()),
            "rule_undershade_px": int((~a & exact).sum()), "ray_undershade_px": int((~b & exact).sum()),
            "exact_shaded_px": int(exact.sum()),
            "iou_rule": float((a & exact).sum() / (a | exact).sum()), "iou_ray": float((b & exact).sum() / (b | exact).sum())})

# ---------------------------------------------------------------- (b) network effect
seg = load_segments()
L = seg.length.values
net = Net(seg.u.values, seg.v.values, L)
OD = pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"]
base = pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"]
bh = rasterio.open(PROC / "bld_h_1m.tif").read(1)
ch = rasterio.open(PROC / "canopy_h_1m.tif").read(1)
bm = rasterio.open(PROC / "bld_mask_1m.tif").read(1) > 0
TF = rasterio.open(PROC / "bld_h_1m.tif").transform
idx = corridor_points(seg.geometry.values, bm, TF)
sun = sun_positions(PERIODS).set_index("time")
out["network"] = {}
for t in PERIODS:
    tr = cast(bh, ch, sun.loc[t, "azimuth"], sun.loc[t, "elevation"], under_crown=False)
    s = edge_sunlit(tr, idx, seg.covered.values)
    s0 = seg[f"s_{t.replace(':', '')}"].values
    c2 = thermal_cost(L, s, seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])
    red, same, jac = [], [], []
    for k, (o, d, _) in enumerate(OD):
        sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
        b10 = sols[budget(sols, L0, 0.10)][2]; sp = sols[0][2]
        ds, db = (L[sp] * s[sp]).sum(), (L[b10] * s[b10]).sum()
        red.append(100 * (ds - db) / ds if ds > 0 else 0)
        same.append(set(b10) == set(base[t][k]["B10"]["edges"]))
        A, B = set(b10), set(base[t][k]["B10"]["edges"]); jac.append(float(L[list(A & B)].sum() / L[list(A | B)].sum()))
    out["network"][t] = {"net_sunlit_ref": float((L * s0).sum() / L.sum()), "net_sunlit_ray": float((L * s).sum() / L.sum()),
                         "len_changed_gt005": float(L[np.abs(s - s0) > 0.05].sum() / L.sum()),
                         "B10_dDsun_med": float(np.median(red)), "B10_same_share": float(np.mean(same)), "B10_jaccard_med": float(np.median(jac)), "B10_jaccard_mean": float(np.mean(jac))}
    print(t, out["network"][t], flush=True)
json.dump(out, open(PROC / "rev_canopy.json", "w"), indent=1)
for g in out["geometry"]:
    if g["azimuth"] == 180:
        print(g)

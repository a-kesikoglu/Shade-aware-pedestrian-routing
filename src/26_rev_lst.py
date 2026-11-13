"""Surface temperature inputs: valid pixels per Landsat scene, agreement of the street-level pattern between
scenes, and routes recomputed with the single scene closest to 21 July 2025 instead of the summer median.
Outputs: data/processed/rev_lst.json"""
import json
import pickle
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import stats
from config import *
from impedance import load_segments, thermal_cost, robust_minmax
from routing import Net, pareto_paths, budget
from sampling import sample_points, edge_mean

seg = load_segments()
L = seg.length.values
net = Net(seg.u.values, seg.v.values, L)
OD = pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"]
base = pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"]
meta = json.load(open(PROC / "satellite_meta.json"))
aoi = gpd.read_file(PROC / "aoi.gpkg").geometry.iloc[0]
with rasterio.open(PROC / "bld_h_1m.tif") as s:
    TF, H, W, CRS = s.transform, s.height, s.width, s.crs
Eb, Rb, Cb = sample_points(seg.geometry.values, 2.0, np.arange(-10, 10.1, 2.0), TF, (H, W))
out = {"scenes": []}
seg_lst = []
for m in meta["landsat"]:
    k = m["k"]
    with rasterio.open(RAW / f"ls_st_{k:02d}.tif") as s, rasterio.open(RAW / f"ls_qa_{k:02d}.tif") as q:
        st = s.read(1).astype("float32"); qa = q.read(1); tf30 = s.transform
        bad = (st == 0) | ((qa & 0b11110) > 0)
        t = st * 0.00341802 + 149.0 - 273.15
        t[bad] = np.nan
        inside = rasterio.features.geometry_mask([aoi], out_shape=st.shape, transform=tf30, invert=True)
    g = np.full((H, W), np.nan, "float32")
    reproject(t, g, src_transform=tf30, src_crs=CRS, dst_transform=TF, dst_crs=CRS, resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    v = edge_mean(Eb, g[Rb, Cb], len(seg))
    seg_lst.append(v)
    out["scenes"].append({"k": k, "id": m["id"], "date": m["datetime"][:10],
                          "valid_px_aoi": int((~bad & inside).sum()), "px_aoi": int(inside.sum()),
                          "mean_aoi": float(np.nanmean(t[inside]))})
SL = np.stack(seg_lst)
ref = seg.lst.values
rho = [float(stats.spearmanr(x, ref, nan_policy="omit").statistic) for x in SL]
pair = [float(stats.spearmanr(SL[i], SL[j], nan_policy="omit").statistic) for i in range(len(SL)) for j in range(i + 1, len(SL))]
out["spearman_scene_vs_median"] = rho
out["spearman_between_scenes"] = {"min": min(pair), "median": float(np.median(pair)), "max": max(pair)}
out["segment_sd_between_scenes_med"] = float(np.nanmedian(np.nanstd(SL, 0)))
out["scene_mean_range"] = [min(s["mean_aoi"] for s in out["scenes"]), max(s["mean_aoi"] for s in out["scenes"])]

# routes with the scene closest to 21 July 2025
kc = min(range(len(meta["landsat"])), key=lambda i: abs(int(meta["landsat"][i]["datetime"][8:10]) - 21)
         if meta["landsat"][i]["datetime"][5:7] == "07" else 99)
lst_n = robust_minmax(SL[kc])
out["single_scene"] = {"id": meta["landsat"][kc]["id"], "date": meta["landsat"][kc]["datetime"][:10]}
c_sat = thermal_cost(L, np.zeros_like(L), seg.ndvi_n.values, lst_n, (0, 0.5, 0.5), beta=0)
sat_b10 = []
for o, d, _ in OD:
    sols, L0 = pareto_paths(net, o, d, L, c_sat, detour=1.5)
    sat_b10.append(sols[budget(sols, L0, 0.10)][2])
for t in PERIODS:
    s = seg[f"s_{t.replace(':', '')}"].values
    c2 = thermal_cost(L, s, seg.ndvi_n.values, lst_n, WEIGHTS["balanced"])
    red, same, sred = [], [], []
    for k, (o, d, _) in enumerate(OD):
        sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
        b = sols[budget(sols, L0, 0.10)][2]
        sp = base[t][k]["SP"]
        db = (L[b] * s[b]).sum(); dsat = (L[sat_b10[k]] * s[sat_b10[k]]).sum()
        red.append(100 * (sp["dsun"] - db) / sp["dsun"])
        sred.append(100 * (sp["dsun"] - dsat) / sp["dsun"])
        same.append(set(b) == set(base[t][k]["B10"]["edges"]))
    out["single_scene"][t] = {"B10_dDsun_med": float(np.median(red)), "B10_same_share": float(np.mean(same)),
                              "sat_dDsun_med": float(np.median(sred)),
                              "sat_worse_share": float(np.mean(np.array(sred) < -1e-9))}
    print(t, out["single_scene"][t], flush=True)
json.dump(out, open(PROC / "rev_lst.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in out if k != "single_scene"}, indent=1))

"""Hourly shadow rasters for 21 July 2025 and the densified pedestrian graph with
edge-level sunlit fraction, NDVI and LST."""
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import rasterio
from shapely.geometry import LineString
from shapely.ops import substring
from config import *
from shade import sun_positions, cast
from sampling import sample_points, corridor_points, edge_mean, edge_sunlit

rd = lambda n: rasterio.open(PROC / f"{n}_1m.tif")
with rd("bld_h") as s:
    bld_h = s.read(1); TF = s.transform; prof = s.profile
canopy_h = rd("canopy_h").read(1)
bmask = rd("bld_mask").read(1) > 0
ndvi = rd("ndvi").read(1)
lst = rd("lst").read(1)
H, W = bld_h.shape

TIMES = sorted(set(HOURLY + PERIODS))
sun = sun_positions(TIMES)
sun.to_csv(TAB / "sun_positions.csv", index=False)
print(sun)

# ---------------------------------------------------------------- graph densification
# segments of at most SEG_MAX m with midpoint in the AOI, largest connected component;
# vehicle underpasses removed, pedestrian underpasses flagged as covered (network.py)
from network import build_segments
aoi = gpd.read_file(PROC / "aoi.gpkg").geometry.iloc[0]
seg, npos, removed = build_segments(RAW / "walk_network.graphml", aoi, return_removed=True)
covered = seg["covered"].values.astype(bool)
seg["eid"] = np.arange(len(seg))
c = np.array([[ls.coords[0], ls.coords[-1]] for ls in seg.geometry])
dx, dy = c[:, 1, 0] - c[:, 0, 0], c[:, 1, 1] - c[:, 0, 1]
seg["bearing"] = (np.degrees(np.arctan2(dx, dy)) + 360) % 180   # 0 = N-S, 90 = E-W


# ---------------------------------------------------------------- edge sampling
n = len(seg)
Ec, Rc, Cc = corridor_points(seg.geometry.values, bmask, TF)
# 10 m buffer for NDVI and LST
Eb, Rb, Cb = sample_points(seg.geometry.values, 2.0, np.arange(-10, 10.1, 2.0), TF, (H, W))
seg["ndvi"] = edge_mean(Eb, ndvi[Rb, Cb], n)
seg["lst"] = edge_mean(Eb, lst[Rb, Cb], n)
seg["canopy_share"] = edge_mean(Ec, (canopy_h[Rc, Cc] > 0).astype(float), n)
# mean height of buildings within the 20 m buffer (street canyon descriptor)
Eh, Rh, Ch = sample_points(seg.geometry.values, 2.0, np.arange(-20, 20.1, 2.0), TF, (H, W))
hb = bld_h[Rh, Ch]
seg["h_adj"] = edge_mean(Eh[hb > 0], hb[hb > 0], n)
seg["bld_share_20m"] = edge_mean(Eh, (hb > 0).astype(float), n)

# ---------------------------------------------------------------- shadows
sun_tab = {}
for r in sun.itertuples():
    t0 = time.time()
    tr = cast(bld_h, canopy_h, r.azimuth, r.elevation)
    seg[f"s_{r.time.replace(':', '')}"] = edge_sunlit(tr, (Ec, Rc, Cc), covered)
    if r.time in PERIODS:
        p = prof.copy(); p.update(dtype="uint8", nodata=255)
        with rasterio.open(PROC / f"sunlit_{r.time.replace(':', '')}.tif", "w", **p) as d:
            d.write((tr * 100).astype("uint8"), 1)
    mm = int(MARGIN)
    m = (bld_h[mm:-mm, mm:-mm] == 0)
    sun_tab[r.time] = {"elev": float(r.elevation), "az": float(r.azimuth),
                       "ground_sunlit_share": float(tr[mm:-mm, mm:-mm][m].mean()),
                       "sec": round(time.time() - t0, 1)}
    print(r.time, sun_tab[r.time])

seg.to_file(PROC / "network_segments.gpkg", driver="GPKG")
pd.DataFrame({k: v for k, v in npos.items()}, index=["x", "y"]).T \
    .rename_axis("node").to_csv(PROC / "network_nodes.csv")
stats = {"n_segments": n, "n_nodes": int(len(set(seg.u) | set(seg.v))),
         "total_length_km": float(seg.length.sum() / 1000), "mean_seg_len": float(seg.length.mean()),
         "covered_km": float(seg.length[covered].sum() / 1000),
         "network_km_with_underpasses": build_segments.full_km,
         "vehicle_underpass_km_removed": float(build_segments.full_km - seg.length.sum() / 1000),
         "vehicle_underpasses": removed.highway.value_counts().to_dict(),
         "highway_share_len": (seg.groupby("highway").length.sum() / seg.length.sum()).round(4).to_dict(),
         "sun": sun_tab}
json.dump(stats, open(PROC / "network_stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != "highway_share_len"}, indent=1))


"""Building heights (OSM footprints + 3D-GloBFP heights), 1 m DSM layers,
Sentinel-2 NDVI and Landsat summer LST composite."""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from config import *

aoi = gpd.read_file(PROC / "aoi.gpkg")
x0, y0, x1, y1 = aoi.total_bounds
M = MARGIN
gx0, gy1 = x0 - M, y1 + M
W = H = int((x1 - x0 + 2 * M) / DSM_RES)
TF = from_origin(gx0, gy1, DSM_RES, DSM_RES)
PROF = dict(driver="GTiff", width=W, height=H, count=1, crs=CRS_UTM, transform=TF,
            dtype="float32", compress="deflate", nodata=None)
stats = {}

# ---------------------------------------------------------------- buildings
osm = gpd.read_file(RAW / "osm_buildings.gpkg").to_crs(CRS_UTM)
osm = osm[osm.geometry.area > 4].reset_index(drop=True)
osm["bid"] = np.arange(len(osm))
glo = gpd.read_file(RAW / "globfp_buildings.gpkg").to_crs(CRS_UTM)
glo = glo[glo.geometry.is_valid & (glo.geometry.area > 4)].reset_index(drop=True)
glo["gid"] = np.arange(len(glo))

inter = gpd.overlay(osm[["bid", "geometry"]], glo[["gid", "Height", "geometry"]], how="intersection",
                    keep_geom_type=True)
inter["a"] = inter.geometry.area
# area-weighted mean GloBFP height for every OSM footprint
wh = inter.groupby("bid").apply(lambda d: np.average(d["Height"], weights=d["a"]), include_groups=False)
cov = inter.groupby("bid")["a"].sum() / osm.set_index("bid").geometry.area
osm["h_glo"] = osm["bid"].map(wh)
osm["cov_glo"] = osm["bid"].map(cov).fillna(0)
osm.loc[osm["cov_glo"] < 0.2, "h_glo"] = np.nan

lev = pd.to_numeric(osm.get("building:levels"), errors="coerce")
hgt = pd.to_numeric(osm.get("height").astype(str).str.replace("m", "").str.strip(), errors="coerce") \
    if "height" in osm else pd.Series(np.nan, index=osm.index)
osm["h_osm"] = np.where(hgt.notna(), hgt, lev * FLOOR_HEIGHT)
osm.loc[(osm["h_osm"] <= 0) | (osm["h_osm"] > 200), "h_osm"] = np.nan

# validation sample: OSM tagged height vs GloBFP
v = osm.dropna(subset=["h_osm", "h_glo"])
err = v["h_glo"] - v["h_osm"]
stats["height_validation"] = {
    "n": int(len(v)), "bias_m": float(err.mean()), "mae_m": float(err.abs().mean()),
    "rmse_m": float(np.sqrt((err ** 2).mean())),
    "r": float(np.corrcoef(v["h_glo"], v["h_osm"])[0, 1]) if len(v) > 2 else None,
    "median_abs_rel_err": float((err.abs() / v["h_osm"]).median())}
v[["bid", "h_osm", "h_glo"]].to_csv(PROC / "height_validation.csv", index=False)

# height assignment hierarchy: OSM tag > GloBFP > local median (100 m)
osm["h"] = osm["h_osm"].where(osm["h_osm"].notna(), osm["h_glo"])
osm["h_src"] = np.where(osm["h_osm"].notna(), "osm", np.where(osm["h_glo"].notna(), "globfp", "imputed"))
known = osm[osm["h"].notna()]
cent = osm.geometry.centroid
from scipy.spatial import cKDTree
kt = cKDTree(np.c_[known.geometry.centroid.x, known.geometry.centroid.y])
miss = osm["h"].isna()
if miss.any():
    idx = kt.query_ball_point(np.c_[cent[miss].x, cent[miss].y], r=100)
    med = [np.median(known["h"].values[i]) if len(i) else np.median(known["h"]) for i in idx]
    osm.loc[miss, "h"] = med

# GloBFP footprints that OSM misses (< 10% overlap with any OSM building)
gcov = inter.groupby("gid")["a"].sum() / glo.set_index("gid").geometry.area
glo["cov_osm"] = glo["gid"].map(gcov).fillna(0)
extra = glo[glo["cov_osm"] < 0.1].copy()
extra = extra.rename(columns={"Height": "h"})
extra["h_src"] = "globfp_footprint"
bld = pd.concat([osm[["h", "h_src", "geometry"]], extra[["h", "h_src", "geometry"]]], ignore_index=True)
bld = gpd.GeoDataFrame(bld, geometry="geometry", crs=CRS_UTM)
bld["h"] = bld["h"].clip(lower=3.0)
bld.to_file(PROC / "buildings_lod1.gpkg", driver="GPKG")
inaoi = bld[bld.intersects(aoi.geometry.iloc[0])]
stats["buildings"] = {"n_total_with_margin": int(len(bld)), "n_in_aoi": int(len(inaoi)),
                      "src_counts_aoi": inaoi["h_src"].value_counts().to_dict(),
                      "h_median": float(inaoi["h"].median()), "h_p90": float(inaoi["h"].quantile(0.9)),
                      "h_max": float(inaoi["h"].max()),
                      "footprint_cover_aoi": float(inaoi.geometry.intersection(aoi.geometry.iloc[0]).area.sum()
                                                   / aoi.area.iloc[0])}

# ---------------------------------------------------------------- 1 m rasters
bmask = features.rasterize(((g, 1) for g in bld.geometry), out_shape=(H, W), transform=TF, dtype="uint8")
order = bld.sort_values("h")  # taller buildings burnt last
bh = features.rasterize(((g, h) for g, h in zip(order.geometry, order.h)), out_shape=(H, W), transform=TF,
                        dtype="float32", fill=0)


def to_grid(src_path, resampling, dtype="float32"):
    with rasterio.open(src_path) as s:
        out = np.zeros((H, W), dtype=dtype)
        reproject(s.read(1), out, src_transform=s.transform, src_crs=s.crs, dst_transform=TF,
                  dst_crs=CRS_UTM, resampling=resampling)
    return out


# NDVI from Sentinel-2 L2A (BOA offset of -1000 applies to processing baseline >= 04.00)
red = to_grid(RAW / "s2_B04.tif", Resampling.bilinear)
nir = to_grid(RAW / "s2_B08.tif", Resampling.bilinear)
red = (red - 1000) / 10000.0
nir = (nir - 1000) / 10000.0
ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0).astype("float32")

# canopy height (Tolan et al., 2024) restricted to vegetated, non-building pixels
chm = to_grid(RAW / "canopy_height_meta.tif", Resampling.bilinear)
veg = (chm >= 2.5) & (bmask == 0) & (ndvi >= 0.15)
ch = np.where(veg, chm, 0).astype("float32")

# Landsat C2 L2 ST: scale 0.00341802, offset 149.0 K; QA_PIXEL bits 1-4 (dilated cloud, cirrus, cloud, shadow)
sts = []
for k in range(len(json.load(open(PROC / "satellite_meta.json"))["landsat"])):
    with rasterio.open(RAW / f"ls_st_{k:02d}.tif") as s, rasterio.open(RAW / f"ls_qa_{k:02d}.tif") as q:
        st = s.read(1).astype("float32"); qa = q.read(1)
        bad = (st == 0) | ((qa & 0b11110) > 0)
        t = st * 0.00341802 + 149.0 - 273.15
        t[bad] = np.nan
        sts.append(t); ls_tf, ls_crs = s.transform, s.crs
stack = np.stack(sts)
stats["lst_scene_valid_share"] = [float(np.isfinite(a).mean()) for a in stack]
lst30 = np.nanmedian(stack, axis=0).astype("float32")
with rasterio.open(PROC / "lst_median_30m.tif", "w", driver="GTiff", width=lst30.shape[1], height=lst30.shape[0],
                   count=1, crs=ls_crs, transform=ls_tf, dtype="float32") as d:
    d.write(lst30, 1)
lst = to_grid(PROC / "lst_median_30m.tif", Resampling.bilinear)

for name, arr in [("bld_h", bh), ("canopy_h", ch), ("ndvi", ndvi), ("lst", lst), ("bld_mask", bmask.astype("float32"))]:
    with rasterio.open(PROC / f"{name}_1m.tif", "w", **PROF) as d:
        d.write(arr.astype("float32"), 1)

inner = (slice(int(M), H - int(M)), slice(int(M), W - int(M)))
stats["canopy_cover_aoi"] = float(veg[inner].mean())
stats["ndvi_aoi"] = {"mean": float(ndvi[inner].mean()), "p95": float(np.percentile(ndvi[inner], 95))}
l_in = lst[inner]
stats["lst_aoi_degC"] = {"mean": float(np.nanmean(l_in)), "p5": float(np.nanpercentile(l_in, 5)),
                         "p95": float(np.nanpercentile(l_in, 95))}
json.dump(stats, open(PROC / "surface_stats.json", "w"), indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))

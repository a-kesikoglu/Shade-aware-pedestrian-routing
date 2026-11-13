"""Independent check of the modelled shade against shadows observed in Sentinel-2 L2A scenes.

For clear scenes without snow, shadows are cast with the surface model at the acquisition time and
aggregated to the 10 m grid of the scene. On 10 m street pixels without buildings, the modelled shaded
fraction is compared with the observed reflectance. Alternative building heights test whether the
3D-GloBFP heights explain the observed shadows better than corrected or constant heights.
Outputs: data/processed/rev_s2_validation.json, outputs/tables/rev_s2_validation.csv"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
import pvlib
import pystac_client
import planetary_computer as pc
from scipy import stats
from config import *
from shade import cast

OUTD = RAW / "s2_validation"
OUTD.mkdir(exist_ok=True)
aoi = gpd.read_file(PROC / "aoi.gpkg").geometry.iloc[0]
xmin, ymin, xmax, ymax = aoi.bounds
B = (np.floor(xmin / 10) * 10, np.floor(ymin / 10) * 10, np.ceil(xmax / 10) * 10, np.ceil(ymax / 10) * 10)
bbox_ll = transform_bounds(CRS_UTM, "EPSG:4326", *B)
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def clip(href, bounds):
    with rasterio.open(href) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        return src.read(1, window=win, boundless=True, fill_value=0), src.window_transform(win), src.crs


# ---------------------------------------------------------------- scene selection
items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime="2025-01-01/2026-11-10",
                        query={"eo:cloud_cover": {"lt": 2}}).items())
cands = []
for it in items:
    elev = 90 - float(it.properties.get("s2:mean_solar_zenith", 90))
    cands.append((it, elev))
print(len(cands), "candidate scenes")
chosen = []
for it, elev in sorted(cands, key=lambda x: x[0].datetime):
    if not (18 <= elev <= 45 or it.id.startswith("S2C_MSIL2A_20250721")):
        continue
    scl, tf, crs = clip(it.assets["SCL"].href, B)
    if str(crs) != CRS_UTM and crs.to_epsg() != 32636:
        continue
    n = scl.size
    if (scl == 11).sum() / n > 0.001 or np.isin(scl, [3, 8, 9, 10]).sum() / n > 0.001 or (scl == 0).sum() / n > 0.01:
        continue                               # snow, cloud shadow, cloud, cirrus or no data over the area
    chosen.append((it, elev))
print([(i.id, round(e, 1)) for i, e in chosen])
# one scene per month at most, keep up to six low-sun scenes plus the scene of 21 July 2025
bym = {}
for it, e in chosen:
    bym.setdefault(it.datetime.strftime("%Y-%m"), (it, e))
low = [v for k, v in sorted(bym.items()) if v[1] <= 45 and not v[0].id.startswith("S2C_MSIL2A_20250721")]
if len(low) > 6:
    low = [low[i] for i in np.linspace(0, len(low) - 1, 6).round().astype(int)]
summer = [v for v in chosen if v[0].id.startswith("S2C_MSIL2A_20250721")]
scenes = low + summer[:1]

# ---------------------------------------------------------------- model inputs
with rasterio.open(PROC / "bld_h_1m.tif") as s:
    TF1 = s.transform; H1, W1 = s.shape
canopy = rasterio.open(PROC / "canopy_h_1m.tif").read(1)
bmask = rasterio.open(PROC / "bld_mask_1m.tif").read(1) > 0
bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
order = bld.sort_values("h")
bid = features.rasterize(((g, i + 1) for i, g in zip(order.index, order.geometry)), out_shape=(H1, W1),
                         transform=TF1, dtype="int32", fill=0)
h0 = bld.h.values
perturb = np.isin(bld.h_src.values, ["globfp", "globfp_footprint", "imputed"])
rh = json.load(open(PROC / "rev_heights.json"))
f_all = rh["all"]["ratio_med"]
GCLS = [0, 10, 20, 400]
gb = np.digitize(h0, GCLS[1:-1])
f_cls = np.array([rh["by_glo_class"][c]["ratio_med"] for c in ["<10", "10-20", ">=20"]])[gb]
hmed = float(np.median(h0))
VERS = {"3D-GloBFP": h0,
        "uniform correction": np.maximum(3, h0 * np.where(perturb, f_all, 1)),
        "class correction": np.maximum(3, h0 * np.where(perturb, f_cls, 1)),
        "constant median": np.where(perturb, hmed, h0),
        "no buildings": np.zeros_like(h0)}

seg = gpd.read_file(PROC / "network_segments.gpkg")
art = seg.highway.astype(str).str.contains("primary|trunk|secondary")
nrows, ncols = int((B[3] - B[1]) / 10), int((B[2] - B[0]) / 10)
TF10 = rasterio.transform.from_origin(B[0], B[3], 10, 10)


def to10(a):
    out = np.zeros((nrows, ncols), "float32")
    reproject(a.astype("float32"), out, src_transform=TF1, src_crs=CRS_UTM, dst_transform=TF10, dst_crs=CRS_UTM,
              resampling=Resampling.average)
    return out


bfrac = to10(bmask)
cfrac = to10(canopy > 0)
dist = lambda geoms: features.rasterize(((g.buffer(5), 1) for g in geoms), out_shape=(nrows, ncols),
                                        transform=TF10, fill=0).astype(bool)
street = dist(seg.geometry) & (bfrac == 0)
in_aoi = features.rasterize([(aoi, 1)], out_shape=(nrows, ncols), transform=TF10, fill=0).astype(bool)
street &= in_aoi
strata = {"all streets": street & (cfrac < 0.05),
          "arterials": dist(seg.geometry[art]) & street & (cfrac < 0.05),
          "local streets": dist(seg.geometry[~art]) & ~dist(seg.geometry[art]) & street & (cfrac < 0.05),
          "tree-lined": street & (cfrac >= 0.2)}


def otsu(x):
    h, e = np.histogram(x, 256)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def auc(score, label):
    r = stats.rankdata(score)
    n1 = label.sum(); n0 = (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


rows, meta = [], []
for it, elev in scenes:
    bands = {}
    for b in ["B02", "B03", "B04", "B08"]:
        a, tf, _ = clip(it.assets[b].href, B)
        a = a.astype("float32")
        off = 1000 if float(it.properties.get("s2:processing_baseline", "4.0")) >= 4 else 0
        bands[b] = (a - off) / 10000
    bright = np.mean([bands[b] for b in bands], 0)
    ts = pd.Timestamp(it.datetime)
    sp = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([ts]), CENTER_LAT, CENTER_LON)
    az, el = float(sp.azimuth.iloc[0]), float(sp.apparent_elevation.iloc[0])
    meta.append({"id": it.id, "datetime": it.datetime.isoformat(), "elev": el, "azimuth": az,
                 "baseline": it.properties.get("s2:processing_baseline")})
    for name, hv in VERS.items():
        bh = np.r_[0.0, hv].astype("float32")[bid]
        tr = cast(bh, canopy, az, el)
        shade10 = to10(1 - tr)
        for sname, m in strata.items():
            x = shade10[m]; y = bright[m]
            thr = otsu(bright[strata["all streets"]])
            obs = y < thr
            mod = x >= 0.5
            tp = (mod & obs).sum(); tn = (~mod & ~obs).sum()
            rows.append({"scene": it.datetime.strftime("%Y-%m-%d"), "elev": el, "heights": name, "stratum": sname,
                         "n": int(m.sum()), "obs_shadow_share": float(obs.mean()), "mod_shadow_share": float(mod.mean()),
                         "spearman": float(stats.spearmanr(x, -y).statistic), "auc": auc(x, obs),
                         "accuracy": float((tp + tn) / len(x)),
                         "balanced_accuracy": float(0.5 * (tp / max(obs.sum(), 1) + tn / max((~obs).sum(), 1))),
                         "iou": float(tp / max((mod | obs).sum(), 1))})
        print(it.id, name, flush=True)
tab = pd.DataFrame(rows)
tab.to_csv(TAB / "rev_s2_validation.csv", index=False)
json.dump({"scenes": meta, "n_candidates": len(cands), "otsu_note": "threshold from all street pixels per scene"},
          open(PROC / "rev_s2_validation.json", "w"), indent=1)
print(tab[tab.stratum == "all streets"].round(3).to_string())

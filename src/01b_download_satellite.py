"""Query Planetary Computer STAC and clip Sentinel-2 L2A (B04, B08, SCL) and
Landsat 8/9 C2 L2 (ST_B10, QA_PIXEL) to the study area for clear summer scenes."""
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import pystac_client
import planetary_computer as pc
import geopandas as gpd
from config import *

aoi = gpd.read_file(PROC / "aoi.gpkg").to_crs(4326) if (PROC / "aoi.gpkg").exists() else None
from shapely.geometry import box
from pyproj import Transformer
tr = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
m = HALF_SIZE_M + 400
bbox_utm = (cx - m, cy - m, cx + m, cy + m)
bbox_ll = transform_bounds(CRS_UTM, "EPSG:4326", *bbox_utm)

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)


def clip(href, out, bounds_utm):
    with rasterio.open(href) as src:
        b = transform_bounds(CRS_UTM, src.crs, *bounds_utm)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        prof = src.profile.copy()
        prof.update(height=arr.shape[0], width=arr.shape[1],
                    transform=src.window_transform(win), driver="GTiff", compress="deflate")
        prof.pop("blockxsize", None); prof.pop("blockysize", None); prof["tiled"] = False
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr, 1)
    return arr


meta = {}
# Sentinel-2: least cloudy scene, 15 Jun - 31 Aug 2025
s2 = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox_ll,
                     datetime="2025-06-15/2025-08-31",
                     query={"eo:cloud_cover": {"lt": 5}}).items())
s2 = sorted([i for i in s2 if i.properties["eo:cloud_cover"] < 1], key=lambda i: abs(int(i.datetime.strftime("%j")) - 202))
it = s2[0]
for band in ["B04", "B08", "SCL"]:
    clip(it.assets[band].href, RAW / f"s2_{band}.tif", bbox_utm)
meta["sentinel2"] = {"id": it.id, "datetime": it.datetime.isoformat(),
                     "cloud_cover": it.properties["eo:cloud_cover"], "n_candidates": len(s2)}

# Landsat 8/9 C2 L2: all clear summer scenes (used for a multi-date composite)
ls = list(cat.search(collections=["landsat-c2-l2"], bbox=bbox_ll,
                     datetime="2025-06-15/2025-08-31",
                     query={"eo:cloud_cover": {"lt": 10}, "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
ls = sorted(ls, key=lambda i: i.datetime)
meta["landsat"] = []
for k, it in enumerate(ls):
    st = clip(it.assets["lwir11"].href, RAW / f"ls_st_{k:02d}.tif", bbox_utm)
    qa = clip(it.assets["qa_pixel"].href, RAW / f"ls_qa_{k:02d}.tif", bbox_utm)
    meta["landsat"].append({"k": k, "id": it.id, "datetime": it.datetime.isoformat(),
                            "platform": it.properties["platform"],
                            "cloud_cover": it.properties["eo:cloud_cover"],
                            "wrs": f'{it.properties.get("landsat:wrs_path")}/{it.properties.get("landsat:wrs_row")}'})
json.dump(meta, open(PROC / "satellite_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))

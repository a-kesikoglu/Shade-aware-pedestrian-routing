"""Clip the 1 m global canopy height map (Tolan et al., 2024; AWS Open Data) to the study area."""
import mercantile
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from pyproj import Transformer
from config import *

URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm/{}.tif"
tr = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
m = HALF_SIZE_M + 400
bb = transform_bounds(CRS_UTM, "EPSG:4326", cx - m, cy - m, cx + m, cy + m)
tiles = [mercantile.quadkey(t) for t in mercantile.tiles(*bb, zooms=9)]
print("quadkeys:", tiles)
srcs = [rasterio.open(URL.format(q)) for q in tiles]
crs = srcs[0].crs
b = transform_bounds("EPSG:4326", crs, *bb)
arr, tf = merge(srcs, bounds=b)
prof = srcs[0].profile.copy()
prof.update(height=arr.shape[1], width=arr.shape[2], transform=tf, driver="GTiff",
            compress="deflate", tiled=False)
prof.pop("blockxsize", None); prof.pop("blockysize", None)
with rasterio.open(RAW / "canopy_height_meta.tif", "w", **prof) as dst:
    dst.write(arr)
print(crs, arr.shape, arr.dtype, float(arr.max()))

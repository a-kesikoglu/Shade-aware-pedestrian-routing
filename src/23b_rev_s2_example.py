"""Example window for Figure 9a: true colour Sentinel-2 image of the lowest-sun scene and the modelled shaded share
on its 10 m grid (central 1.2 km x 1.2 km). Output: data/processed/rev_s2_example.npz"""
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
import pvlib
import pystac_client
import planetary_computer as pc
from config import *
from shade import cast

meta = json.load(open(PROC / "rev_s2_validation.json"))
sc = min(meta["scenes"], key=lambda x: x["elev"])
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
it = cat.get_collection("sentinel-2-l2a").get_item(sc["id"])
from pyproj import Transformer
cx, cy = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True).transform(CENTER_LON, CENTER_LAT)
h = 600
B = (np.floor((cx - h) / 10) * 10, np.floor((cy - h) / 10) * 10, np.floor((cx + h) / 10) * 10, np.floor((cy + h) / 10) * 10)
rgb = []
for b in ["B04", "B03", "B02"]:
    with rasterio.open(it.assets[b].href) as src:
        win = from_bounds(*B, transform=src.transform).round_offsets().round_lengths()
        rgb.append(src.read(1, window=win).astype("float32"))
rgb = np.dstack(rgb)
with rasterio.open(PROC / "bld_h_1m.tif") as s:
    bh = s.read(1); TF1 = s.transform
ch = rasterio.open(PROC / "canopy_h_1m.tif").read(1)
ts = pd.Timestamp(sc["datetime"])
sp = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([ts]), CENTER_LAT, CENTER_LON)
tr = cast(bh, ch, float(sp.azimuth.iloc[0]), float(sp.apparent_elevation.iloc[0]))
nr, nc = int((B[3] - B[1]) / 10), int((B[2] - B[0]) / 10)
shade10 = np.zeros((nr, nc), "float32")
reproject((1 - tr).astype("float32"), shade10, src_transform=TF1, src_crs=CRS_UTM,
          dst_transform=rasterio.transform.from_origin(B[0], B[3], 10, 10), dst_crs=CRS_UTM, resampling=Resampling.average)
np.savez_compressed(PROC / "rev_s2_example.npz", rgb=rgb[:nr, :nc], shade10=shade10, extent=np.array(B))
print("saved", sc["id"], rgb.shape, shade10.shape)

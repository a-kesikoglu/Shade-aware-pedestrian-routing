"""Download the 3D-GloBFP tile covering Kayseri (grid 1452) and clip it to the study area."""
import glob
import zipfile
import requests
import geopandas as gpd
from config import *

URL = "https://ndownloader.figshare.com/files/54039878"   # 1452_35.0_37.5_37.5_40.0_TU.zip
d = RAW / "globfp"
d.mkdir(exist_ok=True)
z = d / "g1452.zip"
if not z.exists():
    with requests.get(URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(z, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
zipfile.ZipFile(z).extractall(d / "g1452")

aoi = gpd.read_file(PROC / "aoi.gpkg")
bb = aoi.buffer(MARGIN, join_style=2).to_crs(4326).total_bounds
g = gpd.read_file(glob.glob(str(d / "g1452" / "*.shp"))[0], bbox=tuple(bb))
g.to_crs(CRS_UTM).to_file(RAW / "globfp_buildings.gpkg", driver="GPKG")
print(len(g), "buildings")

"""Additional open data for the verification and sensitivity checks:
- OSM walk network for the study area plus a 300 m margin (network boundary test)
- OSM points of interest (alternative demand model)
- Copernicus DEM GLO-30 (terrain test)
- Natural Earth admin-0 boundaries (locator map)
"""
import json
import zipfile
import io
import requests
import numpy as np
import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
import planetary_computer as pc
import pystac_client
from shapely.geometry import box
from pyproj import Transformer
from config import *

ox.settings.use_cache = True
ox.settings.overpass_url = "https://maps.mail.ru/osm/tools/overpass/api"
ox.settings.requests_timeout = 300
ox.settings.overpass_rate_limit = False
ox.settings.cache_folder = str(RAW / "osm_cache")

tr = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
aoi = box(cx - HALF_SIZE_M, cy - HALF_SIZE_M, cx + HALF_SIZE_M, cy + HALF_SIZE_M)
ext = gpd.GeoSeries([aoi.buffer(300, join_style=2)], crs=CRS_UTM).to_crs(4326).iloc[0]
meta = {}

if not (RAW / "walk_network_ext300.graphml").exists():
    G = ox.graph_from_polygon(ext, network_type="walk", simplify=True, retain_all=False)
    ox.save_graphml(G, RAW / "walk_network_ext300.graphml")
    meta["ext_network_edges"] = len(G.edges)

if not (RAW / "osm_pois.gpkg").exists():
    tags = {"amenity": True, "shop": True, "office": True, "public_transport": ["platform", "stop_position"],
            "highway": ["bus_stop"], "railway": ["tram_stop", "station"], "tourism": True, "leisure": ["park"]}
    p = ox.features_from_polygon(gpd.GeoSeries([aoi], crs=CRS_UTM).to_crs(4326).iloc[0], tags=tags)
    p = p.reset_index()
    p["geometry"] = p.to_crs(CRS_UTM).geometry.representative_point().to_crs(4326)
    keep = [c for c in ["element", "id", "amenity", "shop", "office", "public_transport", "highway", "railway",
                        "tourism", "leisure"] if c in p.columns]
    p[keep + ["geometry"]].to_crs(CRS_UTM).to_file(RAW / "osm_pois.gpkg", driver="GPKG")
    meta["n_pois"] = len(p)

if not (RAW / "copdem30.tif").exists():
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    bb = transform_bounds(CRS_UTM, "EPSG:4326", *aoi.buffer(700, join_style=2).bounds)
    items = list(cat.search(collections=["cop-dem-glo-30"], bbox=bb).items())
    srcs = [rasterio.open(i.assets["data"].href) for i in items]
    arr, tf = merge(srcs, bounds=bb)
    prof = srcs[0].profile.copy()
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tf, driver="GTiff", compress="deflate",
                tiled=False)
    prof.pop("blockxsize", None); prof.pop("blockysize", None)
    with rasterio.open(RAW / "copdem30.tif", "w", **prof) as d:
        d.write(arr)
    meta["dem_items"] = [i.id for i in items]

if not (RAW / "ne_admin0.gpkg").exists():
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=300).content))
    d = RAW / "ne_admin0"; d.mkdir(exist_ok=True); z.extractall(d)
    g = gpd.read_file(next(d.glob("*.shp")))
    g = g[g.geometry.intersects(box(20, 30, 50, 46))][["ADMIN", "ISO_A3", "geometry"]]
    g.to_file(RAW / "ne_admin0.gpkg", driver="GPKG")
    meta["ne_countries"] = len(g)

print(json.dumps(meta, indent=1))
json.dump(meta, open(PROC / "extra_download_meta.json", "w"), indent=1)

"""Download the OSM pedestrian network and building footprints for the study area."""
import json
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
from config import *

ox.settings.use_cache = True
ox.settings.overpass_url = "https://maps.mail.ru/osm/tools/overpass/api"
ox.settings.requests_timeout = 300
ox.settings.overpass_rate_limit = False
ox.settings.log_console = True
ox.settings.cache_folder = str(RAW / "osm_cache")

tr = Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True)
cx, cy = tr.transform(CENTER_LON, CENTER_LAT)
aoi_utm = box(cx - HALF_SIZE_M, cy - HALF_SIZE_M, cx + HALF_SIZE_M, cy + HALF_SIZE_M)
# building download with a margin so that shadows cast from outside reach the AOI
dl = gpd.GeoSeries([aoi_utm.buffer(MARGIN, join_style=2)], crs=CRS_UTM).to_crs(4326).iloc[0]
gpd.GeoDataFrame(geometry=[aoi_utm], crs=CRS_UTM).to_file(PROC / "aoi.gpkg", driver="GPKG")

if (RAW / "walk_network.graphml").exists():
    G = ox.load_graphml(RAW / "walk_network.graphml")
else:
    G = ox.graph_from_polygon(gpd.GeoSeries([aoi_utm], crs=CRS_UTM).to_crs(4326).iloc[0],
                              network_type="walk", simplify=True, retain_all=False)
    ox.save_graphml(G, RAW / "walk_network.graphml")

b = ox.features_from_polygon(dl, tags={"building": True})
b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
keep = [c for c in ["building", "building:levels", "height", "roof:levels", "name", "amenity"] if c in b.columns]
b = b[keep + ["geometry"]].reset_index()
b.to_crs(CRS_UTM).to_file(RAW / "osm_buildings.gpkg", driver="GPKG")

stats = {"n_nodes": len(G.nodes), "n_edges": len(G.edges), "n_buildings": len(b),
         "share_levels": float(b["building:levels"].notna().mean()) if "building:levels" in b else 0.0,
         "share_height": float(b["height"].notna().mean()) if "height" in b else 0.0}
print(json.dumps(stats, indent=1))
json.dump(stats, open(PROC / "osm_stats.json", "w"), indent=1)

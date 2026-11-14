"""List of the network length removed with the vehicle underpasses (Section 2.2).

Every segment of the walk network inside the study area that does not enter the final network is assigned to
- "vehicle underpass": a segment of a tunnel way of a trunk or primary road or its link, or a sunken ramp of such a
  chain within 30 m of the remaining network (rule of network.build_segments), or
- "disconnected after removal": a segment that lost its connection to the largest component.
Removed segments are joined to the nearest OSM tunnel way (id, name, highway). Output:
outputs/tables/rev_removed_segments.csv (one row per OSM way and reason) and data/processed/rev_removed_segments.json.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from config import *
from network import build_segments

aoi = gpd.read_file(PROC / "aoi.gpkg").geometry.iloc[0]
seg, _, removed = build_segments(RAW / "walk_network.graphml", aoi, return_removed=True)
full_km = build_segments.full_km
final_km = seg.length.sum() / 1000
tw = gpd.read_file(RAW / "osm_tunnel_ways.gpkg")

# vehicle segments: nearest tunnel way of a vehicle class
rem = removed.copy()
rem["length"] = rem.geometry.length
mid = gpd.GeoDataFrame(geometry=rem.geometry.interpolate(0.5, normalized=True), crs=CRS_UTM)
veh = tw[tw.highway.astype(str).str.contains("trunk|primary|secondary|tertiary")].reset_index(drop=True)
j = gpd.sjoin_nearest(mid, veh[["osmid", "name", "highway", "geometry"]], distance_col="dist")
j = j[~j.index.duplicated()]
rem["osmid"] = j["osmid"].values; rem["name"] = j["name"].values; rem["way_highway"] = j["highway_right"].values \
    if "highway_right" in j else j["highway"].values
rem["dist_to_tunnel_way_m"] = j["dist"].values
rem["reason"] = np.where(rem.dist_to_tunnel_way_m <= 3.0, "segment on vehicle tunnel way",
                         "sunken ramp within 30 m of the remaining network")
c = rem.to_crs(4326).geometry.centroid if False else gpd.GeoSeries(mid.geometry, crs=CRS_UTM).to_crs(4326)
rem["lon"], rem["lat"] = c.x.values, c.y.values
tab = (rem.groupby(["osmid", "name", "way_highway", "reason"], dropna=False)
          .agg(segments=("length", "size"), length_m=("length", "sum"), lon=("lon", "mean"), lat=("lat", "mean"))
          .reset_index().sort_values(["name", "osmid"]))
veh_km = rem.length.sum() / 1000
disc_km = full_km - final_km - veh_km
tab = pd.concat([tab, pd.DataFrame([{"osmid": "", "name": "", "way_highway": "", "reason":
                                     "disconnected from the largest component after removal",
                                     "segments": np.nan, "length_m": 1000 * disc_km, "lon": np.nan, "lat": np.nan}])])
tab["length_m"] = tab.length_m.round(1); tab["lon"] = tab.lon.round(6); tab["lat"] = tab.lat.round(6)
tab.to_csv(TAB / "rev_removed_segments.csv", index=False, encoding="utf-8-sig")
out = {"full_km": full_km, "final_km": final_km, "removed_km": full_km - final_km, "vehicle_km": veh_km,
       "disconnected_km": disc_km, "n_ways": int(rem.osmid.nunique()),
       "by_name_km": (rem.groupby("name").length.sum() / 1000).round(3).to_dict()}
json.dump(out, open(PROC / "rev_removed_segments.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))

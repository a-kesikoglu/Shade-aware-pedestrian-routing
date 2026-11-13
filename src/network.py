"""Densified pedestrian graph (segments of at most SEG_MAX m) clipped to a polygon.

Underpasses (OSM ways tagged tunnel, geometries in osm_tunnel_ways.gpkg):
- vehicle underpasses (tunnel ways of trunk, primary, secondary, tertiary and their links) have no pedestrian
  access; segments of the simplified edges containing them are removed where they lie on the tunnel way or run
  within 30 m of the remaining network (sunken ramps beside the at-grade carriageway);
- segments on tunnel ways of other classes (underground pedestrian passages) are flagged as covered."""
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
from shapely.ops import substring
from config import *

MOTOR = ("trunk", "primary", "secondary", "tertiary")


def tunnel_class(highway, tunnel):
    """'' (open), 'vehicle' or 'covered' (pedestrian underpass)."""
    if str(tunnel) in ("nan", "None", "no", ""):
        return ""
    return "vehicle" if any(m in str(highway) for m in MOTOR) else "covered"


def build_segments(graphml, poly, return_removed=False):
    G = ox.project_graph(ox.load_graphml(graphml), to_crs=CRS_UTM)
    Gu = ox.convert.to_undirected(G)
    _, edges = ox.graph_to_gdfs(Gu)
    edges = edges.reset_index()
    pair_count = edges.groupby([edges[["u", "v"]].min(axis=1), edges[["u", "v"]].max(axis=1)]).size()
    npos = {n: (d["x"], d["y"]) for n, d in Gu.nodes(data=True)}
    rows, nid, removed = [], 10 ** 10, []
    for e in edges.itertuples():
        if e.u == e.v:
            continue
        hw = e.highway if isinstance(e.highway, str) else str(e.highway)
        vch = tunnel_class(hw, getattr(e, "tunnel", "nan")) == "vehicle"
        geom = e.geometry if e.geometry is not None else LineString([npos[e.u], npos[e.v]])
        L = geom.length
        if L < 0.5:
            continue
        k = max(1, int(np.ceil(L / SEG_MAX)))
        if pair_count[(min(e.u, e.v), max(e.u, e.v))] > 1:
            k = max(k, 2)
        if np.hypot(geom.coords[0][0] - npos[e.u][0], geom.coords[0][1] - npos[e.u][1]) > \
           np.hypot(geom.coords[-1][0] - npos[e.u][0], geom.coords[-1][1] - npos[e.u][1]):
            geom = LineString(geom.coords[::-1])
        seq = [e.u]
        for i in range(1, k):
            nid += 1
            p = geom.interpolate(i * L / k)
            npos[nid] = (p.x, p.y)
            seq.append(nid)
        seq.append(e.v)
        for i in range(k):
            rows.append((seq[i], seq[i + 1], L / k, hw, vch, substring(geom, i * L / k, (i + 1) * L / k)))
    seg = gpd.GeoDataFrame(rows, columns=["u", "v", "length", "highway", "vchain", "geometry"], crs=CRS_UTM)
    seg["length"] = seg.geometry.length
    seg["covered"] = False
    inp = seg[seg.geometry.interpolate(0.5, normalized=True).within(poly)]
    g0 = nx.Graph(); g0.add_edges_from(zip(inp.u, inp.v))
    cc0 = max(nx.connected_components(g0), key=len)
    build_segments.full_km = float(inp[inp.u.isin(cc0) & inp.v.isin(cc0)].length.sum() / 1000)
    tw = RAW / "osm_tunnel_ways.gpkg"
    tw = gpd.read_file(tw) if tw.exists() else None
    removed = seg.iloc[:0]
    if tw is not None and seg.vchain.any():
        mid = seg.geometry.interpolate(0.5, normalized=True)
        veh = tw[[tunnel_class(h, t) == "vehicle" for h, t in zip(tw.highway, tw.tunnel)]]
        on_tunnel = mid.within(veh.geometry.buffer(3.0).union_all()).values
        rest = seg[~seg.vchain].geometry.union_all()
        near = np.array([m.distance(rest) <= 30.0 for m in mid])
        drop = seg.vchain.values & (on_tunnel | near)
        removed = seg[drop]
        seg = seg[~drop].reset_index(drop=True)
    seg = seg[seg.geometry.interpolate(0.5, normalized=True).within(poly)].reset_index(drop=True)
    g = nx.Graph(); g.add_edges_from(zip(seg.u, seg.v))
    cc = max(nx.connected_components(g), key=len)
    seg = seg[seg.u.isin(cc) & seg.v.isin(cc)].reset_index(drop=True)
    pos = {n: npos[n] for n in set(seg.u) | set(seg.v)}
    if tw is not None:
        ped = tw[[tunnel_class(h, t) == "covered" for h, t in zip(tw.highway, tw.tunnel)]]
        if len(ped):
            zone = ped.geometry.buffer(1.5).union_all()
            inz = seg.geometry.interpolate(0.5, normalized=True).within(zone).values
            seg["covered"] = inz & ~seg.highway.str.contains("|".join(MOTOR)).values
    seg = seg.drop(columns="vchain")
    if return_removed:
        rem = removed[removed.geometry.interpolate(0.5, normalized=True).within(poly)]
        return seg, pos, rem[["highway", "geometry"]]
    return seg, pos

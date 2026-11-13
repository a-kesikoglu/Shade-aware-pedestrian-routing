"""Priority ranking checks: demand weights split across nodes, demand from corrected storey counts,
100 random placements for every outcome of Table 5, names of the priority corridors, sidewalk and
crossing tags along them, and the list of removed vehicle underpasses.
Outputs: data/processed/rev_priority.json, outputs/tables/rev_random100.csv, outputs/tables/rev_corridors.csv"""
import importlib.util
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy.spatial import cKDTree
from shapely.geometry import box
from config import *

spec = importlib.util.spec_from_file_location("pc13", Path(__file__).with_name("13_priority_checks.py"))
# reuse the helpers of 13_priority_checks without rerunning it: exec only its definitions
src = open(Path(__file__).with_name("13_priority_checks.py"), encoding="utf-8").read()
head = src.split("# ---------------------------------------------------------------- 1 concentration curve")[0]
G = {"__name__": "pc13"}
exec(compile(head, "13_head", "exec"), G)
seg, L, net, OD, SH, HRS = G["seg"], G["L"], G["net"], G["OD"], G["SH"], G["HRS"]
day_metrics, b10_15, select, jac_len, count_edges, sp_edges = (G[k] for k in
    ["day_metrics", "b10_15", "select", "jac_len", "count_edges", "sp_edges"])
BASE_D, BASE_C, BASE_B, TB, FLOW = G["BASE_D"], G["BASE_C"], G["BASE_B"], G["TB"], G["FLOW"]
s_day = SH.mean(0)
OUT = {}
REF5 = select(TB, 0.05)


def sample_od(p, n, seed, dmin=500, dmax=2500):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        o, d = rng.choice(net.N, 2, p=p)
        if o == d:
            continue
        l0 = net.sssp(L, o)[d]
        if dmin <= l0 <= dmax:
            out.append((int(o), int(d), float(l0)))
    return out


def evaluate_demand(p, tag):
    ev = sample_od(p, 500, SEED + 30); fl_od = sample_od(p, 5000, SEED + 31)
    fl = count_edges([sp_edges(o, d) for o, d, _ in fl_od])
    ec = count_edges([sp_edges(o, d) for o, d, _ in ev])
    tb = fl * s_day
    b0 = day_metrics(np.array([], int), ec=ec)[0]
    r = {"top5_jaccard_vs_ref": jac_len(REF5, select(tb, .05)),
         "red_TB5": 100 * (1 - day_metrics(select(tb, .05), ec=ec)[0] / b0),
         "red_ref_selection5": 100 * (1 - day_metrics(REF5, ec=ec)[0] / b0),
         "red_random5": float(np.mean([100 * (1 - day_metrics(select(None, .05, np.random.default_rng(700 + j)),
                                                                ec=ec)[0] / b0) for j in range(100)]))}
    print(tag, r, flush=True)
    return r


# ---------------------------------------------------------------- 1 demand weights
nodes = pd.read_csv(PROC / "network_nodes.csv").set_index("node").loc[net.nodes]
xy = nodes[["x", "y"]].values
bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
bc = bld.geometry.centroid
kt = cKDTree(xy)
nb_b = kt.query_ball_point(np.c_[bc.x, bc.y], r=50)            # nodes within 50 m of each building
near = kt.query(np.c_[bc.x, bc.y])[1]


def weights(storey_h=None, split=False):
    h = bld.h.values if storey_h is None else storey_h
    gfa = bld.geometry.area.values * np.maximum(1, np.round(h / FLOOR_HEIGHT))
    w = np.ones(net.N)
    for j, ns in enumerate(nb_b):
        if not ns:
            continue
        if split:
            w[ns] += gfa[j] / len(ns)
        else:
            w[ns] += gfa[j]
    return w / w.sum()


p_ref = weights()
OUT["weights_check_ref_vs_stored"] = float(np.corrcoef(p_ref, G["WN"] / G["WN"].sum())[0, 1])
OUT["demand"] = {}
p_split = weights(split=True)
OUT["demand"]["split"] = evaluate_demand(p_split, "split")
OUT["demand"]["split"]["weight_corr_vs_ref"] = float(np.corrcoef(p_split, p_ref)[0, 1])
rh = json.load(open(PROC / "rev_heights.json"))
perturb = np.isin(bld.h_src.values, ["globfp", "globfp_footprint", "imputed"])
gb = np.digitize(bld.h.values, [10, 20])
f_cls = np.array([rh["by_glo_class"][c]["ratio_med"] for c in ["<10", "10-20", ">=20"]])[gb]
for tag, f in [("corrected_uniform", rh["all"]["ratio_med"]), ("corrected_class", f_cls)]:
    hc = np.maximum(3, bld.h.values * np.where(perturb, f, 1))
    OUT["demand"][tag] = evaluate_demand(weights(storey_h=hc), tag)

# ---------------------------------------------------------------- 2 random placement, 100 draws, all outcomes
rows = []
for frac in [0.01, 0.03, 0.05, 0.10]:
    rng = np.random.default_rng(500 + int(frac * 100))
    for k in range(100):
        tr = select(None, frac, rng)
        d, c = day_metrics(tr)
        rows.append({"frac": frac, "draw": k, "km": L[tr].sum() / 1000, "red_dsun_sp_day": 100 * (1 - d / BASE_D),
                     "red_cost_sp_day": 100 * (1 - c / BASE_C), "red_dsun_b10_15": 100 * (1 - b10_15(tr) / BASE_B)})
    print("random", frac, flush=True)
rnd = pd.DataFrame(rows)
rnd.to_csv(TAB / "rev_random100.csv", index=False)
OUT["random100"] = {f"{f:g}": {c: {"mean": float(g[c].mean()), "p2_5": float(g[c].quantile(.025)),
                                   "p97_5": float(g[c].quantile(.975))}
                               for c in ["red_dsun_sp_day", "red_cost_sp_day", "red_dsun_b10_15", "km"]}
                    for f, g in rnd.groupby("frac")}

# ---------------------------------------------------------------- 3 names of the priority corridors
G_osm = ox.load_graphml(RAW / "walk_network.graphml")
edges = ox.graph_to_gdfs(G_osm, nodes=False).to_crs(CRS_UTM).reset_index()
edges["name"] = edges["name"].apply(lambda x: x[0] if isinstance(x, list) else x)
mid = seg.geometry.interpolate(0.5, normalized=True)
ei = edges.sindex.nearest(mid, return_all=False)[1]
seg["name"] = edges["name"].values[ei]
seg["osmid"] = edges["osmid"].apply(lambda x: x[0] if isinstance(x, list) else x).values[ei]
rows = []
for frac in [0.01, 0.05]:
    sel = select(TB, frac)
    d = seg.iloc[sel].assign(km=lambda x: x.length / 1000)
    g = d.groupby(d.name.fillna("(unnamed)")).km.sum().sort_values(ascending=False)
    for n, km in g.items():
        rows.append({"level": f"{frac:.0%}", "street": n, "km": km, "share": km / d.km.sum()})
cor = pd.DataFrame(rows)
cor.to_csv(TAB / "rev_corridors.csv", index=False)
OUT["corridors"] = {lvl: g.head(12)[["street", "km", "share"]].to_dict(orient="records") for lvl, g in cor.groupby("level")}

# ---------------------------------------------------------------- 4 sidewalk and crossing tags along the top 5%
aoi = gpd.read_file(PROC / "aoi.gpkg").to_crs(4326).geometry.iloc[0]
feat = ox.features_from_polygon(aoi, {"highway": ["crossing", "traffic_signals"], "footway": ["sidewalk", "crossing"],
                                      "crossing": True, "sidewalk": True})
feat = feat.to_crs(CRS_UTM)
top = seg.iloc[REF5]
topu = top.geometry.buffer(15).union_all()
art = top[top.highway.astype(str).str.contains("primary|trunk")]
ways = edges[edges.osmid.isin(set(art.osmid))]
sw = []
pts = feat[feat.geom_type == "Point"]
cross = pts[(pts.get("highway") == "crossing") | pts.get("crossing").notna()] if "crossing" in pts else pts[pts.get("highway") == "crossing"]
sig = pts[pts.get("highway") == "traffic_signals"]
lines = feat[feat.geom_type.isin(["LineString", "MultiLineString"])]
side_lines = lines[lines.get("footway") == "sidewalk"] if "footway" in lines else lines.iloc[:0]
OUT["sidewalks"] = {
    "top5_km": float(top.length.sum() / 1000), "top5_trunk_primary_km": float(art.length.sum() / 1000),
    "crossings_within_15m": int(cross.within(topu).sum()), "signals_within_15m": int(sig.within(topu).sum()),
    "crossings_per_km": float(cross.within(topu).sum() / (top.length.sum() / 1000)),
    "separate_sidewalk_km_within_15m": float(side_lines.intersection(topu).length.sum() / 1000),
    "sidewalk_tag_values_on_ways": {}}
raw = gpd.read_file(RAW / "osm_tunnel_ways.gpkg")
# sidewalk tags of the arterial ways (from the raw OSM ways of the graph)
try:
    tags = ox.features_from_polygon(aoi, {"highway": ["primary", "trunk", "primary_link", "trunk_link"]})
    tags = tags.reset_index()
    tags = tags[tags.id.isin(set(art.osmid))] if "id" in tags else tags
    col = tags["sidewalk"] if "sidewalk" in tags else pd.Series(["(none)"] * len(tags))
    OUT["sidewalks"]["sidewalk_tag_values_on_ways"] = col.fillna("(no tag)").value_counts().to_dict()
    OUT["sidewalks"]["arterial_ways_checked"] = int(len(tags))
except Exception as e:                       # tag lookup is descriptive only
    OUT["sidewalks"]["error"] = str(e)
OUT["sidewalks"]["osm_retrieved"] = pd.Timestamp.now().strftime("%Y-%m-%d")

# ---------------------------------------------------------------- 5 removed vehicle underpasses
tun = raw.to_crs(CRS_UTM)
veh = tun[tun.highway.astype(str).str.contains("trunk|primary")]
veh = veh[veh.intersects(box(*gpd.read_file(PROC / "aoi.gpkg").total_bounds).buffer(50))]
OUT["underpasses"] = veh.assign(m=veh.length).groupby(veh.name.fillna("(unnamed link)")).m.sum().round(0).to_dict()
json.dump(OUT, open(PROC / "rev_priority.json", "w"), indent=1, default=str)
print(json.dumps({k: OUT[k] for k in OUT if k != "random100"}, indent=1, default=str, ensure_ascii=False))

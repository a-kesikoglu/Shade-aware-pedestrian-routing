"""Overlap between the 500 evaluation pairs and the 5000 flow pairs (same pair or reversed), duplicates within
each set, and the occupancy of the 36 origin cells used by the block bootstrap.
Output: data/processed/rev_od_overlap.json"""
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from config import *

d = pickle.load(open(PROC / "od_sets.pkl", "rb"))
ev = [(int(o), int(t)) for o, t, *_ in d["OD"]]
fl = [(int(o), int(t)) for o, t, *_ in d["OD_FLOW"]]
sf = set(fl); sfu = {frozenset(p) for p in fl}
out = {"n_eval": len(ev), "n_flow": len(fl),
       "eval_in_flow_same_direction": int(sum(p in sf for p in ev)),
       "eval_in_flow_either_direction": int(sum(frozenset(p) in sfu for p in ev)),
       "eval_duplicates": int(len(ev) - len(set(ev))), "flow_duplicates": int(len(fl) - len(sf)),
       "eval_origins_unique": int(len({o for o, _ in ev})), "eval_destinations_unique": int(len({t for _, t in ev}))}
# origin cells of the block bootstrap (500 m grid anchored at the south-west corner of the study area)
from impedance import load_segments
from routing import Net
seg = load_segments()
net = Net(seg.u.values, seg.v.values, seg.length.values)
nodes = pd.read_csv(PROC / "network_nodes.csv").set_index("node")
x0, y0 = gpd.read_file(PROC / "aoi.gpkg").total_bounds[:2]
xy = nodes.loc[net.nodes[[o for o, _ in ev]]][["x", "y"]].values
cx = np.floor((xy[:, 0] - x0) / 500).astype(int); cy = np.floor((xy[:, 1] - y0) / 500).astype(int)
cnt = pd.Series(1, index=pd.MultiIndex.from_arrays([cx, cy])).groupby(level=[0, 1]).sum()
out["origin_cells"] = {"non_empty": int(len(cnt)), "sum": int(cnt.sum()), "min": int(cnt.min()), "max": int(cnt.max()),
                       "median": float(cnt.median()), "outside_6x6": int(((cx < 0) | (cx > 5) | (cy < 0) | (cy > 5)).sum())}
json.dump(out, open(PROC / "rev_od_overlap.json", "w"), indent=1)
print(json.dumps(out, indent=1))

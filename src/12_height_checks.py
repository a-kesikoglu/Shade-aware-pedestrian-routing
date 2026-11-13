"""Building height checks.

Part A  reference sample: split by tag type (metres vs storeys) and height class, reconcile building counts,
        quantify one-to-many matches between OSM footprints and 3D-GloBFP polygons.
Part B  Monte Carlo with dependent errors: one multiplier per 3D-GloBFP source polygon ("grouped") and one
        multiplier per 250 m block ("block"), 30 realisations each, plus convergence of the independent run.
Outputs: data/processed/height_checks.json, data/processed/height_mc_dependent.pkl
"""
import json
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from config import *

K = 30


# ---------------------------------------------------------------- Part A
def part_a():
    aoi = gpd.read_file(PROC / "aoi.gpkg").geometry.iloc[0]
    osm = gpd.read_file(RAW / "osm_buildings.gpkg").to_crs(CRS_UTM)
    osm = osm[osm.geometry.area > 4].reset_index(drop=True)
    osm["bid"] = np.arange(len(osm))
    glo = gpd.read_file(RAW / "globfp_buildings.gpkg").to_crs(CRS_UTM)
    glo = glo[glo.geometry.is_valid & (glo.geometry.area > 4)].reset_index(drop=True)
    glo["gid"] = np.arange(len(glo))
    inter = gpd.overlay(osm[["bid", "geometry"]], glo[["gid", "Height", "geometry"]], how="intersection",
                        keep_geom_type=True)
    inter["a"] = inter.geometry.area
    dom = inter.sort_values("a").groupby("bid").tail(1).set_index("bid")["gid"]
    # one-to-many matches counted on the dominant (largest-overlap) source polygon of each matched footprint
    cov0 = inter.groupby("bid")["a"].sum() / osm.set_index("bid").geometry.area
    matched_bids = cov0[cov0 >= 0.2].index
    n_osm_per_gid = dom.loc[matched_bids].value_counts()
    cov = inter.groupby("bid")["a"].sum() / osm.set_index("bid").geometry.area
    osm["cov_glo"] = osm.bid.map(cov).fillna(0)
    osm["gid_dom"] = osm.bid.map(dom)
    osm["merged"] = osm.gid_dom.map(n_osm_per_gid).fillna(0) >= 2

    lev = pd.to_numeric(osm["building:levels"], errors="coerce")
    hgt = pd.to_numeric(osm["height"].astype(str).str.replace("m", "").str.strip(), errors="coerce")
    osm["tag"] = np.where(hgt.notna(), "height", np.where(lev.notna(), "levels", "none"))
    osm["h_osm"] = np.where(hgt.notna(), hgt, lev * FLOOR_HEIGHT)
    bad = (osm.h_osm <= 0) | (osm.h_osm > 200)
    osm.loc[bad, ["h_osm"]] = np.nan
    osm.loc[bad, "tag"] = "none"
    in_aoi = osm.intersects(aoi)

    v = pd.read_csv(PROC / "height_validation.csv").merge(osm[["bid", "tag", "merged", "cov_glo"]], on="bid")
    v["in_aoi"] = v.bid.map(in_aoi)
    v["err"] = v.h_glo - v.h_osm

    def st(d):
        e = d.err
        return {"n": int(len(d)), "bias_m": float(e.mean()) if len(d) else None,
                "rmse_m": float(np.sqrt((e ** 2).mean())) if len(d) else None,
                "mae_m": float(e.abs().mean()) if len(d) else None,
                "median_abs_rel_err": float((e.abs() / d.h_osm).median()) if len(d) else None,
                "r": float(np.corrcoef(d.h_osm, d.h_glo)[0, 1]) if len(d) > 2 else None}

    v["hcls"] = pd.cut(v.h_osm, [0, 10, 20, 400], labels=["<10", "10-20", ">=20"])
    out = {"all": st(v), "by_tag": {k: st(g) for k, g in v.groupby("tag")},
           "by_hclass": {str(k): st(g) for k, g in v.groupby("hcls", observed=True)},
           "by_merged": {("merged" if k else "single"): st(g) for k, g in v.groupby("merged")},
           "in_aoi": st(v[v.in_aoi]),
           "spearman": float(pd.Series(v.h_osm).corr(v.h_glo, method="spearman"))}
    # counts: why 37 OSM-height buildings in the AOI but 62 reference buildings
    tagged = osm[osm.tag != "none"]
    out["counts"] = {
        "osm_footprints_with_margin": int(len(osm)),
        "osm_tagged_with_margin": int(len(tagged)),
        "osm_tagged_in_aoi": int(in_aoi[tagged.index].sum()),
        "osm_tagged_with_glo_cov20": int((tagged.cov_glo >= 0.2).sum()),
        "reference_sample": int(len(v)), "reference_in_aoi": int(v.in_aoi.sum()),
        "osm_with_glo_cov20": int((osm.cov_glo >= 0.2).sum()),
        "osm_cov_lt20": int((osm.cov_glo < 0.2).sum()),
        "osm_matched_merged_share": float(osm.loc[osm.cov_glo >= 0.2, "merged"].mean()),
        "glo_polygons": int(len(glo)),
        "glo_polygons_matching_2plus_osm": int((n_osm_per_gid >= 2).sum()),
        "glo_polygons_matching_5plus_osm": int((n_osm_per_gid >= 5).sum()),
        "match_aoi": (lambda m: {"osm_matched": int(len(m)), "glo_dominant_polygons": int(m.gid_dom.nunique()),
                                 "glo_shared_polygons": int(m.loc[m.merged, "gid_dom"].nunique()),
                                 "osm_in_shared": int(m.merged.sum()), "osm_in_shared_share": float(m.merged.mean())})(
            osm[in_aoi & (osm.cov_glo >= 0.2)]),
        "cov_quartiles_matched": [float(x) for x in osm.loc[osm.cov_glo > 0, "cov_glo"].quantile([.25, .5, .75])]}
    v.to_csv(PROC / "height_validation_detail.csv", index=False)
    osm[["bid", "gid_dom", "merged", "cov_glo", "tag"]].to_csv(PROC / "osm_globfp_match.csv", index=False)
    return out


# ---------------------------------------------------------------- Part B (worker)
_G = {}


def _init():
    from impedance import load_segments
    from routing import Net
    from sampling import corridor_points
    from shade import sun_positions
    seg = load_segments()
    _G["seg"] = seg
    _G["net"] = Net(seg.u.values, seg.v.values, seg.length.values)
    _G["L"] = seg.length.values
    _G["od"] = pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"]
    _G["base"] = pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"]
    with rasterio.open(PROC / "bld_h_1m.tif") as s:
        TF = s.transform; H, W = s.shape
    _G["canopy"] = rasterio.open(PROC / "canopy_h_1m.tif").read(1)
    bmask = rasterio.open(PROC / "bld_mask_1m.tif").read(1) > 0
    bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
    order = bld.sort_values("h")
    _G["bid"] = features.rasterize(((g, i + 1) for i, g in zip(order.index, order.geometry)), out_shape=(H, W),
                                   transform=TF, dtype="int32", fill=0)
    _G["E"] = corridor_points(seg.geometry.values, bmask, TF)
    _G["sun"] = sun_positions(PERIODS).set_index("time")


def _run(args):
    from shade import cast
    from impedance import thermal_cost
    from routing import pareto_paths, budget
    from sampling import edge_sunlit
    tag, hvec = args
    if not _G:
        _init()
    seg, net, L, od, base = _G["seg"], _G["net"], _G["L"], _G["od"], _G["base"]
    Ec, Rc, Cc = _G["E"]
    bh = np.r_[0.0, hvec].astype("float32")[_G["bid"]]
    out = {}
    for t in PERIODS:
        tr = cast(bh, _G["canopy"], _G["sun"].loc[t, "azimuth"], _G["sun"].loc[t, "elevation"])
        s = edge_sunlit(tr, (Ec, Rc, Cc), seg.covered.values)
        c2 = thermal_cost(L, s, seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])
        rows = []
        for k, (o, d, _) in enumerate(od):
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
            b10 = sols[budget(sols, L0, 0.10)][2]; sp = sols[0][2]
            b0 = base[t][k]["B10"]["edges"]
            a, b = set(b10), set(b0)
            jac = L[list(a & b)].sum() / L[list(a | b)].sum()
            d_sp = (L[sp] * s[sp]).sum(); d_b = (L[b10] * s[b10]).sum(); d_b0 = (L[b0] * s[b0]).sum()
            rows.append((jac, d_sp, d_b, d_b0, (d_sp - d_b) / d_sp if d_sp > 0 else 0.0))
        out[t] = {"net_sunlit": float((L * s).sum() / L.sum()), "rows": np.array(rows)}
    print(tag, {t: round(out[t]["net_sunlit"], 4) for t in PERIODS}, flush=True)
    return tag, out


def part_b(workers=8):
    bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
    h0 = bld.h.values
    src = bld.h_src.values
    perturb = np.isin(src, ["globfp", "globfp_footprint", "imputed"])
    v = pd.read_csv(PROC / "height_validation.csv")
    r = np.log(v.h_osm / v.h_glo).values
    r_c = r - np.median(r)
    # source polygon of every building: OSM rows first (same order as in 02_surface_model), then added GloBFP
    match = pd.read_csv(PROC / "osm_globfp_match.csv")
    n_osm = len(match)
    glo = gpd.read_file(RAW / "globfp_buildings.gpkg").to_crs(CRS_UTM)
    glo = glo[glo.geometry.is_valid & (glo.geometry.area > 4)].reset_index(drop=True)
    glo["gid"] = np.arange(len(glo))
    # added GloBFP footprints: recover their gid by geometry match on centroids
    extra = bld.iloc[n_osm:]
    tree = glo.sindex
    ex_gid = []
    for g in extra.geometry:
        c = g.representative_point()
        cand = list(tree.query(c, predicate="within"))
        ex_gid.append(int(glo.gid.iloc[cand[0]]) if cand else -1)
    group = np.r_[match.gid_dom.fillna(-1).astype(int).values, np.array(ex_gid)]
    nxt = group.max() + 1
    for i in np.where(group < 0)[0]:          # no source polygon (imputed): own group
        group[i] = nxt; nxt += 1
    cen = bld.geometry.representative_point()
    block = (np.floor(cen.x / 250).astype(int) * 100000 + np.floor(cen.y / 250).astype(int)).values

    rng = np.random.default_rng(SEED + 11)
    jobs = []
    for name, lab in [("grouped", group), ("block", block)]:
        u, inv = np.unique(lab, return_inverse=True)
        for k in range(K):
            e = rng.choice(r_c, size=len(u), replace=True)[inv]
            jobs.append((f"{name}{k}", np.maximum(h0 * np.where(perturb, np.exp(e), 1.0), 3.0)))
    with Pool(workers) as pool:
        res = dict(pool.map(_run, jobs, chunksize=1))
    out = {"grouped": [res[f"grouped{k}"] for k in range(K)], "block": [res[f"block{k}"] for k in range(K)],
           "n_groups": int(len(np.unique(group[perturb]))), "n_blocks": int(len(np.unique(block[perturb]))),
           "multiplier_mean": float(np.mean(np.exp(r_c))), "multiplier_median": float(np.median(np.exp(r_c))),
           "multiplier_p5_p95": [float(np.percentile(np.exp(r_c), 5)), float(np.percentile(np.exp(r_c), 95))]}
    pickle.dump(out, open(PROC / "height_mc_dependent.pkl", "wb"))


if __name__ == "__main__":
    if "a" in sys.argv[1:] or len(sys.argv) == 1:
        A = part_a()
        json.dump(A, open(PROC / "height_checks.json", "w"), indent=1)
        print(json.dumps(A, indent=1))
    if "b" in sys.argv[1:] or len(sys.argv) == 1:
        part_b()
        print("MC dependent saved")

"""Verification and sensitivity of the shade model.

1  analytic verification of raster shadow casting (single block and vector LoD1 shadows of real buildings)
2  adequacy of the building data margin and the search cap; reference against a 300 m margin and 400 m cap
3  pedestrian underpasses (covered) and removed vehicle underpasses
4  facade-side (sidewalk) corridors against the +/-3 m centreline corridor
5  canopy parameters, buildings-only shade and terrain (Copernicus DEM) as route sensitivity runs
6  network extent: routes recomputed on the network of the study area plus a 300 m margin
Outputs: data/processed/shade_checks.json, outputs/tables/robustness_runs.csv
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
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely import affinity
from config import *
from shade import sun_positions, cast, _shift
from impedance import load_segments, thermal_cost, robust_minmax
from routing import Net, pareto_paths, budget
from sampling import corridor_points, sample_points, edge_mean, edge_sunlit

rd = lambda n: rasterio.open(PROC / f"{n}_1m.tif")
SUN = sun_positions(sorted(set(HOURLY + PERIODS))).set_index("time")


# ---------------------------------------------------------------- helpers
def vector_shadow(poly, h, az, el):
    """Ground shadow of an LoD1 prism (flat ground)."""
    L = h / np.tan(np.deg2rad(el))
    dx, dy = -L * np.sin(np.deg2rad(az)), -L * np.cos(np.deg2rad(az))
    parts = [poly, affinity.translate(poly, dx, dy)]
    for ring in [poly.exterior]:
        c = list(ring.coords)
        for a, b in zip(c[:-1], c[1:]):
            q = Polygon([a, b, (b[0] + dx, b[1] + dy), (a[0] + dx, a[1] + dy)])
            if q.is_valid and q.area > 0:
                parts.append(q)
    return unary_union(parts)


def compare(rast_shade, vec_shade, ground):
    a, b = rast_shade & ground, vec_shade & ground
    inter, union = (a & b).sum(), (a | b).sum()
    dis = a ^ b
    edge = ndimage.distance_transform_edt(~(b ^ ndimage.binary_erosion(b)))  # distance to analytic boundary
    return {"iou": float(inter / union), "agreement": float(1 - dis[ground].mean()),
            "disagree_within_1_5m": float((edge[dis] <= 1.5).mean()) if dis.any() else 1.0,
            "precision": float(inter / max(a.sum(), 1)), "recall": float(inter / max(b.sum(), 1))}


# ---------------------------------------------------------------- 1 analytic verification
def part1():
    out = {"block": [], "real": {}}
    from rasterio.transform import from_origin
    N = 400
    TF = from_origin(0, N, 1, 1)
    blk = box(190, 190, 210, 210)
    for h in (10, 20, 40):
        bh = features.rasterize([(blk, h)], out_shape=(N, N), transform=TF, dtype="float32")
        ground = bh == 0
        for el in (15, 30, 45, 60, 75):
            if h / np.tan(np.deg2rad(el)) > 180:
                continue
            for az in range(0, 360, 15):
                tr = cast(bh, np.zeros_like(bh), az, el)
                vs = features.rasterize([(vector_shadow(blk, h, az, el), 1)], out_shape=(N, N), transform=TF,
                                        dtype="uint8").astype(bool)
                r = compare(tr == 0, vs, ground); r.update(h=h, el=el, az=az)
                out["block"].append(r)
    blk_df = pd.DataFrame(out["block"])
    blk_df["slen"] = blk_df.h / np.tan(np.deg2rad(blk_df.el))
    lg = blk_df[blk_df.slen >= 10]
    out["block_summary"] = {"n": int(len(blk_df)), "iou_min": float(blk_df.iou.min()),
                            "n_long": int(len(lg)), "iou_min_long": float(lg.iou.min()),
                            "iou_mean_long": float(lg.iou.mean()),
                            "iou_mean": float(blk_df.iou.mean()),
                            "disagree_within_1_5m_min": float(blk_df.disagree_within_1_5m.min())}
    del out["block"]
    # real buildings, central 900 m window (buildings only)
    with rd("bld_h") as s:
        bh = s.read(1); TF = s.transform
    bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
    aoi = gpd.read_file(PROC / "aoi.gpkg").total_bounds
    cx, cy = (aoi[0] + aoi[2]) / 2, (aoi[1] + aoi[3]) / 2
    win = box(cx - 450, cy - 450, cx + 450, cy + 450)
    near = bld[bld.intersects(win.buffer(450))]
    r0 = int((TF.f - (cy + 450)) / 1); c0 = int((cx - 450 - TF.c) / 1)
    sl = (slice(r0, r0 + 900), slice(c0, c0 + 900))
    ground = bh[sl] == 0
    for t in PERIODS:
        az, el = SUN.loc[t, "azimuth"], SUN.loc[t, "elevation"]
        tr = cast(bh, np.zeros_like(bh), az, el)
        shp = unary_union([vector_shadow(g, h, az, el) for g, h in zip(near.geometry, near.h)
                           if g.geom_type == "Polygon"] +
                          [vector_shadow(p, h, az, el) for g, h in zip(near.geometry, near.h)
                           if g.geom_type == "MultiPolygon" for p in g.geoms])
        vs = features.rasterize([(shp, 1)], out_shape=bh.shape, transform=TF, dtype="uint8").astype(bool)
        out["real"][t] = compare(tr[sl] == 0, vs[sl], ground)
    return out


# ---------------------------------------------------------------- 2 margin and search cap
def part2():
    bld = gpd.read_file(PROC / "buildings_lod1.gpkg")
    aoi = gpd.read_file(PROC / "aoi.gpkg").geometry.iloc[0]
    ring = bld[~bld.intersects(aoi)]
    hmax = float(bld.h.max())
    out = {"margin_m": MARGIN, "cap_m": SHADOW_CAP, "h_max": hmax, "h_max_margin": float(ring.h.max()),
           "n_margin": int(len(ring)), "hours": {}}
    for t, r in SUN.iterrows():
        tan = np.tan(np.deg2rad(r.elevation))
        out["hours"][t] = {"elev": float(r.elevation), "h_reach_margin": float(MARGIN * tan),
                           "n_margin_above": int((ring.h > MARGIN * tan).sum()),
                           "cap_limits_h_above": float(SHADOW_CAP * tan),
                           "n_bld_above_cap": int((bld.h > SHADOW_CAP * tan).sum()),
                           "shadow_len_hmax": float(hmax / tan),
                           "h_reach_300m": float(300 * tan), "n_beyond300_above": int(
                               (bld[~bld.intersects(aoi.buffer(300, join_style=2))].h > 300 * tan).sum())}
    # empirical test: reference rasters against a 300 m building margin with a 400 m search cap
    seg = load_segments(); L = seg.length.values
    with rd("bld_h") as s_:
        bh = s_.read(1); TF = s_.transform
    ch = rd("canopy_h").read(1)
    bmask = rd("bld_mask").read(1) > 0
    E = corridor_points(seg.geometry.values, bmask, TF)
    inner = features.rasterize([(aoi.buffer(300, join_style=2), 1)], out_shape=bh.shape, transform=TF,
                               dtype="uint8") > 0
    bh3, ch3 = np.where(inner, bh, 0).astype("float32"), np.where(inner, ch, 0).astype("float32")
    comp = {}
    for t, r in SUN.iterrows():
        s_ref = edge_sunlit(cast(bh, ch, r.azimuth, r.elevation), E, seg.covered.values)
        s_300 = edge_sunlit(cast(bh3, ch3, r.azimuth, r.elevation, max_len=400.0), E, seg.covered.values)
        d = s_300 - s_ref
        comp[t] = {"net_sunlit_ref": float((L * s_ref).sum() / L.sum()), "net_sunlit_300": float((L * s_300).sum() / L.sum()),
                   "len_share_changed_gt005": float(L[np.abs(d) > 0.05].sum() / L.sum()),
                   "max_abs_diff": float(np.abs(d).max())}
        print("margin", t, comp[t], flush=True)
    out["vs_300m"] = comp
    return out


# ---------------------------------------------------------------- shared routing evaluation
_G = {}


def _ctx():
    if not _G:
        seg = load_segments()
        _G.update(seg=seg, L=seg.length.values, net=Net(seg.u.values, seg.v.values, seg.length.values),
                  od=pickle.load(open(PROC / "od_sets.pkl", "rb"))["OD"],
                  base=pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"])
    return _G


def eval_routes(S_var, S_eval=None, tag=""):
    """B10 routes under sunlit fractions S_var[t]; returns medians of the reduction in sunlit distance
    (evaluated in S_eval, default S_var), Jaccard with the reference B10 and identical share."""
    g = _ctx(); seg, L, net = g["seg"], g["L"], g["net"]
    S_eval = S_eval or S_var
    out = {}
    for t in PERIODS:
        c2 = thermal_cost(L, S_var[t], seg.ndvi_n.values, seg.lst_n.values, WEIGHTS["balanced"])
        red, jac, same = [], [], []
        for k, (o, d, _) in enumerate(g["od"]):
            sols, L0 = pareto_paths(net, o, d, L, c2, detour=1.5)
            b, sp = sols[budget(sols, L0, 0.10)][2], sols[0][2]
            s = S_eval[t]
            dsp, db = (L[sp] * s[sp]).sum(), (L[b] * s[b]).sum()
            red.append(100 * (1 - db / dsp) if dsp > 0 else 0.0)
            b0 = g["base"][t][k]["B10"]["edges"]
            A, B = set(b), set(b0)
            jac.append(L[list(A & B)].sum() / L[list(A | B)].sum()); same.append(A == B)
        out[t] = {"net_sunlit": float((L * S_eval[t]).sum() / L.sum()), "dDsun_med": float(np.median(red)),
                  "jaccard_med": float(np.median(jac)), "identical": float(np.mean(same))}
    print(tag, {t: round(out[t]["dDsun_med"], 1) for t in PERIODS}, flush=True)
    return out


# ---------------------------------------------------------------- 5 canopy / terrain variants (worker)
def _variant(args):
    name, kw = args
    g = _ctx(); seg = g["seg"]
    with rd("bld_h") as s:
        bh = s.read(1); TF = s.transform
    bmask = rd("bld_mask").read(1) > 0
    ndvi = rd("ndvi").read(1)
    chm = np.zeros(bh.shape, "float32")
    with rasterio.open(RAW / "canopy_height_meta.tif") as s:
        reproject(s.read(1), chm, src_transform=s.transform, src_crs=s.crs, dst_transform=TF, dst_crs=CRS_UTM,
                  resampling=Resampling.bilinear)
    hmin, nmin = kw.get("hmin", 2.5), kw.get("ndvi_min", 0.15)
    ch = np.where((chm >= hmin) & ~bmask & (ndvi >= nmin), chm, 0).astype("float32")
    if kw.get("no_canopy"):
        ch[:] = 0
    Ec, Rc, Cc = corridor_points(seg.geometry.values, bmask, TF)
    z = None
    if kw.get("terrain"):
        z = terrain(TF, bh.shape)
    S = {}
    for t in PERIODS:
        az, el = SUN.loc[t, "azimuth"], SUN.loc[t, "elevation"]
        if z is None:
            tr = cast(bh, ch, az, el, trunk=kw.get("trunk", 0.25), tau_c=kw.get("tau_c", CANOPY_TRANSMISSIVITY))
        else:
            tr = cast_terrain(bh, ch, z, az, el)
        S[t] = edge_sunlit(tr, (Ec, Rc, Cc), seg.covered.values)
    return name, eval_routes(S, tag=name)


def terrain(TF, shape):
    """Bare-earth approximation from the Copernicus GLO-30 surface model: morphological opening (210 m)
    removes buildings and trees, followed by Gaussian smoothing."""
    with rasterio.open(RAW / "copdem30.tif") as s:
        dem = s.read(1).astype("float32"); stf, scrs = s.transform, s.crs
    px = abs(stf.a) * 111320 * np.cos(np.deg2rad(CENTER_LAT))
    k = max(3, int(round(210 / px)) | 1)
    dtm = ndimage.gaussian_filter(ndimage.grey_opening(dem, size=(k, k)), sigma=2)
    z = np.zeros(shape, "float32")
    reproject(dtm, z, src_transform=stf, src_crs=scrs, dst_transform=TF, dst_crs=CRS_UTM,
              resampling=Resampling.bilinear)
    return z


def cast_terrain(bld_h, canopy_h, z, azimuth, elevation, res=1.0, max_len=SHADOW_CAP, trunk=0.25,
                 tau_c=CANOPY_TRANSMISSIVITY):
    """Shadow casting over sloping ground: obstacle tops are compared in absolute elevation."""
    az = np.deg2rad(azimuth)
    sx, sy = np.sin(az), np.cos(az)
    step = 1.0 / max(abs(sx), abs(sy))
    dz = step * res * np.tan(np.deg2rad(elevation))
    relief = float(z.max() - z.min())
    hmax = max(float(bld_h.max()), float(canopy_h.max())) + relief
    n = int(min(max_len / res, hmax / np.tan(np.deg2rad(elevation)) / res) / step) + 1
    bshade = np.zeros(bld_h.shape, bool)
    vshade = canopy_h > 0
    for k in range(1, n + 1):
        dc = int(round(k * step * sx)); dr = int(round(-k * step * sy))
        zq = _shift(z, dr, dc); valid = _shift(np.ones_like(z), dr, dc) > 0
        rel = np.where(valid, zq - z, -1e4)
        h = k * dz
        bq = _shift(bld_h, dr, dc)
        bshade |= (bq + rel) > h
        cq = _shift(canopy_h, dr, dc)
        vshade |= (cq > 0) & (cq + rel > h) & (trunk * cq + rel <= h)
    return np.where(bshade, 0.0, np.where(vshade, tau_c, 1.0)).astype("float32")


def terrain_stats():
    with rd("bld_h") as s:
        TF = s.transform; shape = s.shape
    m = int(MARGIN)
    z = terrain(TF, shape)[m:-m, m:-m]
    gy, gx = np.gradient(z)
    sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    return {"relief_m": float(np.percentile(z, 99) - np.percentile(z, 1)), "slope_mean_deg": float(sl.mean()),
            "slope_p95_deg": float(np.percentile(sl, 95))}


# ---------------------------------------------------------------- 3, 4 tunnels and facade-side corridors
def part34():
    g = _ctx(); seg, L = g["seg"], g["L"]
    with rd("bld_h") as s:
        TF = s.transform; H, W = s.shape
    bmask = rd("bld_mask").read(1) > 0
    T = {t: rasterio.open(PROC / f"sunlit_{t.replace(':', '')}.tif").read(1) / 100.0 for t in PERIODS}
    out = {}
    # underpasses: vehicle underpasses removed in the reference network, pedestrian underpasses covered
    used = {t: np.zeros(len(seg), bool) for t in PERIODS}
    for t in PERIODS:
        for r in g["base"][t]:
            used[t][r["SP"]["edges"]] = True; used[t][r["B10"]["edges"]] = True
    cov = seg.covered.values
    ns = json.load(open(PROC / "network_stats.json"))
    out["tunnel"] = {"covered_n": int(cov.sum()), "covered_m": float(L[cov].sum()),
                     "covered_len_share": float(L[cov].sum() / L.sum()),
                     "covered_used_by_routes": {t: int((cov & used[t]).sum()) for t in PERIODS},
                     "vehicle_removed_km": ns["vehicle_underpass_km_removed"],
                     "vehicle_removed_share": ns["vehicle_underpass_km_removed"] / ns["network_km_with_underpasses"]}
    steps = seg.highway.str.contains("steps")
    out["steps_len_share"] = float(L[steps].sum() / L.sum())
    S0 = {t: seg[f"s_{t.replace(':', '')}"].values for t in PERIODS}
    T_open = {}
    if cov.any():
        Eo = corridor_points(seg.geometry.values, bmask, TF)
        for t in PERIODS:
            T_open[t] = np.where(cov, edge_mean(Eo[0], T[t][Eo[1], Eo[2]], len(seg)), S0[t])
        out["tunnel_run"] = eval_routes(T_open, tag="underpasses_open")

    # facade-side corridors
    cls = np.select([seg.highway.str.contains("primary|trunk|secondary"),
                     seg.highway.str.contains("tertiary|residential|unclassified|living|service"),
                     seg.highway.str.contains("footway|pedestrian|path|steps|cycleway|track")],
                    ["arterial", "local", "pedestrian"], "other")
    sideS = {t: np.full((len(seg), 2), np.nan) for t in PERIODS}
    for i, ls in enumerate(seg.geometry.values):
        Lg = ls.length
        dd = np.arange(1.0, Lg, 2.0) if Lg > 2 else np.array([Lg / 2])
        pts = np.array([ls.interpolate(x).coords[0] for x in dd])
        p2 = np.array([ls.interpolate(min(Lg, x + 0.5)).coords[0] for x in dd])
        tv = p2 - pts; tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
        nrm = np.c_[-tv[:, 1], tv[:, 0]]
        for sgn, side in ((1, 0), (-1, 1)):
            offs = np.arange(0.5, 30.01, 0.5)
            q = pts[:, None, :] + sgn * offs[None, :, None] * nrm[:, None, :]
            col = ((q[..., 0] - TF.c) / TF.a).astype(int); row = ((q[..., 1] - TF.f) / TF.e).astype(int)
            ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
            hit = np.zeros(ok.shape, bool); hit[ok] = bmask[row[ok], col[ok]]
            first = np.where(hit.any(1), hit.argmax(1), -1)
            vals = {t: [] for t in PERIODS}
            for j, f in enumerate(first):
                if f < 0:
                    continue
                dfac = offs[f]
                for back in (1.0, 2.0, 3.0):
                    o = dfac - back
                    if o < 0:
                        continue
                    qq = pts[j] + sgn * o * nrm[j]
                    c_, r_ = int((qq[0] - TF.c) / TF.a), int((qq[1] - TF.f) / TF.e)
                    if 0 <= r_ < H and 0 <= c_ < W and not bmask[r_, c_]:
                        for t in PERIODS:
                            vals[t].append(T[t][r_, c_])
            if len(vals[PERIODS[0]]) >= 3 * max(1, 0.5 * len(dd)):
                for t in PERIODS:
                    sideS[t][i, side] = np.mean(vals[t])
    fac = {}
    for t in PERIODS:
        sl, sr = sideS[t][:, 0], sideS[t][:, 1]
        both = ~np.isnan(sl) & ~np.isnan(sr)
        anys = ~np.isnan(sl) | ~np.isnan(sr)
        r = {}
        for c in ["arterial", "local", "pedestrian"]:
            m = (cls == c)
            mb = m & both
            side_mean = np.nanmean(np.c_[sl, sr], axis=1)
            r[c] = {"len_share_class": float(L[m].sum() / L.sum()),
                    "both_sides_share": float(L[mb].sum() / max(L[m].sum(), 1)),
                    "s_center": float(np.average(S0[t][mb], weights=L[mb])) if mb.any() else None,
                    "s_sides_mean": float(np.average(side_mean[mb], weights=L[mb])) if mb.any() else None,
                    "s_shadier_side": float(np.average(np.minimum(sl, sr)[mb], weights=L[mb])) if mb.any() else None,
                    "mad_center_vs_side": float(np.average(np.abs(side_mean[mb] - S0[t][mb]), weights=L[mb])) if mb.any() else None,
                    "sides_differ_gt05": float(L[mb & (np.abs(sl - sr) > 0.5)].sum() / max(L[mb].sum(), 1))}
        fac[t] = r
    out["facade"] = fac
    # fallback: centreline corridor where no facade is found on either side; one-sided value where only one side
    # has a facade; covered segments keep 0
    best = {t: np.where(np.isnan(sideS[t]).all(1) | cov, S0[t], np.nanmin(sideS[t], axis=1)) for t in PERIODS}
    mean = {t: np.where(np.isnan(sideS[t]).all(1) | cov, S0[t], np.nanmean(sideS[t], axis=1)) for t in PERIODS}
    nb = ~np.isnan(sideS[PERIODS[0]])
    out["facade_coverage"] = {c: {"both": float(L[(cls == c) & nb.all(1)].sum() / max(L[cls == c].sum(), 1)),
                                  "one": float(L[(cls == c) & (nb.sum(1) == 1)].sum() / max(L[cls == c].sum(), 1)),
                                  "none": float(L[(cls == c) & ~nb.any(1)].sum() / max(L[cls == c].sum(), 1))}
                              for c in ["arterial", "local", "pedestrian", "other"]}
    out["facade_coverage"]["all"] = {"both": float(L[nb.all(1)].sum() / L.sum()),
                                     "one": float(L[nb.sum(1) == 1].sum() / L.sum()),
                                     "none": float(L[~nb.any(1)].sum() / L.sum())}
    out["facade_run_shadier"] = eval_routes(best, tag="facade_shadier")
    out["facade_run_mean"] = eval_routes(mean, tag="facade_mean")
    # reference routes evaluated in the facade-side exposure field
    ev = {}
    for t in PERIODS:
        red = []
        for r in g["base"][t]:
            sp, b = r["SP"]["edges"], r["B10"]["edges"]
            dsp, db = (L[sp] * best[t][sp]).sum(), (L[b] * best[t][b]).sum()
            red.append(100 * (1 - db / dsp) if dsp > 0 else 0)
        ev[t] = float(np.median(red))
    out["reference_B10_in_shadier_field"] = ev
    np.save(PROC / "facade_side_sunlit.npy", np.stack([sideS[t] for t in PERIODS]))
    return out


# ---------------------------------------------------------------- 6 network extent
def part6():
    from network import build_segments
    from scipy.spatial import cKDTree
    g = _ctx(); seg0 = g["seg"]
    aoi = gpd.read_file(PROC / "aoi.gpkg").geometry.iloc[0]
    seg, pos = build_segments(RAW / "walk_network_ext300.graphml", aoi.buffer(300, join_style=2))
    with rd("bld_h") as s:
        TF = s.transform; H, W = s.shape
    bmask = rd("bld_mask").read(1) > 0
    n = len(seg)
    Ec, Rc, Cc = corridor_points(seg.geometry.values, bmask, TF)
    Eb, Rb, Cb = sample_points(seg.geometry.values, 2.0, np.arange(-10, 10.1, 2.0), TF, (H, W))
    nd = edge_mean(Eb, rd("ndvi").read(1)[Rb, Cb], n); ls = edge_mean(Eb, rd("lst").read(1)[Rb, Cb], n)
    a, b = np.nanpercentile(seg0.ndvi, [2, 98]); a2, b2 = np.nanpercentile(seg0.lst, [2, 98])
    ndn = np.clip((nd - a) / (b - a), 0, 1); lsn = np.clip((ls - a2) / (b2 - a2), 0, 1)
    ndn = np.nan_to_num(ndn, nan=np.nanmedian(ndn)); lsn = np.nan_to_num(lsn, nan=np.nanmedian(lsn))
    L = seg.length.values
    net = Net(seg.u.values, seg.v.values, L)
    inside = seg.geometry.interpolate(0.5, normalized=True).within(aoi).values
    nodes0 = pd.read_csv(PROC / "network_nodes.csv").set_index("node")
    net0 = g["net"]
    xy0 = nodes0.loc[net0.nodes][["x", "y"]].values
    xy = np.array([pos[x] for x in net.nodes])
    dist, idx = cKDTree(xy).query(xy0)
    rows = []
    for t in PERIODS:
        tr = rasterio.open(PROC / f"sunlit_{t.replace(':', '')}.tif").read(1) / 100.0
        s = edge_sunlit(tr, (Ec, Rc, Cc), seg.covered.values)
        c2 = thermal_cost(L, s, ndn, lsn, WEIGHTS["balanced"])
        for k, (o, d, l0) in enumerate(g["od"]):
            if dist[o] > 1.0 or dist[d] > 1.0:
                continue
            sols, L0 = pareto_paths(net, int(idx[o]), int(idx[d]), L, c2, detour=1.5)
            sp, bb = sols[0][2], sols[budget(sols, L0, 0.10)][2]
            dsp, db = (L[sp] * s[sp]).sum(), (L[bb] * s[bb]).sum()
            r0 = g["base"][t][k]
            rows.append((t, k, L0 / l0 - 1, float((~inside[sp]).any()), float((~inside[bb]).any()),
                         100 * (1 - db / dsp) if dsp > 0 else 0, (100 * (1 - r0["B10"]["dsun"] / r0["SP"]["dsun"]) if r0["SP"]["dsun"] > 0 else 0.0)))
    df = pd.DataFrame(rows, columns=["time", "od", "dSPlen", "sp_out", "b10_out", "dDsun_ext", "dDsun_base"])
    return {"n_matched": int(df.od.nunique()), "n_segments_ext": int(n),
            "sp_shorter_share": float((df[df.time == "09:00"].dSPlen < -1e-6).mean()),
            "sp_len_change_med": float(100 * df[df.time == "09:00"].dSPlen.median()),
            "sp_len_change_min": float(100 * df[df.time == "09:00"].dSPlen.min()),
            "sp_uses_margin_share": float(df[df.time == "09:00"].sp_out.mean()),
            "by_time": {t: {"b10_uses_margin_share": float(d.b10_out.mean()),
                            "dDsun_ext_med": float(d.dDsun_ext.median()), "dDsun_base_med": float(d.dDsun_base.median())}
                        for t, d in df.groupby("time")}}


VARIANTS = [("trunk 0.10", {"trunk": 0.10}), ("trunk 0.40", {"trunk": 0.40}),
            ("tau_c 0.00", {"tau_c": 0.0}), ("tau_c 0.10", {"tau_c": 0.10}), ("tau_c 0.20", {"tau_c": 0.20}),
            ("canopy h >= 2.0 m", {"hmin": 2.0}), ("canopy h >= 3.0 m", {"hmin": 3.0}),
            ("no NDVI filter", {"ndvi_min": -1.0}), ("NDVI >= 0.25", {"ndvi_min": 0.25}),
            ("buildings only", {"no_canopy": True}), ("terrain", {"terrain": True})]

if __name__ == "__main__":
    parts = sys.argv[1:] or ["1", "2", "34", "5", "6"]
    f = PROC / "shade_checks.json"
    out = json.load(open(f)) if f.exists() else {}
    if "1" in parts:
        out["analytic"] = part1(); print(json.dumps(out["analytic"], indent=1), flush=True)
    if "2" in parts:
        out["margin"] = part2(); out["terrain_stats"] = terrain_stats()
        print(json.dumps(out["terrain_stats"]), flush=True)
    if "34" in parts:
        out.update(part34())
    if "6" in parts:
        out["extent"] = part6(); print(json.dumps(out["extent"], indent=1), flush=True)
    json.dump(out, open(f, "w"), indent=1)
    if "5" in parts:
        with Pool(6) as pool:
            res = dict(pool.map(_variant, VARIANTS, chunksize=1))
        out["variants"] = res
        json.dump(out, open(f, "w"), indent=1)
    print("shade checks saved", list(out))

"""Publication figures (300 dpi JPG for submission, PDF for vector proofs).
Common visual language: route classes keep one colour and line style in every figure; maps share extent,
scale bar and north arrow; hours are local time (UTC+3) and solar elevation is apparent elevation."""
import json
import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from config import *
from impedance import load_segments

mpl.rcParams.update({"font.family": "Cambria", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 400,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.unicode_minus": True})
INK, INK2, GRID, NODATA = "#0b0b0b", "#52514e", "#dcdbd6", "#f2f1ec"
C_ROUTE = {"SP": "#3d3c39", "B5": "#c98a00", "B10": "#e0602a", "KP": "#2a6fc4", "MTEP": "#1b9a6c"}
LS_ROUTE = {"SP": "-", "B5": (0, (4, 1.5)), "B10": "-", "KP": (0, (5, 2)), "MTEP": (0, (1.5, 1.5))}
C_PER = dict(zip(PERIODS, ["#2a6fc4", "#c98a00", "#e0602a", "#7b3fa0"]))
M_PER = dict(zip(PERIODS, ["o", "s", "^", "D"]))
C_STR = {"TB": "#2a6fc4", "gain": "#7b3fa0", "flow": "#e0602a", "sun": "#1b9a6c", "random100": "#8c8b86"}
LBL_STR = {"TB": "Thermal betweenness", "gain": "Direct gain ranking", "flow": "Pedestrian flow",
           "sun": "Daily sunlit fraction", "random100": "Random (mean, 95% range)"}

seg = load_segments()
L = seg.length.values
aoi = gpd.read_file(PROC / "aoi.gpkg")
X0, Y0, X1, Y1 = aoi.total_bounds
summ = json.load(open(ROOT / "outputs" / "results_summary.json"))
BLD = gpd.read_file(PROC / "buildings_lod1.gpkg")


LANG_TR = os.environ.get("FIG_LANG") == "tr"
if LANG_TR:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manuscript"))
    from fig_tr import translate_figure


def save(fig, name):
    out = FIG
    if LANG_TR:
        out = FIG.parent / "figures_tr"; out.mkdir(exist_ok=True)
        translate_figure(fig)
    if os.environ.get("FIG_DUMP"):
        fig.canvas.draw()
        with open(FIG.parent / "figure_texts.txt", "a", encoding="utf8") as fh:
            for t in fig.findobj(mpl.text.Text):
                if t.get_text().strip():
                    fh.write(name + chr(9) + repr(t.get_text()) + chr(10))
    fig.savefig(out / f"{name}.jpg", dpi=400, bbox_inches="tight", pil_kwargs={"quality": 95})
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def mapdecor(ax, x0, y0, x1, y1, bar=500, north=True):
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color(INK2)
    w = x1 - x0
    bx, by = x0 + 0.05 * w, y0 + 0.05 * (y1 - y0)
    ax.add_patch(Rectangle((bx, by), bar, 0.012 * w, color=INK, lw=0, zorder=10))
    ax.text(bx + bar / 2, by + 0.025 * w, f"{bar} m" if bar < 1000 else f"{bar / 1000:g} km", ha="center",
            va="bottom", fontsize=7, color=INK, zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    if north:
        an = ax.annotate("N", xy=(x1 - 0.06 * w, y1 - 0.04 * w), xytext=(x1 - 0.06 * w, y1 - 0.13 * w), ha="center",
                         va="center", fontsize=8, color=INK, zorder=10,
                         arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
        an.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def read(name):
    with rasterio.open(PROC / name) as s:
        a = s.read(1).astype(float); tf = s.transform
    return a, (tf.c, tf.c + tf.a * a.shape[1], tf.f + tf.e * a.shape[0], tf.f)


def panel(ax, letter, x=0.0):
    ax.text(x, 1.02, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def plot_route(ax, edges, color, lw, ls):
    from matplotlib.collections import LineCollection
    segs = [np.asarray(g.coords) for g in seg.geometry.values[edges]]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=[ls], capstyle="round", zorder=6))


def route_handles(types):
    return [Line2D([], [], color=C_ROUTE[t], ls=LS_ROUTE[t], lw=1.6, label=t) for t in types]


# ---------------------------------------------------------------- Figure 1: workflow
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.axis("off"); ax.set_xlim(-1.5, 101.5); ax.set_ylim(0, 50)

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0", fc=fc, ec=INK2, lw=0.6))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=7.2, color=INK2, linespacing=1.4)

    def arr(x0, y0, x1, y1, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, linestyle=ls))
    box(0, 27, 22, 20, "1 Open inputs", "OSM network, footprints\n3D-GloBFP heights\ncanopy height map\nSentinel-2, Landsat", "#f4f3ef")
    box(26, 27, 22, 20, "2 Surface and shade", "1 m building and\ncanopy surface\nhourly shadow casting\n07:00 to 19:00", "#eef3fb")
    box(52, 27, 22, 20, "3 Routing", "edge sunlit fraction\ncomposite cost\nPareto set, ε ≤ 50%\n500 evaluation pairs", "#eef3fb")
    box(78, 27, 22, 20, "4 Prioritisation", "flow of 5000 pairs\nthermal betweenness\ntreated length 1–10%\nscenario comparison", "#fdf0ea")
    for x in (22.7, 48.7, 74.7):
        arr(x, 37, x + 2.6, 37)
    box(13, 1, 34, 18, "Verification and validation", "analytic shadow test, exhaustive\npath enumeration, heights against\nOSM tags, network extent test", "#f4f3ef")
    box(53, 1, 34, 18, "Uncertainty and sensitivity", "height Monte Carlo (three error\nmodels), weights, penalty, canopy,\nterrain, corridor, demand", "#f4f3ef")
    for x in (30, 37):
        arr(x, 19.6, x, 26.3, "--")
    for x0, x1 in ((63, 63), (70, 70), (80, 89)):
        arr(x0, 19.6, x1, 26.3, "--")
    save(fig, "Figure1")


# ---------------------------------------------------------------- Figure 2: study area and data
def fig2():
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ne = gpd.read_file(RAW / "ne_admin0.gpkg")
    ne.plot(ax=ax, color="#e9e7e1", edgecolor="white", lw=0.4)
    ne[ne.ISO_A3 == "TUR"].plot(ax=ax, color="#c9c6bd", edgecolor="white", lw=0.4)
    ax.plot(CENTER_LON, CENTER_LAT, marker="s", ms=5, color="#e0602a", mec=INK, mew=0.5)
    ax.text(CENTER_LON + 0.5, CENTER_LAT + 0.4, "Kayseri", fontsize=8, color=INK)
    ax.set_xlim(25.5, 45); ax.set_ylim(35.5, 42.5); ax.set_aspect(1 / np.cos(np.deg2rad(39)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color(INK2)
    ax.text(0.03, 0.04, "Türkiye", transform=ax.transAxes, fontsize=7.5, color=INK2)
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    src_c = {"globfp": "#9bb7d4", "globfp_footprint": "#2a6fc4", "imputed": "#e0602a", "osm": "#1b9a6c"}
    src_l = {"globfp": "OSM footprint, 3D-GloBFP height", "globfp_footprint": "Added 3D-GloBFP footprint",
             "imputed": "Local median", "osm": "OSM height tag"}
    for k, c in src_c.items():
        BLD[BLD.h_src == k].plot(ax=ax, color=c, lw=0)
    aoi.boundary.plot(ax=ax, color=INK, lw=0.6)
    mapdecor(ax, X0 - 150, Y0 - 150, X1 + 150, Y1 + 150)
    ax.legend(handles=[Patch(color=c, label=src_l[k]) for k, c in src_c.items()], loc="upper center",
              bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=6.5)
    panel(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    BLD.plot(ax=ax, column="h", cmap="Blues", vmin=0, vmax=40, lw=0, legend=True,
             legend_kwds={"label": "Building height (m)", "shrink": 0.75, "extend": "max"})
    aoi.boundary.plot(ax=ax, color=INK, lw=0.6)
    mapdecor(ax, X0 - 150, Y0 - 150, X1 + 150, Y1 + 150); panel(ax, "c")
    for i, (name, cmap, lab, vr, ext) in enumerate([("canopy_h_1m.tif", "Greens", "Canopy height (m)", (0, 15), "max"),
                                                     ("ndvi_1m.tif", "Greens", "NDVI", (0, 0.6), "both"),
                                                     ("lst_1m.tif", "YlOrRd", "Surface temperature (°C)", (40, 50), "both")]):
        ax = fig.add_subplot(gs[1, i])
        a, e = read(name)
        if name.startswith("canopy"):
            a[a <= 0] = np.nan
            ax.set_facecolor(NODATA)
        im = ax.imshow(a, extent=e, cmap=cmap, vmin=vr[0], vmax=vr[1], interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, shrink=0.75, extend=ext); cb.set_label(lab)
        if name.startswith("canopy"):
            seg.plot(ax=ax, color=INK2, lw=0.25)
        aoi.boundary.plot(ax=ax, color=INK, lw=0.6)
        mapdecor(ax, X0 - 150, Y0 - 150, X1 + 150, Y1 + 150); panel(ax, "def"[i])
    save(fig, "Figure2")


# ---------------------------------------------------------------- Figure 3: shade on the network
def fig3():
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.22, wspace=0.08)
    cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
    w = 450
    sub = seg.cx[cx - w:cx + w, cy - w:cy + w]
    bld = BLD.cx[cx - w:cx + w, cy - w:cy + w]
    for k, t in enumerate(PERIODS):
        ax = fig.add_subplot(gs[0, k])
        a, ext = read(f"sunlit_{t.replace(':', '')}.tif")
        ax.imshow(a / 100, extent=ext, cmap="Greys_r", vmin=0, vmax=1.6, interpolation="nearest")
        bld.plot(ax=ax, color="#b9b7b0", lw=0)
        sub.plot(ax=ax, column=f"s_{t.replace(':', '')}", cmap="YlOrRd", vmin=0, vmax=1, lw=1.3)
        mapdecor(ax, cx - w, cy - w, cx + w, cy + w, bar=200, north=(k == 0))
        sn = summ["network_stats"]["sun"][t]
        az = np.deg2rad(sn["az"]); ox_, oy_ = cx - w + 120, cy + w - 120
        ax.add_patch(plt.Circle((ox_, oy_), 95, color="white", alpha=0.85, zorder=11))
        ax.annotate("", xy=(ox_ + 80 * np.sin(az), oy_ + 80 * np.cos(az)),
                    xytext=(ox_ - 80 * np.sin(az), oy_ - 80 * np.cos(az)),
                    arrowprops=dict(arrowstyle="-|>", color="#c98a00", lw=1.8, mutation_scale=12), zorder=12)
        ax.set_title(f"({'abcd'[k]}) {t}, α = {sn['elev']:.0f}°", fontsize=8, loc="left")
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.92, 0.58, 0.012, 0.28])
    fig.colorbar(sm, cax=cax, label="Edge sunlit fraction")
    rq1 = pd.read_csv(TAB / "network_sunlit_by_hour.csv")
    rq1 = rq1[rq1.time.isin(HOURLY)]
    hrs = [int(x[:2]) for x in rq1.time]
    ax = fig.add_subplot(gs[1, :2])
    osh = summ["orientation_len_share"]
    oc = dict(zip(["N-S", "NE-SW", "E-W", "NW-SE"], ["#2a6fc4", "#c98a00", "#e0602a", "#1b9a6c"]))
    om = dict(zip(oc, ["o", "s", "^", "D"]))
    for o, c in oc.items():
        ax.plot(hrs, rq1[o], color=c, lw=1.4, marker=om[o], ms=3.2, label=f"{o.replace('-', '–')} ({100 * osh[o]:.0f}%)")
    ax.plot(hrs, rq1["all"], color=INK, lw=1.4, ls="--", label="All")
    for t in PERIODS:
        ax.axvline(int(t[:2]) + int(t[3:]) / 60, color=GRID, lw=0.7, zorder=0)
    ax.set_xlabel("Local time (h, UTC+3)"); ax.set_ylabel("Length-weighted sunlit fraction")
    ax.set_ylim(0, 1); ax.set_xticks(range(7, 20, 2)); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="lower center", title="Orientation (length share)", title_fontsize=7.5)
    panel(ax, "e")
    ax = fig.add_subplot(gs[1, 2:], sharey=ax)
    labs = ["<10", "10–15", "15–20", "20–30", ">30"]
    seg["hcls"] = pd.cut(seg.h_adj, [0, 10, 15, 20, 30, 200], labels=labs)
    g0 = seg.dropna(subset=["hcls"]).groupby("hcls", observed=True)
    km = g0.length.sum() / 1000
    for t in PERIODS:
        v = g0.apply(lambda d: np.average(d[f"s_{t.replace(':', '')}"], weights=d.length), include_groups=False)
        ax.plot(range(len(v)), v.values, color=C_PER[t], lw=1.4, marker=M_PER[t], ms=3.2, label=t)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{l}\n{km[l]:.0f} km" for l in labs], fontsize=7.5)
    ax.set_xlabel("Mean adjacent building height (m)"); plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.5); ax.legend(frameon=False, ncol=2, loc="lower left")
    panel(ax, "f")
    save(fig, "Figure3")


# ---------------------------------------------------------------- Figure 4: representative routes
def fig4():
    R = pickle.load(open(PROC / "routing_results.pkl", "rb"))["res"]
    ex = summ["examples3"]
    t = "15:00"; s = seg["s_1500"].values
    a, ext = read("sunlit_1500.tif")
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.85], hspace=0.38, wspace=0.12)
    types = ["SP", "KP", "B10"]
    lw = {"SP": 3.2, "KP": 2.2, "B10": 1.5}
    for i, (q, lab) in enumerate([("p10", "Low gain"), ("p50", "Median gain"), ("p90", "High gain")]):
        k = ex[q]["index"]
        ax = fig.add_subplot(gs[0, i])
        es_all = set().union(*[set(R[t][k][x]["edges"]) for x in types])
        bb = seg.iloc[list(es_all)].total_bounds; pad = 230
        side = max(bb[2] - bb[0], bb[3] - bb[1]) / 2 + pad
        mx, my = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        ax.imshow(a / 100, extent=ext, cmap="Greys_r", vmin=0, vmax=1.6, interpolation="nearest")
        BLD.cx[mx - side:mx + side, my - side:my + side].plot(ax=ax, color="#b9b7b0", lw=0)
        for x in types:
            plot_route(ax, R[t][k][x]["edges"], C_ROUTE[x], lw[x], LS_ROUTE[x])
        sp = R[t][k]["SP"]["edges"]
        g0, g1 = seg.iloc[sp[0]].geometry, seg.iloc[sp[-1]].geometry
        p0 = g0.coords[0] if len(sp) < 2 or g0.coords[0] not in seg.iloc[sp[1]].geometry.coords else g0.coords[-1]
        p1 = g1.coords[-1] if len(sp) < 2 or g1.coords[-1] not in seg.iloc[sp[-2]].geometry.coords else g1.coords[0]
        ax.plot(*p0, marker="o", ms=6, color="white", mec=INK, zorder=8)
        ax.plot(*p1, marker="s", ms=6, color=INK, zorder=8)
        mapdecor(ax, mx - side, my - side, mx + side, my + side, bar=200, north=(i == 0))
        r = R[t][k]
        txt = "\n".join(f"{x}: {r[x]['len']:,.0f} m, +{100 * (r[x]['len'] / r['SP']['len'] - 1):.1f}%, "
                        f"{r[x]['dsun']:,.1f} m sunlit" for x in types)
        ax.text(0.5, -0.03, txt, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color=INK2, linespacing=1.3)
        ax.set_title(f"({'abc'[i]}) {lab}", fontsize=8, loc="left")
    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[1, 2.1], wspace=0.5)
    ax = fig.add_subplot(gsb[0, 0])
    k = ex["p50"]["index"]
    f = np.array(R[t][k]["front"])
    g = (f - f.min(0)) / np.where(np.ptp(f, 0) > 0, np.ptp(f, 0), 1)
    ax.plot(g[:, 0], g[:, 1], color=INK2, marker="o", ms=3.5, lw=0.8)
    ax.plot([g[0, 0], g[-1, 0]], [g[0, 1], g[-1, 1]], color=INK2, lw=0.8, ls="--")
    kp = int(np.argmin(np.abs(f[:, 0] - R[t][k]["KP"]["len"])))
    dv = g[-1] - g[0]; u = dv / np.linalg.norm(dv)
    foot = g[0] + np.dot(g[kp] - g[0], u) * u
    ax.plot([g[kp, 0], foot[0]], [g[kp, 1], foot[1]], color=C_ROUTE["KP"], lw=1.2)
    ax.plot(*g[kp], marker="o", ms=8, mfc="none", mec=C_ROUTE["KP"], mew=1.4)
    ax.text(g[kp, 0] + 0.05, g[kp, 1] + 0.03, "KP", color=C_ROUTE["KP"], fontsize=8)
    ax.text(g[0, 0] + 0.04, g[0, 1] - 0.02, "SP", color=C_ROUTE["SP"], fontsize=8)
    ax.text(g[-1, 0] - 0.02, g[-1, 1] + 0.06, "MTEP", color=C_ROUTE["MTEP"], fontsize=8, ha="right")
    ax.set_xlabel("Normalised length"); ax.set_ylabel("Normalised composite cost")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.1); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.4)
    panel(ax, "d")
    ax = fig.add_subplot(gsb[0, 1])
    y = 0; ticks, tl = [], []
    cmap = plt.cm.YlOrRd
    xmax = 0
    for q, lab in [("p90", "High"), ("p50", "Median"), ("p10", "Low")]:
        k = ex[q]["index"]
        for x in ["B10", "SP"]:
            es = R[t][k][x]["edges"]; x0 = 0
            for e in es:
                ax.add_patch(Rectangle((x0, y), L[e], 0.8, color=cmap(s[e]), lw=0)); x0 += L[e]
            xmax = max(xmax, x0)
            ticks.append(y + 0.4); tl.append(f"{lab}, {x}"); y += 1
        y += 0.5
    ax.set_yticks(ticks); ax.set_yticklabels(tl, fontsize=7.5); ax.set_ylim(-0.2, y)
    ax.set_xlim(0, xmax * 1.02); ax.set_xlabel("Distance along route (m)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.01); cb.set_label("Sunlit fraction")
    panel(ax, "e")
    fig.legend(handles=route_handles(types), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    save(fig, "Figure4")


# ---------------------------------------------------------------- Figure 5: 500 OD distributions
def fig5():
    cur = pd.read_csv(TAB / "budget_curve_dsun.csv")
    rt = pd.read_csv(TAB / "route_metrics_long.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.35, 1]})
    ax = axs[0]
    for t in PERIODS:
        d = cur[cur.time == t]
        ax.fill_between(100 * d.eps, d.dDsun_p25, d.dDsun_p75, color=C_PER[t], alpha=0.12, lw=0)
        ax.plot(100 * d.eps, d.dDsun_med, color=C_PER[t], lw=1.4, marker=M_PER[t], ms=3, label=t)
    ax.set_xlabel("Detour budget ε (%)"); ax.set_ylabel("ΔDsun (%)")
    ax.grid(color=GRID, lw=0.4); ax.legend(frameon=False, title="Period", title_fontsize=7.5); panel(ax, "a")
    ax = axs[1]
    types = ["B5", "B10", "KP", "MTEP"]
    pos, data, cols = [], [], []
    for i, t in enumerate(PERIODS):
        for j, typ in enumerate(types):
            pos.append(i * 5.2 + j); data.append(rt[(rt.time == t) & (rt.type == typ)].dDsun.values); cols.append(C_ROUTE[typ])
    bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=False, whis=1.5,
                    medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=INK2, lw=0.7),
                    capprops=dict(color=INK2, lw=0.7))
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c); b.set_edgecolor("white"); b.set_linewidth(0.8)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([i * 5.2 + 1.5 for i in range(4)]); ax.set_xticklabels(PERIODS)
    ax.set_ylabel("ΔDsun (%)"); ax.grid(axis="y", color=GRID, lw=0.4)
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=C_ROUTE[t]) for t in types], types, frameon=False, ncol=2,
              loc="upper center", fontsize=7, columnspacing=1.0, handlelength=1.2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    panel(ax, "b")
    ax = axs[2]
    for t in PERIODS:
        x = np.sort(rt[(rt.time == t) & (rt.type == "B10")].dDsun.values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=C_PER[t], lw=1.3, label=t)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("ΔDsun of B10 (%)"); ax.set_ylabel("Cumulative share of pairs")
    ax.set_ylim(0, 1); ax.grid(color=GRID, lw=0.4); panel(ax, "c")
    plt.tight_layout(); save(fig, "Figure5")


# ---------------------------------------------------------------- Figure 6: satellite-only comparison
def fig6():
    R = pickle.load(open(PROC / "routing_results.pkl", "rb"))
    res, sat = R["res"], R["sat"]
    full, sv = {}, {}
    for t in PERIODS:
        S_ = seg[f"s_{t.replace(':', '')}"].values
        full[t] = np.array([100 * (1 - r["B10"]["dsun"] / r["SP"]["dsun"]) for r in res[t]])
        sv[t] = np.array([100 * (1 - (L[sat[k]["B10"]] * S_[sat[k]["B10"]]).sum() / r["SP"]["dsun"])
                          for k, r in enumerate(res[t])])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axs[0]
    bp = ax.boxplot([full[t] - sv[t] for t in PERIODS], positions=range(4), widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=INK2, lw=0.7),
                    capprops=dict(color=INK2, lw=0.7))
    for b, t in zip(bp["boxes"], PERIODS):
        b.set_facecolor(C_PER[t]); b.set_alpha(0.8); b.set_edgecolor("white")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(PERIODS)
    ax.set_ylabel("Paired difference in ΔDsun (pp)"); ax.grid(axis="y", color=GRID, lw=0.4); panel(ax, "a")
    ax = axs[1]
    t = "15:00"
    ax.scatter(sv[t], full[t], s=7, color=C_PER[t], alpha=0.5, lw=0)
    lo, hi = -60, 70
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    out_ = np.where((sv[t] < lo) | (full[t] < lo) | (sv[t] > hi) | (full[t] > hi))[0]
    ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
    lab_ = []
    for i_ in out_:
        x_, y_ = np.clip(sv[t][i_], lo + 2, hi - 2), np.clip(full[t][i_], lo + 2, hi - 2)
        mk = "<" if sv[t][i_] < lo else (">" if sv[t][i_] > hi else ("v" if full[t][i_] < lo else "^"))
        ax.plot(x_, y_, marker=mk, ms=4.5, color=C_PER[t], mec=INK, mew=0.4, ls="")
        lab_.append(f"({sv[t][i_]:.0f}, {full[t][i_]:.0f})")
    if lab_:
        ax.text(0.97, 0.03, "Off axis (x, y):\n" + "\n".join(lab_), transform=ax.transAxes, fontsize=6, color=INK2,
                ha="right", va="bottom", linespacing=1.15)
    ax.set_xlabel("Satellite-only B10, ΔDsun (%)"); ax.set_ylabel("Shade model B10, ΔDsun (%)")
    ax.set_title("15:00", fontsize=8, loc="right"); panel(ax, "b")
    ax = axs[2]
    x = np.arange(4)
    ax.bar(x - 0.2, [100 * np.mean(sv[t] < -1e-9) for t in PERIODS], 0.38, color="#8c8b86", label="Satellite-only B10")
    ax.bar(x + 0.2, [100 * np.mean(full[t] < -1e-9) for t in PERIODS], 0.38, color=C_ROUTE["B10"], label="Shade model B10")
    ax.set_xticks(x); ax.set_xticklabels(PERIODS); ax.set_ylabel("Pairs sunnier than SP (%)"); ax.set_ylim(0, 68)
    ax.grid(axis="y", color=GRID, lw=0.4); ax.legend(frameon=False, loc="upper left", fontsize=7); panel(ax, "c")
    plt.tight_layout(); save(fig, "Figure6")


# ---------------------------------------------------------------- Figure 7: heights and uncertainty
def fig7():
    v = pd.read_csv(PROC / "height_validation_detail.csv")
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=1.0)
    ax = fig.add_subplot(gs[0, :2])
    for tag, mk in [("levels", "o"), ("height", "^")]:
        for mg, fc in [(False, "white"), (True, "#2a6fc4")]:
            d = v[(v.tag == tag) & (v.merged == mg)]
            ax.scatter(d.h_osm, d.h_glo, s=16, marker=mk, facecolor=fc, edgecolor="#2a6fc4", lw=0.7)
    m = max(v.h_osm.max(), v.h_glo.max()) + 5
    ax.plot([0, m], [0, m], color=INK2, lw=0.7, ls="--")
    ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_aspect("equal")
    ax.set_xlabel("OSM-tagged height (m)"); ax.set_ylabel("3D-GloBFP height (m)"); ax.grid(color=GRID, lw=0.4)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#2a6fc4", label="Storeys tag"),
                       Line2D([], [], marker="^", ls="", mfc="white", mec="#2a6fc4", label="Height tag"),
                       Line2D([], [], marker="o", ls="", mfc="#2a6fc4", mec="#2a6fc4", label="Shared source polygon")],
              frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.17, 0.91), borderpad=0.3, handletextpad=0.3)
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 2:4])
    ax.scatter(v.h_osm, v.h_glo - v.h_osm, s=14, color="#2a6fc4", alpha=0.8, lw=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("OSM-tagged height (m)"); ax.set_ylabel("Height error (m)"); ax.grid(color=GRID, lw=0.4)
    panel(ax, "b")
    ax = fig.add_subplot(gs[0, 4:])
    osm = gpd.read_file(RAW / "osm_buildings.gpkg").to_crs(CRS_UTM)
    osm = osm[osm.geometry.area > 4].reset_index(drop=True)
    pts = osm.iloc[v.bid.values].geometry.representative_point()
    BLD.plot(ax=ax, color="#dcdbd6", lw=0)
    ax.scatter(pts.x, pts.y, s=10, color="#e0602a", edgecolor=INK, lw=0.3, zorder=5)
    aoi.boundary.plot(ax=ax, color=INK, lw=0.6)
    mapdecor(ax, X0 - MARGIN, Y0 - MARGIN, X1 + MARGIN, Y1 + MARGIN, bar=1000); panel(ax, "c")
    ax = fig.add_subplot(gs[1, :3])
    rq4 = summ["rq4"]; rq2 = summ["rq2"]
    sc = [("Independent", lambda t: rq4[t]["mc_B10_dDsun_med_range"], "#2a6fc4"),
          ("Source polygon", lambda t: rq4["dependent"]["grouped"][t]["red_range"], "#7b3fa0"),
          ("250 m block", lambda t: rq4["dependent"]["block"][t]["red_range"], "#1b9a6c")]
    _hd = pickle.load(open(PROC / "rev_height_mc.pkl", "rb"))
    _hdr = {t: [100 * np.median(x[t]["rows"][:, 4]) for x in _hd["mc"]] for t in PERIODS}
    sc.append(("Height dependent", lambda t: (min(_hdr[t]), max(_hdr[t])), "#c98a00"))
    for i, t in enumerate(PERIODS):
        ax.plot(i - 0.3, rq2[t]["B10"]["dDsun_med"], marker="o", color=INK, ms=5)
        for j, (lab, f, c) in enumerate(sc):
            lo, hi = f(t)
            ax.plot([i - 0.15 + 0.1 * j] * 2, [lo, hi], color=c, lw=2.2, solid_capstyle="butt")
        ax.plot(i + 0.3, rq4[t]["bias_B10_dDsun_med"], marker="x", color="#e0602a", ms=6, mew=1.5)
        ax.plot(i + 0.3, 100 * np.median(_hd["hd_bias"][t]["rows"][:, 4]), marker="+", color=INK, ms=7, mew=1.3)
    ax.set_xticks(range(4)); ax.set_xticklabels(PERIODS); ax.set_ylabel("Median ΔDsun of B10 (%)")
    ax.set_ylim(0, None); ax.grid(axis="y", color=GRID, lw=0.4)
    top = ax.get_ylim()[1]; ax.set_ylim(0, top * 1.35)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, label="Reference")] +
              [Line2D([], [], color=c, lw=2.2, label=f"{lab}, range of 30") for lab, _, c in sc] +
              [Line2D([], [], marker="x", ls="", color="#e0602a", mew=1.5, label="Uniform bias (stress test)"),
               Line2D([], [], marker="+", ls="", color=INK, mew=1.3, label="Class-specific bias")],
              frameon=False, fontsize=6.3, loc="upper center", ncol=2)
    panel(ax, "d")
    ax = fig.add_subplot(gs[1, 3:])
    Mi = pickle.load(open(PROC / "height_mc.pkl", "rb"))
    Md = pickle.load(open(PROC / "height_mc_dependent.pkl", "rb"))
    for lab, runs, c in [("Independent", Mi["mc"], "#2a6fc4"), ("Source polygon", Md["grouped"], "#7b3fa0"),
                         ("250 m block", Md["block"], "#1b9a6c"),
                         ("Height dependent", pickle.load(open(PROC / "rev_height_mc.pkl", "rb"))["mc"], "#c98a00")]:
        J = np.sort(np.concatenate([mm[t]["rows"][:, 0] for mm in runs for t in PERIODS]))
        ident = np.mean(J >= 0.999)
        ax.step(J, np.arange(1, len(J) + 1) / len(J), where="post", color=c, lw=1.3,
                label=f"{lab}, {100 * ident:.0f}% identical")
    ax.set_xlabel("Overlap with reference B10 (Jaccard)"); ax.set_ylabel("Cumulative share")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5); panel(ax, "e")
    save(fig, "Figure7")


# ---------------------------------------------------------------- Figure 8: shade priorities
def fig8():
    g = gpd.read_file(PROC / "thermal_betweenness.gpkg")
    ext = pd.read_csv(TAB / "intervention_extended.csv")
    con = pd.read_csv(TAB / "concentration_curve.csv")
    fig = plt.figure(figsize=(7.2, 7.0))
    axa = fig.add_axes([0.04, 0.44, 0.45, 0.5]); axb = fig.add_axes([0.53, 0.44, 0.45, 0.5])
    pal = np.array([["#e8e8e8", "#ace4e4", "#5ac8c8"], ["#dfb0d6", "#a5add3", "#5698b9"],
                    ["#be64ac", "#8c62aa", "#3b4994"]])
    nz = (g.flow > 0).values

    def wq(x, w, q):
        o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
        return x[o][np.searchsorted(c, q)]
    fq = [wq(g.flow.values[nz], g.length.values[nz], q) for q in (1 / 3, 2 / 3)]
    sq = [wq(g.s_day.values, g.length.values, q) for q in (1 / 3, 2 / 3)]
    fc = np.digitize(g.flow, fq); scl = np.digitize(g.s_day, sq)
    g[~nz].plot(ax=axa, color="#ecebe6", lw=0.3)
    for i in range(3):
        for j in range(3):
            d = g[nz & (fc == j) & (scl == i)]
            if len(d):
                d.plot(ax=axa, color=pal[i, j], lw=0.5 + 0.6 * j)
    mapdecor(axa, X0, Y0, X1, Y1); panel(axa, "a")
    axa.add_patch(Rectangle((0.015, 0.595), 0.305, 0.32, transform=axa.transAxes, facecolor="white", edgecolor=INK2,
                            lw=0.4, zorder=11))
    lg = axa.inset_axes([0.13, 0.69, 0.15, 0.15], zorder=12)
    for i in range(3):
        for j in range(3):
            lg.add_patch(Rectangle((j, i), 1, 1, color=pal[i, j]))
    lg.set_xlim(0, 3); lg.set_ylim(0, 3)
    lg.set_xticks([1, 2]); lg.set_xticklabels([f"{fq[0]:.0f}", f"{fq[1]:.0f}"], fontsize=6.5)
    lg.set_yticks([1, 2]); lg.set_yticklabels([f"{sq[0]:.2f}", f"{sq[1]:.2f}"], fontsize=6.5)
    lg.tick_params(length=2, pad=1)
    lg.set_xlabel("Flow (trips)", fontsize=7, labelpad=1); lg.set_ylabel("Sunlit fraction", fontsize=7, labelpad=2)
    o = np.argsort(-g.tb.values); cum = np.cumsum(g.length.values[o]) / g.length.sum()
    rank = np.empty(len(g)); rank[o] = cum
    g["cls"] = np.select([rank <= 0.01, rank <= 0.03, rank <= 0.05], [3, 2, 1], 0)
    g[g.cls == 0].plot(ax=axb, color="#d6d4ce", lw=0.35)
    for c, col, lw in [(1, "#f5c08a", 1.3), (2, "#e0602a", 1.7), (3, "#8f2a10", 2.1)]:
        g[g.cls == c].plot(ax=axb, color=col, lw=lw)
    axb.legend(handles=[Line2D([], [], color=c, lw=2, label=l) for c, l in
                        [("#8f2a10", "0–1%"), ("#e0602a", "1–3%"), ("#f5c08a", "3–5%")]],
               title="Rank by length", title_fontsize=7.5, frameon=True, framealpha=0.9, loc="lower right", fontsize=7.5)
    import osmnx as ox
    ed = ox.graph_to_gdfs(ox.load_graphml(RAW / "walk_network.graphml"), nodes=False).to_crs(CRS_UTM).reset_index()
    ed["name"] = ed["name"].apply(lambda x: x[0] if isinstance(x, list) else x)
    top = g[g.cls > 0].copy()
    top["name"] = ed["name"].values[ed.sindex.nearest(top.geometry.interpolate(0.5, normalized=True), return_all=False)[1]]
    km_ = top.groupby("name").geometry.apply(lambda x: x.length.sum()).sort_values(ascending=False)
    for k_, nm in enumerate(km_.index[:5]):
        gg = top[top.name == nm].geometry.union_all()
        c_ = gg.interpolate(0.5, normalized=True) if gg.geom_type == "LineString" else gg.representative_point()
        axb.annotate(nm.replace(" Caddesi", " Cd.").replace(" Bulvarı", " Bulv."), xy=(c_.x, c_.y), fontsize=6.3,
                     xytext=(6, 6 if k_ % 2 else -10), textcoords="offset points", color=INK,
                     path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], zorder=15)
    mapdecor(axb, X0, Y0, X1, Y1); panel(axb, "b")
    ax = fig.add_axes([0.07, 0.07, 0.25, 0.3])
    ax.plot(100 * con.len_share, 100 * con.tripkm_share, color=INK, lw=1.4)
    ax.plot([0, 100], [0, 100], color=GRID, lw=0.8, ls="--")
    fcn = summ["flow_concentration"]
    for p, key in [(1, "top1pct"), (5, "top5pct"), (10, "top10pct")]:
        y = 100 * fcn[f"{key}_len_share_of_tripkm"]
        ax.plot(p, y, marker="o", ms=4, color="#e0602a"); ax.text(p + 3, y - 5, f"{y:.0f}%", fontsize=7.5)
    ax.set_xlabel("Network length (%)"); ax.set_ylabel("Modelled trip distance (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(color=GRID, lw=0.4); panel(ax, "c")
    ax = fig.add_axes([0.41, 0.07, 0.25, 0.3])
    _r = pd.read_csv(TAB / "rev_random100.csv")
    r = _r.groupby("frac").agg(km=("km", "mean"), red_p2_5=("red_dsun_sp_day", lambda x: x.quantile(.025)),
                               red_p97_5=("red_dsun_sp_day", lambda x: x.quantile(.975))).reset_index()
    ext = pd.concat([ext[ext.strategy != "random100"],
                     _r.groupby("frac").agg(km=("km", "mean"), red_dsun_sp_day=("red_dsun_sp_day", "mean")).reset_index()
                     .assign(strategy="random100")])
    ax.fill_between(np.r_[0, r.km], np.r_[0, r.red_p2_5], np.r_[0, r.red_p97_5], color=C_STR["random100"], alpha=0.25, lw=0)
    for s in ["TB", "gain", "flow", "sun", "random100"]:
        d = ext[ext.strategy == s].sort_values("frac")
        ax.plot(np.r_[0, d.km], np.r_[0, d.red_dsun_sp_day], color=C_STR[s], lw=1.3, marker="o", ms=2.8,
                ls=":" if s == "gain" else "-", label=LBL_STR[s])
    ax.set_xlabel("Treated length (km)"); ax.set_ylabel("ΔDday on SP (%)"); ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.02, 0.42)); panel(ax, "d")
    ax = fig.add_axes([0.75, 0.07, 0.23, 0.3])
    d = ext[ext.strategy == "TB"].sort_values("frac")
    km = np.r_[0, d.km.values]; red = np.r_[0, d.red_dsun_sp_day.values]
    inc = np.diff(red) / np.diff(km)
    ax.bar(range(4), inc, color=C_STR["TB"], width=0.6)
    for i, x in enumerate(inc):
        ax.text(i, x + 0.05, f"{x:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["0–1%", "1–3%", "3–5%", "5–10%"], fontsize=7.5)
    ax.set_xlabel("Treated length class"); ax.set_ylabel("Added ΔDday (pp per km)")
    ax.grid(axis="y", color=GRID, lw=0.4); panel(ax, "e")
    save(fig, "Figure8")


# ---------------------------------------------------------------- Figure 9: Sentinel-2 shadow check
def fig9():
    from rasterio.warp import transform_bounds
    vt = pd.read_csv(TAB / "rev_s2_validation.csv")
    meta = json.load(open(PROC / "rev_s2_validation.json"))
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 0.85], wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    sc = min(meta["scenes"], key=lambda x: x["elev"])
    img = np.load(PROC / "rev_s2_example.npz")
    rgb = np.clip(img["rgb"] / np.percentile(img["rgb"], 99), 0, 1)
    e = img["extent"]
    ax.imshow(rgb, extent=[e[0], e[2], e[1], e[3]], interpolation="nearest")
    sh = np.ma.masked_less(img["shade10"], 0.5)
    ax.imshow(sh, extent=[e[0], e[2], e[1], e[3]], cmap=mpl.colors.ListedColormap(["#ffd400"]), alpha=0.35,
              interpolation="nearest")
    mapdecor(ax, e[0], e[1], e[2], e[3], bar=200)
    panel(ax, "a")
    ax.text(1.0, 1.02, f"{pd.to_datetime(sc['datetime']).day} {pd.to_datetime(sc['datetime']).strftime('%b %Y')}, "
            f"α = {sc['elev']:.0f}°", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    ax = fig.add_subplot(gs[0, 1])
    d = vt[vt.stratum == "all streets"]
    hv = ["3D-GloBFP", "class correction", "uniform correction", "constant median", "no buildings"]
    col = dict(zip(hv, ["#e0602a", "#c98a00", "#7b3fa0", "#2a6fc4", "#8c8b86"]))
    scenes = sorted(d.scene.unique())
    for h in hv:
        x = d[d.heights == h].set_index("scene").loc[scenes]
        ax.plot(range(len(scenes)), x.auc, marker="o", ms=3, lw=1.1, color=col[h], label=h.capitalize() if h != "3D-GloBFP" else h)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([pd.to_datetime(s_).strftime("%b") + "\n" + pd.to_datetime(s_).strftime("%Y") +
                        f"\n{d[d.scene == s_].elev.iloc[0]:.0f}°" for s_ in scenes], fontsize=6.3)
    ax.set_ylabel("AUC, street pixels"); ax.grid(axis="y", color=GRID, lw=0.4); ax.set_ylim(0.49, 0.86)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.4); panel(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    d = vt[(vt.heights == "3D-GloBFP") & (vt.stratum != "all streets")]
    st_ = ["arterials", "local streets", "tree-lined"]
    for k_, s_ in enumerate(st_):
        x = d[d.stratum == s_]
        ax.scatter(np.full(len(x), k_) + np.linspace(-0.15, 0.15, len(x)), x.balanced_accuracy, s=12,
                   color=["#e0602a", "#2a6fc4", "#1b9a6c"][k_])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Arterials", "Local", "Tree-lined"], fontsize=7.5)
    ax.set_ylim(0.4, 1); ax.axhline(0.5, color=INK2, lw=0.6, ls="--")
    ax.set_ylabel("Balanced accuracy"); ax.grid(axis="y", color=GRID, lw=0.4); panel(ax, "c")
    save(fig, "Figure9")


# ---------------------------------------------------------------- Figure 10: weight simplex
def fig10():
    d = pd.read_csv(TAB / "rev_weight_grid.csv"); d["w1"] = d.w1.round(1); d["w2"] = d.w2.round(1)
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.32, 1, 1], hspace=0.5, wspace=0.3)
    h3 = np.sqrt(3) / 2
    vmax = d.dDsun_med.max()
    for k, t in enumerate(PERIODS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        x = d[d.time == t]
        px = x.w2 + 0.5 * x.w1; py = h3 * x.w1
        sc = ax.scatter(px, py, c=x.dDsun_med, cmap="YlOrRd", vmin=0, vmax=vmax, s=28, marker="h", lw=0)
        ax.plot([0, 1, 0.5, 0], [0, 0, h3, 0], color=INK2, lw=0.5)
        ax.plot(1 / 3 + 0.5 / 3, h3 / 3, marker="o", mfc="none", mec=INK, ms=7, mew=1.0)
        ax.text(0.5, h3 + 0.05, "Shade", ha="center", fontsize=6.5); ax.text(-0.04, -0.09, "LST", fontsize=6.5)
        ax.text(0.92, -0.09, "NDVI", fontsize=6.5)
        ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.12, 1.0); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"({'abcd'[k]}) {t}", fontsize=8, loc="left")
    cax = fig.add_axes([0.06, 0.08, 0.38, 0.025])
    fig.colorbar(sc, cax=cax, orientation="horizontal", label="Median ΔDsun of B10 (%)")
    ax = fig.add_subplot(gs[:, 3:])
    for t in PERIODS:
        x = d[d.time == t].groupby("w1").dDsun_med.agg(["median", "min", "max"])
        ax.fill_between(x.index, x["min"], x["max"], color=C_PER[t], alpha=0.2, lw=0)
        ax.plot(x.index, x["median"], color=C_PER[t], marker=M_PER[t], ms=3, lw=1.2, label=t)
    ax.axvline(1 / 3, color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("Shade weight w1"); ax.set_ylabel("Median ΔDsun of B10 (%)"); ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, fontsize=7); panel(ax, "e")
    save(fig, "Figure10")


# ---------------------------------------------------------------- supplementary figures
def figS1():
    conv = summ["rq4"]["convergence"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for t in PERIODS:
        y = conv[t]["running_red"]
        ax.plot(range(1, len(y) + 1), y, color=C_PER[t], marker=M_PER[t], ms=2.5, lw=1.2, label=t)
    ax.set_xlabel("Number of realisations"); ax.set_ylabel("Running median of ΔDsun (%)")
    ax.grid(color=GRID, lw=0.4); ax.legend(frameon=False, ncol=2, fontsize=7)
    save(fig, "FigureS1")


def figS2():
    rq1 = pd.read_csv(TAB / "network_sunlit_by_hour.csv")
    rq1 = rq1[rq1.time.isin(HOURLY)]
    ori = ["N-S", "NE-SW", "E-W", "NW-SE"]
    M = rq1[ori].values.T
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(4)); ax.set_yticklabels([o.replace("-", "–") for o in ori])
    ax.set_xticks(range(len(rq1))); ax.set_xticklabels([t[:2] for t in rq1.time])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.7 else INK)
    ax.set_xlabel("Local time (h, UTC+3)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Sunlit fraction")
    save(fig, "FigureS2")


if __name__ == "__main__":
    import sys
    for k in sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        globals()[f"fig{k}"]()
        print("figure", k, flush=True)

"""Edge thermal impedance: composite thermal exposure index (CTEI) with a convex sun penalty."""
import numpy as np
import geopandas as gpd
from config import *


def robust_minmax(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a), 0, 1)


def load_segments():
    seg = gpd.read_file(PROC / "network_segments.gpkg")
    seg["covered"] = seg["covered"].astype(bool)
    seg["ndvi_n"] = robust_minmax(seg["ndvi"].values)
    seg["lst_n"] = robust_minmax(seg["lst"].values)
    return seg


def ctei(sunlit, ndvi_n, lst_n, w):
    w1, w2, w3 = w
    return w1 * sunlit + w2 * (1 - ndvi_n) + w3 * lst_n


def thermal_cost(length, sunlit, ndvi_n, lst_n, w, beta=BETA, tau=TAU_S):
    pen = 1 + beta * np.maximum(0, sunlit - tau) ** 2
    return length * ctei(sunlit, ndvi_n, lst_n, w) * pen

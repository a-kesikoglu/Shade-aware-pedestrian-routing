"""Raster shadow casting on a 1 m surface model (shifting method of Ratti and Richens, 2004)."""
import numpy as np
import pandas as pd
import pvlib
from config import *


def sun_positions(times_local):
    idx = pd.DatetimeIndex([pd.Timestamp(f"{DAY} {t}", tz=TZ) for t in times_local])
    sp = pvlib.solarposition.get_solarposition(idx, CENTER_LAT, CENTER_LON, method="nrel_numpy")
    # "azimuth" is the grid azimuth on the UTM raster; "azimuth_true" refers to true north
    return pd.DataFrame({"time": times_local, "azimuth": sp["azimuth"].values - GRID_CONVERGENCE,
                         "azimuth_true": sp["azimuth"].values,
                         "elevation": sp["apparent_elevation"].values})


def _shift(a, dr, dc):
    """out[r, c] = a[r + dr, c + dc]; zero outside."""
    out = np.zeros_like(a)
    H, W = a.shape
    r0, r1 = max(0, -dr), min(H, H - dr)
    c0, c1 = max(0, -dc), min(W, W - dc)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def cast(bld_h, canopy_h, azimuth, elevation, res=1.0, max_len=SHADOW_CAP, trunk=0.25, tau_c=CANOPY_TRANSMISSIVITY, under_crown=True):
    """Return direct-beam transmission factor at ground level (1 = sunlit, 0 = building shade).
    A ground point is shaded by a building pixel at horizontal distance d towards the sun if the
    building is taller than d * tan(elevation); canopy pixels block rays passing between the
    trunk zone (trunk * height) and the canopy top."""
    if elevation <= 0:
        return np.zeros_like(bld_h)
    az = np.deg2rad(azimuth)
    sx, sy = np.sin(az), np.cos(az)          # towards the sun: +x east, +y north
    step = 1.0 / max(abs(sx), abs(sy))       # pixels per iteration along the sun vector
    dz = step * res * np.tan(np.deg2rad(elevation))
    hmax = max(float(bld_h.max()), float(canopy_h.max()))
    n = int(min(max_len / res, hmax / np.tan(np.deg2rad(elevation)) / res) / step) + 1
    bshade = np.zeros(bld_h.shape, bool)
    # ground directly beneath a crown; with under_crown=False only the ray test decides
    vshade = (canopy_h > 0) if under_crown else np.zeros(canopy_h.shape, bool)
    for k in range(1, n + 1):
        dc = int(round(k * step * sx))
        dr = int(round(-k * step * sy))      # rows increase southwards
        z = k * dz
        bq = _shift(bld_h, dr, dc)
        bshade |= bq > z
        cq = _shift(canopy_h, dr, dc)
        vshade |= (cq > z) & (trunk * cq <= z)
    tr = np.where(bshade, 0.0, np.where(vshade, tau_c, 1.0)).astype("float32")
    return tr

"""Shared configuration for the cool-routes analysis pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for p in (RAW, PROC, FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

# Study area: central Kayseri, 3 km x 3 km centred on Cumhuriyet Square
CENTER_LAT, CENTER_LON = 38.7215, 35.4870
HALF_SIZE_M = 1500.0
CRS_UTM = "EPSG:32636"
TZ = "Europe/Istanbul"

# Analysis day and diurnal periods (local time, UTC+3)
DAY = "2025-07-21"
PERIODS = ["09:00", "12:00", "15:00", "17:30"]
HOURLY = [f"{h:02d}:00" for h in range(7, 20)]

MARGIN = 700.0              # m, building data beyond the study area (shadows cast into it)
SHADOW_CAP = 700.0          # m, maximum horizontal search distance of the shadow casting
FLOOR_HEIGHT = 3.0          # m per storey
SEG_MAX = 25.0              # m, maximum edge length after densification
DSM_RES = 1.0               # m
CANOPY_TRANSMISSIVITY = 0.03  # Konarska et al. (2014)
BUFFER_R = 10.0             # m, buffer for NDVI/LST sampling
TAU_S = 0.70
BETA = 4.0
WEIGHTS = {"balanced": (1/3, 1/3, 1/3),
           "shade": (0.6, 0.2, 0.2),
           "vegetation": (0.2, 0.6, 0.2),
           "lst": (0.2, 0.2, 0.6),
           "shade_only": (1.0, 0.0, 0.0)}
SEED = 20250721
# Meridian convergence at the study-area centre (UTM 36N), about 1.56 deg.
# Solar azimuths from pvlib refer to true north; the rasters are in UTM grid coordinates, so azimuths are
# converted with az_grid = az_true - GRID_CONVERGENCE before casting (true north has grid bearing -GRID_CONVERGENCE).
def _grid_convergence():
    from pyproj import Proj
    return float(Proj(CRS_UTM).get_factors(CENTER_LON, CENTER_LAT).meridian_convergence)
GRID_CONVERGENCE = _grid_convergence()

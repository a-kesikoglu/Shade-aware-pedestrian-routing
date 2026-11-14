"""Run the complete analysis in order and log every step (data/processed/run_all_log.txt).

Usage: python run_all.py [--from STEP]   e.g. --from 03 skips the downloads and the surface model.
Every number in the article comes from the outputs written by these steps.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("01a", "src/01a_download_osm.py"), ("01b", "src/01b_download_satellite.py"),
    ("01c", "src/01c_download_canopy.py"), ("01d", "src/01d_clip_globfp.py"),
    ("01e", "src/01e_download_extra.py"), ("02", "src/02_surface_model.py"),
    ("03", "src/03_shadows_network.py"), ("04", "src/04_routing_experiments.py"),
    ("05", "src/05_height_uncertainty.py"), ("06", "src/06_intervention.py"),
    ("10", "src/10_routing_checks.py"), ("11", "src/11_shade_checks.py"),
    ("12", "src/12_height_checks.py"), ("13", "src/13_priority_checks.py"),
    ("20", "src/20_rev_routing.py"), ("21", "src/21_rev_heights.py mc"), ("22", "src/22_rev_canopy.py"),
    ("23", "src/23_rev_s2_validation.py"), ("23b", "src/23b_rev_s2_example.py"), ("24", "src/24_rev_priority.py"),
    ("25", "src/25_rev_weights.py"), ("26", "src/26_rev_lst.py"), ("27", "src/27_rev_search_tests.py"), ("28", "src/28_rev_od_overlap.py"), ("29", "src/29_rev_removed_segments.py"),
    ("07", "src/07_summary.py"), ("09", "src/09_tables.py"), ("08", "src/08_figures.py"),
]

start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else STEPS[0][0]
keys = [k for k, _ in STEPS]
log = open(ROOT / "data" / "processed" / "run_all_log.txt", "a", encoding="utf-8")
for key, script in STEPS[keys.index(start):]:
    t0 = time.time()
    cwd = ROOT / Path(script.split()[0]).parent
    r = subprocess.run([sys.executable, *Path(script).name.split()], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {key:4s} {script:40s} exit {r.returncode}  {time.time() - t0:7.0f} s"
    print(line, flush=True); log.write(line + "\n"); log.flush()
    (ROOT / "data" / "processed" / f"run_{key}.log").write_text(r.stdout[-200000:], encoding="utf-8")
    if r.returncode != 0:
        log.write(r.stderr[-4000:] + "\n"); print(r.stderr[-4000:]); sys.exit(r.returncode)

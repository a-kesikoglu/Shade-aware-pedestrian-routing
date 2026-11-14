# Shade-aware pedestrian routing from open 3D data (Kayseri)

Code, inputs and results for the article *Shade-aware pedestrian routing and shade prioritisation from open 3D data in Kayseri, Türkiye* (Research & Design, Kayseri University).

The workflow builds a 1 m building and canopy surface model of central Kayseri from open global data, casts hourly shadows on the OSM pedestrian network for 21 July 2025, computes exact bounded biobjective (distance and composite thermal cost) Pareto sets for 500 origin–destination pairs at four times of day, and ranks street segments for shading by thermal betweenness evaluated on a separate set of trips.

## Contents

```
src/                         analysis pipeline (run from src/)
  config.py                  study area, dates, parameters and random seeds (base seed 20250721)
  run_all.py                 runs every step in order and logs it
  01a-01e                    downloads: OSM network, footprints and POIs (OSMnx), Sentinel-2 L2A and Landsat C2 L2
                             (Planetary Computer STAC), 1 m canopy height map, 3D-GloBFP tile 1452, Copernicus GLO-30
  02_surface_model.py        building heights (OSM + 3D-GloBFP), 1 m rasters, NDVI, summer LST median
  03_shadows_network.py      hourly shadows and densified network with edge attributes
  04_routing_experiments.py  OD sampling, exact Pareto routing, satellite baseline, weight and penalty runs
  05, 12                     building height validation and Monte Carlo error propagation
  06, 13                     thermal betweenness, shade intervention scenarios and robustness checks
  10, 11                     routing verification (path enumeration) and shade checks (analytic shadows, margins,
                             facade-side corridors, canopy and terrain runs)
  20_rev_routing.py          unsupported Pareto solutions, direct minimisation of sunlit distance, paired tests
  21_rev_heights.py          reference sample checks, storey conversion, height-dependent error model ("mc" argument)
  22_rev_canopy.py           canopy shadow rule against an exact ray test
  23_rev_s2_validation.py    modelled shadows against a Sentinel-2 dark-pixel proxy (scene manifest:
                             outputs/tables/rev_s2_scene_manifest.csv; paired AUC: rev_s2_paired_auc.csv)
  24_rev_priority.py         demand variants, 100 random placements, corridor names, crossing and sidewalk tags
  25_rev_weights.py          composite cost weights over the simplex
  26_rev_lst.py              Landsat scene statistics and single-scene run
  27_rev_search_tests.py     deterministic edge-case tests of the Pareto search (ties, budget and detour limits, knee)
  28_rev_od_overlap.py       overlap of evaluation and flow pairs, occupancy of the block bootstrap cells
  29_rev_removed_segments.py removed underpass length by OSM way, location and reason
                             (outputs/tables/rev_removed_segments.csv)
  07_summary.py, 09_tables.py, 08_figures.py   statistics, tables and figures
  network.py, shade.py, routing.py, impedance.py, sampling.py, mcworker.py   shared modules
data/raw/                    clipped inputs used in the article (OSM snapshot of 3-4 November 2026 as GraphML and
                             GeoPackage, 3D-GloBFP clip, canopy height, Sentinel-2 and Landsat clips, GLO-30)
data/processed/              surface rasters, network segments and nodes, OD pairs (pickle and CSV), result files
outputs/tables/              inputs of all tables (CSV); outputs/results_summary.json holds the values cited in the text
```

The full 3D-GloBFP tile (figshare file 54039878) is not included because of its size; `01d_clip_globfp.py` downloads it.

## Reproduce

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cd src
python run_all.py --from 02        # start from the included inputs; omit --from to download everything again
```

Solar azimuths from pvlib refer to true north and are converted to UTM grid north (meridian convergence of 1.56° at the study area centre, `config.GRID_CONVERGENCE`) before shadows are cast. Release v1.1 is the version that produces the results of the revised article.

Tested with Python 3.12.10 on Windows 10. The exact package versions are pinned in `requirements.txt`, and `requirements-lock.txt` lists the full environment (`pip freeze`). Shadow casting takes a few seconds per hour; the Monte Carlo and weight-grid steps run for several hours on a laptop.

## Data sources and licences

- OpenStreetMap data © OpenStreetMap contributors, Open Database Licence (ODbL). Derived network and building files in `data/` are distributed under the ODbL.
- 3D-GloBFP: Che et al. (2024), Earth System Science Data 16, 5357-5374 (CC BY 4.0).
- Global canopy height map: Tolan et al. (2024), Meta and World Resources Institute (CC BY 4.0).
- Copernicus Sentinel-2 and Copernicus DEM GLO-30 data (ESA, Copernicus licence); Landsat 8 and 9 Collection 2 data courtesy of the U.S. Geological Survey.
- Code: MIT licence (see `LICENSE`).

## Citation

Kesikoğlu A. Shade-aware pedestrian routing and shade prioritisation from open 3D data in Kayseri, Türkiye. Research & Design (under review).

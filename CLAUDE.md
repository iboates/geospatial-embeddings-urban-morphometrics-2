# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Urban morphometrics computation pipeline that calculates urban form metrics from OpenStreetMap data. Reads a study area GeoDataFrame (H3 hex cells with `region_id` index), loads OSM buildings/highways/landuse via QuackOSM, and computes per-cell aggregated metrics.

## Commands

```bash
# Install dependencies
poetry install

# Run the pipeline (CLI)
python -m urban_morphometrics study_area.gpkg pbf_url run_name /output/folder \
  --neighbourhood-distance 500 --num-quantiles 10 --metrics "volume,floor_area" --debug

# Quick test run (Karlsruhe, Germany)
python run.py
```

No test suite, linter, or formatter is configured yet.

## Architecture

### Data Flow

Study area GeoDataFrame → `load_osm_data()` → `OsmData(buildings, highways, landuse)` → per-cell `CellContext` (lazy-computed, Parquet-cached) → `compute_metrics()` → results GeoPackage.

### Key Modules

- **`main.py`** — CLI entry point and `compute_urban_morphometrics()` orchestrator
- **`osm_loader.py`** — QuackOSM PBF loading with download caching
- **`cell_context.py`** — Per-cell spatial context; all properties are `@cached_property` backed by Parquet files for pipeline resumption
- **`height.py`** — Building height resolution: `height` tag → `building:levels × 3m` → default `6m`
- **`metrics/__init__.py`** — Registry-based dispatch (`@register("name")` decorator) and `compute_metrics()`
- **`metrics/aggregation.py`** — `aggregate_series()` produces `{prefix}_mean`, `_median`, `_std`, `_q10`…`_q100`

### Metric Function Contract

Each metric is a decorated function: `@register("metric_name")` with signature `def compute(ctx: CellContext, num_quantiles: int) -> dict[str, float]`. Returns a flat dict of aggregated statistics.

### Three CRS Strategy

- **Equal-area (EPSG:3395)** — area/volume calculations
- **Equidistant (EPSG:4087)** — distance/network metrics
- **Conformal (EPSG:3857)** — shape metrics (angle-preserving)

CellContext provides buildings pre-projected to each CRS (`buildings_ea`, `buildings_ed`, `buildings_cf`).

### Cache Structure

```
{output_folder}/{run_name}/
├── cache/{region_id}/     # Per-cell Parquet files (buildings, highways, metrics)
├── results/metrics.gpkg   # Final output
└── debug/                 # Optional debug GeoPackages
```

## Implementation Status

Following a 12-step plan in `IMPLEMENTATION_PLAN.md`. Steps 1–5 complete (scaffolding, OSM loading, CellContext, metric registry, dimension + shape metrics). Next: distribution, intensity, and street connectivity metrics. See `SPECIFICATION.md` and `METRICS.md` for the full metric catalogue.

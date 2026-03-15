# Implementation Plan

This plan implements the urban morphometrics pipeline described in SPECIFICATION.md and METRICS.md. Each step produces a working, reviewable state.

---

## Step 1: Project scaffolding and CLI entry point

Create the package structure and Fire CLI entry point.

**Files to create:**
- `urban_morphometrics/__init__.py`
- `urban_morphometrics/cli.py` — defines `compute_urban_morphometrics()` with all parameters from the spec (study_area_gdf path, pbf_url, run_name, neighbourhood_distance, num_quantiles, output_folder, metrics list, equal_area_crs, equidistant_crs, conformal_crs, debug). Uses Fire to expose it. For now the function body just validates inputs, creates the output folder structure, and prints the config.
- `urban_morphometrics/constants.py` — vehicle highway types, pedestrian highway types, default height, metres-per-storey constant.

**Output folder structure created by the CLI:**
```
{output_folder}/{run_name}/
    cache/              # intermediate cached data per region
    results/            # final metric GeoDataFrames
    debug/              # debug layers (only populated if debug=True)
```

**Reviewable state:** `python -m urban_morphometrics.cli --help` works and shows all parameters. Calling it with dummy args creates the folder structure.

---

## Step 2: OSM data loading

Implement the QuackOSM data loading step.

**Files to create/modify:**
- `urban_morphometrics/osm_loader.py` — a function `load_osm_data(pbf_path, study_area_gdf)` that:
  1. Computes the unary union of `study_area_gdf` as the geometry filter
  2. Calls `convert_pbf_to_geodataframe` three times with the appropriate tag filters:
     - buildings: `{"building": True, "building:levels": True, "height": True}`
     - highways: `{"highway": True, "oneway": True}`
     - landuse: `{"landuse": True}`
  3. Uses `keep_all_tags=False`
  4. Returns a named tuple or dataclass `OsmData(buildings, highways, landuse)`
- `urban_morphometrics/cli.py` — update to load the study area GeoDataFrame from a file path (GeoPackage/GeoJSON/Shapefile), validate it has `region_id` and is in EPSG:4326, then call the loader.

**Reviewable state:** Running the CLI with a real .pbf and study area loads three GeoDataFrames and logs their sizes. Debug mode dumps them to the debug folder.

---

## Step 3: CellContext class — data preparation and projection

Implement the CellContext class that prepares per-cell data with caching.

**Files to create:**
- `urban_morphometrics/cell_context.py` — the `CellContext` class. Constructor takes:
  - `region_id`, `cell_geometry` (single polygon, WGS84)
  - `osm_data` (the global OsmData)
  - `neighbourhood_distance`
  - `equal_area_crs`, `equidistant_crs`, `conformal_crs`
  - `cache_dir` (path under `{output_folder}/{run_name}/cache/{region_id}/`)

  Lazy-computed (cached) properties:
  1. **`buildings_ea`/`buildings_ed`/`buildings_cf`** — buildings intersecting the cell, projected to equal-area / equidistant / conformal CRS
  2. **`buildings_with_height`** — buildings in the cell with height resolved per the spec logic (prefer `height` column → `building_levels * 3` → default 6m)
  3. **`vehicle_highways_ea`/`_ed`/`_cf`** — highways filtered to vehicle types, projected; oneway parsed via `_parse_oneway`
  4. **`pedestrian_highways_ea`/`_ed`/`_cf`** — highways filtered to pedestrian types, projected
  5. **`landuse_ea`** — landuse projected to equal-area CRS
  6. **`neighbourhood_buildings`** — buildings within `neighbourhood_distance` of the cell but NOT intersecting it, projected
  7. **`neighbourhood_highways`** — highways within `neighbourhood_distance` of the cell but NOT intersecting it, projected
  8. **`focal_plus_neighbourhood_buildings`** — union of cell buildings + neighbourhood buildings (convenience property for neighbourhood metrics)
  9. **`focal_plus_neighbourhood_highways`** — same for highways

- `urban_morphometrics/height.py` — `resolve_heights(buildings_gdf)` function implementing the height inference logic from the spec, and the `_infer_height_from_tags` helper.
- `urban_morphometrics/oneway.py` — `parse_oneway(tags, highway_val)` and `_default_oneway(highway_val, junction)` functions from the spec.

**Caching strategy:** Each lazy property serializes its result to Parquet in the cache dir. On construction, if the Parquet file exists, it's loaded instead of recomputed. This allows the pipeline to resume.

**Reviewable state:** A test script can create a CellContext for one region, access each property, and see correct projections and filtering. Cache files appear in the cache dir. A second run loads from cache.

---

## Step 4: Metric registry and aggregation framework

Build the infrastructure for computing and aggregating metrics before implementing any specific metric.

**Files to create:**
- `urban_morphometrics/metrics/__init__.py` — metric registry. A dictionary mapping metric name strings to metric functions. A `compute_metrics(cell_context, metric_names, num_quantiles)` function that:
  1. Looks up each requested metric in the registry
  2. Calls its function with the CellContext
  3. The function returns a dict of `{column_name: value}` pairs (already aggregated)
  4. Merges all dicts into a single row dict for that cell
- `urban_morphometrics/metrics/aggregation.py` — `aggregate_series(series, prefix, num_quantiles)` helper that takes a pandas Series of per-feature values and returns a dict with keys like `{prefix}_mean`, `{prefix}_median`, `{prefix}_std`, `{prefix}_q1` ... `{prefix}_qN`.
- `urban_morphometrics/cli.py` — update the main function to loop over cells, create CellContext, compute metrics, collect rows, and assemble the final GeoDataFrame. Add per-region caching of metric results (Parquet) so it can resume. Write final results to `{output_folder}/{run_name}/results/`.

**Reviewable state:** The pipeline runs end-to-end but produces a GeoDataFrame with only `region_id` and `geometry` (no metric columns yet). The metric loop, caching, and output writing all work.

---

## Step 5: Dimension metrics

Implement all dimension metrics (no neighbourhood context needed).

**Files to create:**
- `urban_morphometrics/metrics/courtyard_area.py` — dissolve adjacent buildings, extract interior rings, compute areas, aggregate. Also returns courtyard count.
- `urban_morphometrics/metrics/floor_area.py` — `footprint_area * floor_count` (floors from `building_levels` or `height / 3`).
- `urban_morphometrics/metrics/longest_axis_length.py` — wraps `momepy.longest_axis_length`.
- `urban_morphometrics/metrics/perimeter_wall.py` — wraps `momepy.perimeter_wall`.
- `urban_morphometrics/metrics/volume.py` — `footprint_area * height`.

**Register** all five in the metric registry.

**Reviewable state:** Running the pipeline with `--metrics courtyard_area,floor_area,longest_axis_length,perimeter_wall,volume` produces a results GeoDataFrame with the corresponding aggregated columns.

---

## Step 6: Shape metrics

Implement all shape metrics (no neighbourhood context needed).

**Files to create (one per metric, all in `urban_morphometrics/metrics/`):**
- `circular_compactness.py`
- `square_compactness.py`
- `convexity.py`
- `courtyard_index.py`
- `rectangularity.py`
- `shape_index.py`
- `corners.py`
- `squareness.py`
- `equivalent_rectangular_index.py`
- `elongation.py`
- `facade_ratio.py` — computed per raw building AND per dissolved structure (two prefixes)
- `fractal_dimension.py` — computed per raw building AND per dissolved structure
- `form_factor.py` — uses building height for 3D compactness
- `compactness_weighted_axis.py` — per raw building AND per dissolved structure
- `centroid_corner_distance.py` — per raw building AND per dissolved structure

Most of these are thin wrappers around the corresponding `momepy.*` function, with `aggregate_series` applied to produce the summary statistics.

**Register** all in the metric registry.

**Reviewable state:** Running with `--metrics circular_compactness,elongation,...` produces correct aggregated columns.

---

## Step 7: Distribution metrics (neighbourhood context)

Implement spatial distribution metrics. These require building spatial graphs over the focal + neighbourhood buildings.

**Files to create (in `urban_morphometrics/metrics/`):**
- `orientation.py` — wraps `momepy.orientation` (no neighbourhood needed, but listed here for grouping)
- `shared_walls.py` — `momepy.shared_walls` on focal+neighbourhood buildings, then filter results to focal buildings only for aggregation. Also compute ratio variant.
- `alignment.py` — build KNN graph (k=15) on focal+neighbourhood buildings, compute `momepy.alignment`, filter to focal.
- `neighbor_distance.py` — build Delaunay graph on focal+neighbourhood, `momepy.neighbor_distance`, filter to focal.
- `mean_interbuilding_distance.py` — `momepy.mean_interbuilding_distance` on focal+neighbourhood, filter to focal.
- `building_adjacency.py` — `momepy.building_adjacency` with rook contiguity + KNN on focal+neighbourhood, filter to focal.
- `neighbors.py` — build morphological tessellation on focal+neighbourhood, queen contiguity graph on tessellation, `momepy.neighbors`, filter to focal.
- `cell_alignment.py` — compare building orientation to tessellation cell orientation (focal+neighbourhood), filter to focal.
- `street_alignment.py` — `momepy.get_nearest_street` (500m max) on focal+neighbourhood buildings against focal+neighbourhood highways, compare orientations, filter to focal.

**Shared utility (in the metrics folder or cell_context):**
- A helper to build tessellation + common spatial graphs (KNN, Delaunay, contiguity) since several metrics share these. Cache the intermediate spatial graphs per cell.

**Reviewable state:** Running with neighbourhood metrics produces correct columns. Verify that edge buildings (near cell boundary) use neighbourhood context correctly.

---

## Step 8: Intensity and street relationship metrics

**Files to create (in `urban_morphometrics/metrics/`):**
- `courtyards.py` — `momepy.courtyards` with rook contiguity graph on focal+neighbourhood buildings.
- `street_profile.py` — `momepy.street_profile` on focal+neighbourhood highways with focal+neighbourhood buildings. Returns width, openness, height/width ratio as separate columns.
- `nearest_street_distance.py` — custom: minimum distance from each building centroid to nearest street segment using focal+neighbourhood highways.

**Reviewable state:** All building-level and street-relationship metrics work.

---

## Step 9: Street connectivity metrics — graph construction

Build the street network graph infrastructure shared by all connectivity metrics.

**Files to create/modify:**
- `urban_morphometrics/street_graph.py` — functions to:
  1. Convert focal+neighbourhood highways GeoDataFrame to a momepy-compatible NetworkX graph (using `momepy.gdf_to_nx` or equivalent)
  2. Build **vehicle graph** (directed, respecting oneway) and **pedestrian graph** (undirected)
  3. After computing all node-level attributes, filter results to nodes within the focal cell for aggregation

**Reviewable state:** Given a cell's highways, produces a valid NetworkX graph. A simple test computes node count and edge count.

---

## Step 10: Street connectivity metrics — per-node metrics

Implement all per-node connectivity metrics, each aggregated to mean/median/std/quantiles.

**Files to create (in `urban_morphometrics/metrics/`):**
- `degree.py` — `momepy.node_degree`
- `meshedness.py` — `momepy.meshedness` (with radius)
- `mean_node_dist.py` — `momepy.mean_node_dist`
- `mean_node_degree.py` — `momepy.mean_node_degree`
- `gamma.py` — `momepy.gamma` (with radius)
- `edge_node_ratio.py` — `momepy.edge_node_ratio`
- `cyclomatic.py` — `momepy.cyclomatic`
- `clustering.py` — `momepy.clustering`
- `closeness_centrality.py` — `momepy.closeness_centrality`
- `betweenness_centrality.py` — `momepy.betweenness_centrality`
- `straightness_centrality.py` — `momepy.straightness_centrality`

Each metric is computed for both the vehicle and pedestrian graph, producing `{metric}_vehicle` and `{metric}_pedestrian` prefixed columns.

**Reviewable state:** Running with `--metrics degree,meshedness,...` produces vehicle and pedestrian variants with aggregated statistics.

---

## Step 11: Street connectivity metrics — graph-level and proportion metrics

Implement single-value-per-cell connectivity metrics.

**Files to create (in `urban_morphometrics/metrics/`):**
- `meshedness_global.py` — `momepy.meshedness(radius=None)` returning a single float per graph.
- `gamma_global.py` — `momepy.gamma(radius=None)`.
- `cds_length_total.py` — `momepy.cds_length` total.
- `proportion.py` — `momepy.proportion` → extract `proportion_three`, `proportion_four`, `proportion_dead`.

Each computed for both vehicle and pedestrian graphs.

**Reviewable state:** All connectivity metrics complete. Full metric catalogue from METRICS.md is covered.

---

## Step 12: End-to-end integration, error handling, and debug output

Polish the pipeline for production use.

**Changes:**
- `cli.py` — add logging throughout (use Python `logging`). Log progress per region (e.g., "Processing region 42/500"). Add error handling: if a single cell fails, log the error and continue to the next cell (don't crash the whole run).
- Debug mode: when `debug=True`, dump intermediate layers to `{output_folder}/{run_name}/debug/` — the loaded OSM data, per-cell buildings/highways after filtering, tessellations, spatial graphs.
- Validate the `metrics` list input against the registry and fail fast with a clear message if an unknown metric is requested.
- Add a `__main__.py` so `python -m urban_morphometrics` works.
- Wire up the Fire CLI entry point in `pyproject.toml` under `[project.scripts]`.

**Reviewable state:** The full pipeline runs end-to-end on real data with informative logging, graceful error handling, and debug outputs. Unknown metric names are rejected early.

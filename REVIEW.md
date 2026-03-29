# Urban Morphometrics Pipeline — Comprehensive Review

This document evaluates the current pipeline against the full momepy API, the broader urban morphology literature, and the specific goal of producing rich, informative embeddings for geospatial neural networks. It is organised as a gap analysis: what the pipeline does well, where it can be improved, and what it is missing entirely.

---

## 1. Summary Verdict

The pipeline is a solid, well-engineered starting point. It correctly implements the core momepy building-geometry and street-network metrics, handles edge effects via neighbourhood buffering, uses appropriate CRS strategies per metric family, and produces a wide statistical distribution (mean/median/std/quantiles) for each measure. The result is a ~290+ column feature vector per H3 cell.

However, for the goal of creating a *comprehensive morphological profile* suitable for training geospatial embeddings, three critical categories of information are entirely absent:

1. **Building intensity / density** — how much floor area is packed into the cell relative to its area (GSI, FSI/FAR, OSR). These are arguably the most canonical metrics in the built-environment literature.
2. **Land use composition** — what the cell's ground surface is allocated to (residential, commercial, green, industrial, etc.). Land use is already loaded from OSM but not yet converted to metrics.
3. **Vertical variation** — the statistical distribution of building heights as a standalone measure, separate from the proxies already captured by `volume` and `form_factor`.

Beyond these three critical gaps, there are a number of secondary gaps (street density, tessellation size, building use diversity, mixed-use indicators) and several quality issues in the current implementation worth addressing.

---

## 2. Theoretical Framework

### 2.1 What Urban Morphology Literature Considers Essential

The foundational work in quantitative urban morphology (Conzen 1960; Caniggia & Maffei 1979; Moudon 1997; Kropf 2014) identifies three inseparable components of urban form:

- **The street network** — layout, hierarchy, connectivity
- **The plot pattern** — subdivision of land into parcels
- **The building pattern** — footprint, height, arrangement, use

Most contemporary computational frameworks, including momepy (Fleischmann et al. 2019, 2021), add a fourth:

- **The block pattern** — areas enclosed by streets

The current pipeline captures the first and third well, partially the fourth via tessellation, and omits the second (no plot data from OSM, which is expected). What it underweights is the *quantitative relationship between buildings and the space they occupy* — density.

### 2.2 momepy's Theoretical Position

momepy (Morphological Measuring Toolkit for Python) is described in Fleischmann et al. (2019, JOSS) and the companion paper Fleischmann et al. (2021, Environment and Planning B) which introduced the morphological tessellation as the primary analysis unit. The library draws from:

- Dibble et al. (2017) on urban morphometrics
- Berghauser Pont & Haupt (2009) on spacemate / density trivariate (GSI, FSI, OSR)
- Gil et al. (2012) on network-based morphological analysis
- Space Syntax (Hillier & Hanson 1984) for network centrality variants
- Porta et al. (2006) for primal graph network analysis

The key theoretical contribution of momepy is that it works at the *morphological tessellation cell* (Voronoi cell clipped to local context) as the fundamental unit, rather than administrative parcels which are not universally available. This approach is used correctly in the current pipeline for `neighbors`, `cell_alignment`, and the queen-contiguity graph.

---

## 3. Audit of Implemented Metrics

### 3.1 Dimension Metrics — Assessment: **Good with gaps**

| Metric | Assessment |
|--------|-----------|
| `volume` | Correct. footprint × height. Well-implemented with proper height fallback. |
| `floor_area` | Correct. footprint × levels. Reasonable levels fallback. |
| `longest_axis_length` | Correct. momepy.longest_axis_length wraps minimum bounding circle diameter. |
| `perimeter_wall` | Correct. Dissolves touching buildings before measuring. |
| `courtyard_area` | Correct. Extracts interior rings from dissolved structures. |

**Missing in this category:**
- `area` — raw building footprint area. Appears not to be aggregated directly (only used as part of volume/floor_area computations). Including `building_area` distribution statistics would be valuable on its own: it captures building footprint size diversity and is one of the most commonly used urban form descriptors.
- `street_length` — total length and mean/std of individual street segment lengths. momepy provides `momepy.street_length()` for per-edge lengths.
- `GSI`, `FSI`, `OSR` — see Section 4.1.

### 3.2 Shape Metrics — Assessment: **Excellent coverage**

This is the most thoroughly covered category. The 15 implemented metrics span all major shape families:

- **Compactness family**: `circular_compactness`, `square_compactness`, `convexity`, `rectangularity`, `shape_index`, `equivalent_rectangular_index`
- **Complexity family**: `corners`, `squareness`, `fractal_dimension`
- **Proportional family**: `elongation`, `facade_ratio`, `form_factor`
- **Composite family**: `compactness_weighted_axis`, `centroid_corner_distance`
- **Courtyard family**: `courtyard_index`

The dual computation (raw building + dissolved structure) for `facade_ratio`, `fractal_dimension`, `compactness_weighted_axis`, and `centroid_corner_distance` is methodologically sound and captures the difference between individual building complexity and the complexity of joined building clusters.

**Quality note on CRS**: Shape metrics use the conformal CRS (EPSG:3857 / Web Mercator). Web Mercator is only locally angle-preserving and introduces scale distortions that grow with latitude (approximately ±5% at 40° latitude, ±15% at 60°). For research applications targeting high-latitude cities (Scandinavia, Canada, Northern UK), a locally-optimal conformal projection (e.g., Transverse Mercator or a UTM zone) would produce more accurate shape indices. EPSG:3857 is acceptable for mid-latitude applications but should be documented as a limitation.

**Missing in this category:**
- `solar_orientation` — orientation relative to south (0° = north-facing, 180° = south-facing). Useful for solar access and energy performance profiling.
- `setback_distance` — distance from building centroid/edge to nearest street, measured differently from `nearest_street_distance`. Some works measure this as the minimum perpendicular distance from the street edge to the building facade.

### 3.3 Distribution Metrics — Assessment: **Good, one gap**

The nine distribution metrics cover the key spatial arrangement measures momepy provides:

| Metric | Assessment |
|--------|-----------|
| `orientation` | Correct. Deviation from cardinal directions (0–45° range). |
| `shared_walls` | Correct, including the ratio variant. |
| `alignment` | Correct. KNN-15 neighbourhood orientation consistency. |
| `neighbor_distance` | Correct. Delaunay-based neighbour spacing. |
| `mean_interbuilding_distance` | Correct. Combined Delaunay+KNN spacing. |
| `building_adjacency` | Correct. Rook contiguity to KNN ratio. |
| `neighbors` | Correct. Tessellation queen contiguity count. |
| `cell_alignment` | Correct. Building vs. tessellation cell orientation deviation. |
| `street_alignment` | Correct. Building vs. nearest street orientation deviation. |

**Quality notes:**

- `alignment` and `street_alignment` together measure how buildings orient relative to each other and to streets, but neither measures the *absolute* orientation coherence of the whole cell's building fabric. A cell with all buildings at 45° would score the same as one with all buildings at 0° on alignment — only `orientation` captures the absolute angle. This is correct behaviour, but worth noting in downstream interpretation.

- `neighbor_distance` vs `mean_interbuilding_distance`: these are correlated but not redundant. Delaunay-only captures nearest topological neighbours, while combined Delaunay+KNN captures a wider neighbourhood. Both are valuable.

- `building_adjacency` uses rook contiguity which requires precise touching geometries. OSM building polygons frequently have small gaps (0.01–1m) between party-wall buildings due to independent digitising. This will cause systematic under-reporting of adjacency in OSM-derived data compared to cadastral data. This is an inherent OSM data quality limitation but should be noted.

**Missing:**
- `tessellation_area` — distribution statistics (mean, std, CV, quantiles) of morphological tessellation cell areas. The tessellation cell area is the closest proxy for plot size in the absence of cadastral data, and is considered a fundamental dimension metric in the momepy framework. This is computed during tessellation creation but not aggregated as an output metric.

### 3.4 Intensity Metrics — Assessment: **Severely incomplete**

This is the most significant gap in the pipeline. Only `courtyards` is implemented in this category.

**What urban morphology literature considers essential intensity metrics:**

The seminal work on spatial intensity in urban morphology is Berghauser Pont & Haupt's *Spacematrix: Space, Density and Urban Form* (2010). Their "spacemate" framework defines:

- **GSI (Ground Space Index)** = Σ(building_footprint_area) / cell_area
  - Also known as "building coverage ratio" or "lot coverage"
  - Values: sparse rural ~0.02, suburban ~0.15–0.25, dense urban ~0.4–0.7
  - This is one of the two most important urban form descriptors

- **FSI (Floor Space Index)** = Σ(gross_floor_area) / cell_area
  - Also known as "floor area ratio" (FAR) or "plot ratio"
  - Captures total building floor space per unit land area
  - Values: suburban ~0.2–0.5, medium density ~1–2, dense urban core ~3–8+
  - This is the other most important urban form descriptor

- **OSR (Open Space Ratio)** = (1 - GSI) / FSI
  - Amount of open space per unit of gross floor area
  - Captures the relative generosity of outdoor space

- **L (number of Layers / average building height)** = FSI / GSI
  - Also interpretable as average number of storeys
  - Derived from GSI and FSI, so not strictly additional

These four values define the "spacemate" diagram which is a standard tool for classifying urban form typologies (detached houses, row houses, perimeter blocks, point towers, slab blocks, etc.). Without GSI and FSI, the pipeline cannot produce spacemate-compatible profiles, which significantly limits comparability with published urban morphology research.

**Also missing from intensity category:**
- `building_count` — raw number of buildings in the cell. Extremely basic but useful when combined with cell area to give building density.
- `building_density` — building count per cell area (buildings/km²).
- `floor_count_mean/std` — direct statistics on `building:levels` OSM tag values (not just as a height proxy).

### 3.5 Street Relationship Metrics — Assessment: **Good, some gaps**

| Metric | Assessment |
|--------|-----------|
| `street_profile` (width, openness, hw_ratio) | Correct methodology. Perpendicular ticks approach is standard. |
| `nearest_street_distance` | Correct. Building centroid to nearest vehicle street. |

**Quality notes:**

- `street_profile` uses `tick_length=50m`. In dense urban areas with narrow streets (e.g., 8–15m typical European street), this is appropriate. However in suburban or low-density areas, buildings may be more than 50m from the street centreline on one side, causing false "openness" readings. Adaptive tick length based on local density would be more robust.

- `street_profile` is computed only on vehicle highways. Many important urban streets — particularly in pedestrianised areas or historic town centres — are only tagged as `highway=pedestrian` in OSM. Running street profile on pedestrian highways too would capture a more complete picture of the street canyon environment.

- `nearest_street_distance` measures building centroid to nearest vehicle street, but does not distinguish between buildings that are directly fronting a street (low setback) vs. buildings that are set back in a compound. A `nearest_pedestrian_street_distance` variant would capture access via footpaths.

**Missing:**
- `street_canyon_enclosure` — a more sophisticated version of the height-to-width ratio. The current `hw_ratio` from street_profile captures this but only for vehicle streets. A cell-level statistic of street enclosure would be valuable.
- `building_setback_distance` — median perpendicular distance from building facade to street edge, not street centreline. The current `nearest_street_distance` measures centroid to centreline.

### 3.6 Street Connectivity Metrics — Assessment: **Very thorough**

This is the most complete category. The pipeline implements 11 per-node metrics × 2 network types + 4 graph-level metrics × 2 network types = 30 metric families.

**Per-node assessment:**

| Metric | Assessment |
|--------|-----------|
| `degree` | Correct and important. Dead ends vs. T-junctions vs. crossroads. |
| `meshedness` | Correct. Local circuit density at 5-hop radius. |
| `mean_node_dist` | Correct. Block length proxy. |
| `mean_node_degree` | Correct. Smoothed degree. |
| `gamma` | Correct. Edges vs. max possible edges. |
| `edge_node_ratio` | Correct. Redundant with gamma but traditional measure. |
| `cyclomatic` | Correct. Independent cycle count. |
| `clustering` | Correct. Local triangle density. |
| `closeness_centrality` | Correct. Inverse mean shortest path. |
| `betweenness_centrality` | Correct. Through-traffic fraction. |
| `straightness_centrality` | Correct. Euclidean / network distance ratio. |

**Quality notes:**

- `betweenness_centrality` is computed on the entire focal+neighbourhood subgraph. For a 500m neighbourhood at typical street densities, this will be computationally expensive and may produce unstable estimates for larger cells. Consider whether the `radius` parameter (currently = None for all centrality measures) is appropriate. momepy supports spatial weights radius cutoffs.

- `closeness_centrality` and `betweenness_centrality` are computed without normalisation in the current implementation (using NetworkX defaults). For comparison across cells of different sizes and network densities, normalised variants would be more appropriate.

- The vehicle graph is directed (respects oneway streets) while the pedestrian graph is undirected. This is the correct theoretical choice: vehicle accessibility depends on direction, pedestrian accessibility does not (except one-way pedestrian zones, which are rare in OSM).

- `straightness_centrality` is a metric from the Space Syntax and "multiple centrality assessment" (MCA) tradition (Porta et al. 2006). It captures how "direct" routes are through each node, which correlates with commercial activity and pedestrian movement.

**Missing from connectivity:**
- **Reached** — `momepy.reached()` computes how many entities (nodes, buildings, street segments, or area) can be reached within a threshold distance via the network. This is an accessibility metric rather than a pure topology metric and would complement the current centrality measures.
- **COINS (Continuity In Street Networks)** — `momepy.COINS()` groups street segments into natural "strokes" based on azimuth continuity (i.e., streets that form natural continuations of each other). This captures the cognitive legibility of the street network. Stroke count, mean stroke length, and stroke length distribution are meaningful urban form descriptors not captured by node-based metrics.
- **Street type diversity** — Shannon entropy of vehicle highway type values (motorway, primary, secondary, tertiary, residential, etc.) within the cell. Captures whether a cell has a monolithic or hierarchically diverse street network.
- **Total street length and density** — Σ(street segment lengths) / cell_area (km/km²). This is perhaps the simplest and most widely used street network descriptor in the literature and is entirely absent. It can be computed from the existing highways data trivially.
- **Intersection density** — number of network nodes per cell area. Closely related to street density but captures connectivity rather than coverage.

---

## 4. Critical Missing Metrics

### 4.1 Building Intensity / Density Metrics (HIGH PRIORITY)

These metrics require only data already available (building footprints + heights + cell area) and are foundational to urban morphology:

**Ground Space Index (GSI)**
- Formula: `Σ building_footprint_area / cell_area`
- Also known as: building coverage ratio, site coverage, lot coverage
- Interpretation: 0.05 = very sparse; 0.20 = suburban; 0.50 = dense urban; 0.80 = historic core
- Why important: Single strongest predictor of urban form type

**Floor Space Index (FSI) / Floor Area Ratio (FAR)**
- Formula: `Σ gross_floor_area / cell_area`
- `gross_floor_area = footprint_area × building_levels` (or `footprint_area × height / 3`)
- Interpretation: 0.3 = low-density suburban; 1.5 = medium-density; 4+ = dense urban core
- Why important: Primary planning intensity metric worldwide; captures building volume regardless of height distribution

**Open Space Ratio (OSR)**
- Formula: `(1 - GSI) / FSI`
- Interpretation: high OSR = generous open space per unit floor area (suburban); low OSR = compact/intense development
- Why important: Captures the quality/generosity of open space; part of the spacemate typology framework

**Average Building Height (cell-level)**
- Formula: `Σ(building_area × height) / Σ building_area` (area-weighted mean height)
- Also: simple mean height, height standard deviation, coefficient of variation
- Why important: Height statistics are completely absent from the current output. The `volume` metric aggregates height with footprint area and cannot be disentangled by the neural network.

**Building Count and Building Density**
- Formula: `count(buildings_in_cell)` and `count / cell_area`
- Why important: Most basic count metric; cells with 0 buildings are indistinguishable from cells with 1 building in the current statistics

### 4.2 Land Use Composition (HIGH PRIORITY)

Already flagged as "coming soon" in the project description. The OSM `landuse` layer is loaded but not yet processed. Key land use categories to compute coverage proportions for:

- `residential` — general residential land
- `commercial` — commercial / retail zones
- `industrial` — factories, warehouses, logistics
- `retail` — specifically retail (separate from commercial in OSM)
- `recreation_ground`, `grass`, `meadow`, `forest`, `nature_reserve` — green/open
- `farmland`, `farmyard` — agricultural
- `cemetery`, `religious` — institutional/special
- `parking`, `garages` — transport infrastructure
- `education`, `institutional` — public facilities

For each category: area as a proportion of cell area (0–1). Additionally, a Shannon entropy / Simpson diversity index over all landuse types would capture how mixed-use vs. single-use a cell is.

Note: OSM landuse polygons are inconsistently tagged and often incomplete, but coverage proportions will still provide signal even with partial data.

### 4.3 Height Statistics (MEDIUM PRIORITY)

Currently, building height is used as an input to `volume`, `floor_area`, and `form_factor`, but the distribution of heights itself is not captured as a standalone output. Consider adding:

- `height_mean` — area-weighted mean building height across the cell
- `height_std` — standard deviation of building heights
- `height_cv` — coefficient of variation (std / mean), dimensionless normalised spread
- `height_entropy` — Shannon entropy of height decile bins; captures whether heights are clustered around one value or spread across many
- `height_range` — max - min height; captures the vertical scale contrast in the cell

These are particularly important for distinguishing:
- Uniform low-rise residential (low mean, low std)
- Mixed old-city core (variable mean, high std)
- High-rise cluster (high mean, low std)
- Transitional zone with one skyscraper (high range, high entropy)

### 4.4 Street Network Physical Descriptors (MEDIUM PRIORITY)

**Street density** (one of the most widely used urban form metrics):
- Total vehicle street length / cell area (km/km²)
- Total pedestrian street length / cell area
- Ratio of pedestrian to vehicle network

**Intersection density:**
- Count of network nodes with degree ≥ 3 (true intersections, not dead ends) / cell area
- Count of all network nodes / cell area

**Average segment length:**
- Mean and standard deviation of individual street segment lengths (different from `mean_node_dist` which is per-node average of edge lengths)

**Street type composition:**
- Proportion of network length by road type (motorway/trunk, primary, secondary, tertiary, residential, unclassified)
- Presence of cycle infrastructure (cycleway tags)

These physical descriptors are computationally trivial (length summation + area division) and provide essential context for interpreting the topological metrics already computed.

### 4.5 Morphological Tessellation Area Statistics (MEDIUM PRIORITY)

The pipeline builds a morphological tessellation (Voronoi cells) for the neighbourhood context, using it for `neighbors` and `cell_alignment`. However, the area of each tessellation cell — which represents the approximate "territory" or proximate space controlled by each building — is never aggregated as a metric.

Tessellation cell area statistics would add:
- `tessellation_area_mean/median/std` — typical plot size proxy
- `tessellation_area_cv` — heterogeneity of plot sizes (high in mixed residential/industrial; low in homogeneous subdivisions)
- Compactness of tessellation cells

This is low-effort to add since the tessellation is already computed.

### 4.6 Building Use Diversity (LOW-MEDIUM PRIORITY)

OSM's `building=*` tag encodes the type or use of buildings:
- Residential types: `house`, `detached`, `semidetached_house`, `apartments`, `terrace`, `bungalow`, `dormitory`
- Non-residential: `commercial`, `retail`, `office`, `industrial`, `warehouse`, `hospital`, `school`, `church`, `civic`, `public`
- Infrastructure: `garage`, `garages`, `shed`, `greenhouse`, `barn`, `roof`

Computing the Shannon entropy of building type values within each cell (excluding the generic `yes`) would capture land use mix at the building level, which is more granular than OSM landuse polygons and covers areas where landuse polygons are absent.

Additionally, the proportion of non-residential buildings (commercial/office/industrial/retail count / total count) would be a useful ground-floor activation proxy.

### 4.7 Reached / Network Accessibility (LOW PRIORITY for embeddings, MEDIUM for research)

`momepy.reached()` computes how much area, how many nodes, or how many buildings can be reached from each network node within a given distance (network distance, not Euclidean). This is fundamentally different from centrality metrics:

- Centrality asks: "how important is this node in the global network?"
- Reached asks: "how many destinations are accessible from here?"

For H3 cells at typical resolutions (resolution 8 = ~0.7 km² cells), "reached within 800m pedestrian walk" would capture the walkability potential of the cell. This is analogous to a simplified Walk Score computation.

---

## 5. Quality and Methodological Issues

### 5.1 CRS Selection for Shape Metrics

**Current approach**: EPSG:3857 (Web Mercator) for conformal metrics.

**Issue**: Web Mercator distorts scale by a factor of 1/cos(φ) where φ is latitude. At 40°N (Rome, Madrid, New York), scale distortion is ~30%. At 55°N (London, Copenhagen), it is ~75%. At 60°N (Helsinki, Oslo, Saint Petersburg), it is ~100%.

Shape ratios (like `elongation = shorter_side / longer_side`) are dimensionless and therefore preserved under conformal projection regardless of scale factor. However, **absolute lengths** like `facade_ratio = area / perimeter` are affected because area scales as the square of the scale factor while perimeter scales linearly. This means `facade_ratio` values will be numerically different in high-latitude vs. low-latitude cities even for geometrically identical buildings.

**Recommendation**: Document this limitation clearly. For research applications that will span wide latitude ranges, consider using UTM zones (via pyproj's `CRS.from_authority("auto:42001", ...)` pattern or osmnx's projected CRS approach) for conformal metrics.

### 5.2 Height Estimation Bias

**Current approach**: height tag → building_levels × 3m → 6m default.

**Known biases**:
- **Under-coverage**: OSM height data is sparse (typically 5–20% of buildings have explicit height tags, and `building:levels` coverage is higher but still incomplete in many regions).
- **Default bias**: The 6m default applies to all buildings without any height information. This lumps residential sheds, garden walls, and high-rise towers all at 6m when the level tag is absent.
- **Level conversion**: The 3m/storey factor is a common convention but varies: commercial floors are typically 3.5–4m, residential 2.7–3.0m. First floor retail in mixed-use buildings can be 5–6m.
- **Regional variation**: The 6m default (≈ 2 storeys) is reasonable for residential in many countries, but in dense city centres with predominantly 4–6 storey perimeter block buildings, this systematically underestimates height.

**Impact on metrics**: `volume`, `floor_area`, `form_factor`, and `street_profile.hw_ratio` all depend directly on estimated height and will reflect these biases. For training neural network embeddings, this systematic bias is less problematic than random noise (the model can learn to adjust), but for interpretability of learned representations, it is worth noting.

**Potential improvements**:
- Use building type (from `building=` tag) to set type-specific defaults: `house/detached/semidetached` → 6m; `apartments` → 9m; `commercial/retail` → 6m; `office` → 12m; `industrial/warehouse` → 8m.
- Note: This adds complexity and may introduce new biases in regions where tagging conventions differ.

### 5.3 Building Adjacency Under-Detection

**Current approach**: Rook contiguity graph using `libpysal.graph.Graph.build_contiguity()`.

**Issue**: Rook contiguity requires geometries to share at least a line segment (wall). OSM buildings frequently have small coordinate gaps (0.01–2m) between party-wall buildings because neighbouring buildings were digitised independently. These gaps prevent rook contiguity detection.

**Impact**: `shared_walls`, `building_adjacency`, and `courtyards` will systematically under-report adjacency in OSM data. A pre-processing step of snapping nearby vertices (within a tolerance of 0.5–1m) before computing contiguity would improve accuracy but risks merging genuinely separate buildings.

**Alternative**: Buffer building footprints by a small amount (0.25–0.5m) before computing contiguity to capture "near-touching" buildings as adjacent. `momepy.buffered_limit` uses a similar concept for the tessellation clip boundary.

### 5.4 Fixed Neighbourhood Distance

**Current approach**: Single `neighbourhood_distance` parameter (default 500m) applied uniformly.

**Issue**: H3 hexagons vary in area by resolution. At resolution 8 (~0.7 km² per cell, ~0.5 km across), a 500m neighbourhood buffer is appropriate. At resolution 7 (~5.2 km², ~1.3 km across), a 500m buffer is barely larger than the cell itself, providing almost no neighbourhood context for edge buildings.

**Recommendation**: Make the pipeline aware of H3 resolution if possible, or at minimum document that the default 500m is tuned for resolution 8 and should be increased for coarser resolutions. Alternatively, express the neighbourhood as a multiplier of cell radius rather than an absolute distance.

### 5.5 Network Metric Stability

**Betweenness centrality** is computed over the full focal+neighbourhood subgraph. For a 500m neighbourhood around a typical urban H3 cell at resolution 8, this subgraph may contain 50–500 nodes. Betweenness centrality values computed on truncated subgraphs are not comparable to global betweenness values computed over entire city networks — they reflect only local hierarchy, which is actually appropriate here. This should be clearly documented.

**Closeness centrality** formula in momepy is `1 / mean_shortest_path_distance`, which produces values on the scale of 1/distance (units: m⁻¹ for weighted graphs or dimensionless for hop-count). When aggregating over focal nodes (mean/median of closeness values), the resulting number is interpretable but depends on the size of the subgraph. Larger subgraphs will produce smaller mean closeness values simply due to having more nodes to average over. This is another reason the neighbourhood distance choice matters for comparability.

### 5.6 Street Profile Tick Length

**Current default**: `tick_length=50m`.

This means the perpendicular ticks extend 25m on each side of the street centreline. For narrow streets (5–10m carriageway), this is more than adequate to reach flanking buildings. However, for wide boulevards or highways with service roads, 25m may not reach the building line.

In very dense areas (e.g., Tokyo backstreets, historic medinas), buildings on opposite sides of a 3m alley may be reached by far shorter ticks. The tick length is effectively a sensitivity parameter: shorter = only detects immediate flanking buildings; longer = may pick up buildings across secondary streets.

This is not a bug but a methodological choice. Document it clearly.

### 5.7 Isolated Cells

**Current behaviour**: Cells with zero buildings return all-NaN for building metrics. Cells with no streets return all-NaN for connectivity metrics.

**Issue for neural network training**: All-NaN rows may be handled as missing data by the neural network, but they carry semantic information (the cell is empty / unbuilt). A set of indicator variables would disambiguate:
- `has_buildings` (boolean)
- `has_vehicle_streets` (boolean)
- `has_pedestrian_streets` (boolean)
- `building_count` (integer, may be 0)

These indicators allow the network to distinguish "empty farmland" from "missing data".

---

## 6. momepy Functions Not Yet Used

The following momepy functions are available but not currently used in the pipeline:

| Function | What it provides | Priority |
|----------|-----------------|----------|
| `momepy.covered_area_ratio()` | GSI (building coverage ratio) using tessellation cells | HIGH |
| `momepy.floor_area_ratio()` | FSI using tessellation cells | HIGH |
| `momepy.COINS()` | Continuity In Street Networks — grouping streets into natural strokes | MEDIUM |
| `momepy.reached()` | Network accessibility: nodes/area reachable within distance | MEDIUM |
| `momepy.describe()` | Statistical description of a variable within a spatial graph neighbourhood | MEDIUM |
| `momepy.weighted_character()` | Area-weighted aggregation of building values within tessellation | LOW |
| `momepy.blocks()` or `Blocks` class | Block polygon extraction from street network | LOW |
| `momepy.street_length()` | Per-segment street lengths | LOW |
| `momepy.cds_length()` (per segment) | Per-segment cul-de-sac length | LOW |

---

## 7. Comparison Against momepy's Full Metric Catalogue

The table below maps every momepy metric against implementation status:

### Dimension
| momepy function | Implemented | Notes |
|----------------|-------------|-------|
| `longest_axis_length` | ✅ | |
| `perimeter_wall` | ✅ | |
| `courtyard_area` | ✅ | |
| `covered_area_ratio` (GSI) | ❌ | Critical gap |
| `floor_area_ratio` (FSI) | ❌ | Critical gap |
| `street_length` | ❌ | Not yet |

### Shape
| momepy function | Implemented | Notes |
|----------------|-------------|-------|
| `fractal_dimension` | ✅ | Raw + dissolved |
| `circular_compactness` | ✅ | |
| `square_compactness` | ✅ | |
| `convexity` | ✅ | |
| `courtyard_index` | ✅ | |
| `rectangularity` | ✅ | |
| `shape_index` | ✅ | |
| `corners` | ✅ | |
| `squareness` | ✅ | |
| `equivalent_rectangular_index` | ✅ | |
| `elongation` | ✅ | |
| `facade_ratio` | ✅ | Raw + dissolved |
| `form_factor` | ✅ | |
| `compactness_weighted_axis` | ✅ | Raw + dissolved |
| `centroid_corner_distance` | ✅ | Raw + dissolved |

### Spatial Distribution
| momepy function | Implemented | Notes |
|----------------|-------------|-------|
| `orientation` | ✅ | |
| `shared_walls` | ✅ | Including ratio |
| `alignment` | ✅ | KNN-15 |
| `neighbor_distance` | ✅ | |
| `mean_interbuilding_distance` | ✅ | |
| `building_adjacency` | ✅ | |
| `neighbors` | ✅ | Queen tessellation |
| `cell_alignment` | ✅ | |
| `street_alignment` | ✅ | |

### Intensity
| momepy function | Implemented | Notes |
|----------------|-------------|-------|
| `courtyards` | ✅ | |
| `covered_area_ratio` | ❌ | GSI — critical |
| `floor_area_ratio` | ❌ | FSI — critical |

### Street Network (per node)
| momepy function | Implemented | Notes |
|----------------|-------------|-------|
| `node_degree` | ✅ | Both graphs |
| `meshedness` | ✅ | Both graphs |
| `mean_node_dist` | ✅ | Both graphs |
| `mean_node_degree` | ✅ | Both graphs |
| `gamma` | ✅ | Both graphs |
| `edge_node_ratio` | ✅ | Both graphs |
| `cyclomatic` | ✅ | Both graphs |
| `clustering` | ✅ | Both graphs |
| `closeness_centrality` | ✅ | Both graphs |
| `betweenness_centrality` | ✅ | Both graphs |
| `straightness_centrality` | ✅ | Both graphs |
| `reached` | ❌ | Accessibility metric |

### Street Network (graph-level)
| momepy function | Implemented | Notes |
|----------------|-------------|-------|
| `meshedness` (global) | ✅ | Both graphs |
| `gamma` (global) | ✅ | Both graphs |
| `cds_length` (total) | ✅ | Both graphs |
| `proportion` (3/4/dead) | ✅ | Both graphs |
| `COINS` | ❌ | Street continuity |

### Street Profile
| momepy function | Implemented | Notes |
|----------------|-------------|-------|
| `street_profile` (width, openness, hw_ratio) | ✅ | Vehicle streets only |

---

## 8. Recommendations by Priority

### Priority 1: Critical (implement before publishing results)

1. **GSI (Ground Space Index)**: `Σ building_footprint_area / cell_area`. Single scalar per cell. Requires only `buildings_ea` and cell area in equal-area CRS.

2. **FSI (Floor Space Index / FAR)**: `Σ gross_floor_area / cell_area`. Single scalar per cell. Requires `buildings_with_height` and cell area.

3. **OSR (Open Space Ratio)**: `(1 - GSI) / FSI`. Derived from the above.

4. **Land use coverage proportions**: For each major OSM landuse category, compute `landuse_area / cell_area`. `landuse_ea` is already loaded and cached. This needs intersection/clipping with the cell geometry and per-category area summation.

5. **Building height statistics**: `height_mean`, `height_std`, `height_cv` as cell-level scalars, plus aggregation statistics. These are trivially computed from the `height` column in `buildings_with_height`.

### Priority 2: High (implement for richer profiles)

6. **Street density**: `Σ street_length / cell_area` (km/km²) for vehicle and pedestrian networks separately.

7. **Intersection density**: `node_count_with_degree_≥3 / cell_area`. Vehicle and pedestrian separately.

8. **Building count and building density**: `len(buildings_ea)` and `/ cell_area`.

9. **Tessellation area statistics**: `mean`, `std`, `cv` of morphological tessellation cell areas — already computed during tessellation creation, just not aggregated.

10. **Building area statistics**: `mean`, `std`, `cv` of individual building footprint areas. Building size distribution is one of the most informative building-level metrics.

### Priority 3: Medium (implement to improve depth)

11. **Height entropy**: Shannon entropy of building height deciles within the cell. Captures vertical heterogeneity not captured by mean/std alone.

12. **Building use type diversity**: Shannon entropy of `building=*` tag values (excluding `yes`). Requires no new OSM data loading.

13. **Street profile on pedestrian streets**: Run `momepy.street_profile` on pedestrian highways in addition to vehicle highways.

14. **`proportion_five_plus`**: Proportion of intersections with ≥5 ways (common in organic street networks).

15. **COINS stroke metrics**: Street continuity analysis — mean stroke length, stroke count density. Requires `momepy.COINS()`.

### Priority 4: Consider for future iterations

16. **`momepy.reached()`**: Accessibility metric — nodes/area reachable within 400m and 800m network distance.

17. **Multiscale network centrality**: Compute closeness and betweenness at multiple radii (400m, 800m, 1600m) to capture walkability at different scales.

18. **Green space ratio**: OSM `leisure=park`, `natural=wood`, etc. coverage within cell (can use `landuse_ea` layer if it includes these tags, otherwise requires additional OSM query).

19. **Building footprint area distribution**: Not just mean/std but full quantile profile (already done for per-building shape metrics; consider adding raw area).

20. **Presence indicators**: `has_buildings`, `has_vehicle_streets`, `has_pedestrian_streets` boolean flags to disambiguate truly empty cells from data gaps.

---

## 9. Structural Observations

### 9.1 The Three-CRS Strategy

Using different CRS per metric family (equal-area for area calculations, equidistant for distances, conformal for shapes) is theoretically sound and follows best practice. However, the conformal CRS (EPSG:3857) has latitude-dependent scale distortion that affects metrics that mix area and length (e.g., `facade_ratio`). This is documented above but worth flagging in any published methodology.

### 9.2 Statistical Aggregation

Computing mean, median, std, and evenly-spaced quantiles for per-building metrics is a good strategy. The quantile distribution captures non-normality and multimodality (e.g., a cell with two distinct building types will show a bimodal height distribution that mean/std alone cannot describe). For the neural network application, the quantile distribution functions essentially as a non-parametric histogram.

One potential refinement: the current `aggregate_series` uses evenly-spaced quantiles (q10, q20, ..., q100). Percentile statistics at conventional points (p5, p25, p75, p95) might be more interpretable and contain similar information in fewer columns. However, for neural network training, the current approach is fine — the model will discover which quantiles carry discriminative information.

### 9.3 Neighbourhood Graph Builds

The shared spatial graphs (KNN, Delaunay, contiguity, tessellation) computed on focal+neighbourhood buildings and cached on the CellContext are a clean design. One possible issue: if the neighbourhood buffer is large enough that some cells' neighbourhoods overlap with many other cells' neighbourhoods, the same buildings participate in many different graph computations. This is expected and correct, but it means graph-dependent metrics for a building near the cell boundary are estimated from slightly different neighbourhood contexts in each cell — a "fuzzy boundary" effect that is not eliminable without a true parcel-based analysis.

### 9.4 Cache Design

The Parquet-backed caching per `region_id` is robust and allows pipeline resumption. One potential improvement: the current cache invalidation strategy is presence/absence of the file. If the OSM source data changes (new PBF download) or metric configuration changes (different `knn_k`), the cache is stale but will be used. Consider storing a metadata/config hash alongside the cache files to detect staleness.

---

## 10. Summary Table of Gaps

| Gap | Category | Priority | Difficulty |
|-----|----------|----------|-----------|
| GSI (building coverage ratio) | Intensity | Critical | Low |
| FSI / FAR (floor area ratio) | Intensity | Critical | Low |
| OSR (open space ratio) | Intensity | Critical | Trivial (derived) |
| Land use coverage proportions | Land use | Critical | Medium |
| Building height statistics (mean, std, cv) | Dimension | High | Low |
| Street density (length/area) | Connectivity | High | Low |
| Intersection density | Connectivity | High | Low |
| Building count & density | Intensity | High | Trivial |
| Tessellation area statistics | Distribution | High | Low |
| Building footprint area statistics | Dimension | High | Low |
| Height entropy | Dimension | Medium | Low |
| Building use type diversity | Diversity | Medium | Low |
| Street profile on pedestrian streets | Street | Medium | Low |
| Proportion of 5+-way intersections | Connectivity | Medium | Low |
| COINS stroke length metrics | Connectivity | Medium | Medium |
| Reached (network accessibility) | Accessibility | Medium | Medium |
| Multiscale network centrality | Connectivity | Low | Medium |
| Green space ratio (from leisure/natural tags) | Land use | Low | Medium |
| Presence indicators (has_buildings, etc.) | Metadata | Low | Trivial |
| Cache staleness detection | Infrastructure | Low | Low |

---

## References

- Berghauser Pont, M. & Haupt, P. (2010). *Spacematrix: Space, Density and Urban Form*. NAi Publishers.
- Caniggia, G. & Maffei, G.L. (1979). *Composizione architettonica e tipologia edilizia*. Marsilio.
- Conzen, M.R.G. (1960). Alnwick, Northumberland: A Study in Town-Plan Analysis. *Transactions of the Institute of British Geographers*, 27, 1–122.
- Dibble, J., et al. (2017). Urban Morphometrics: Towards a Science of Urban Form. *Urban Morphology*, 21(1), 55–67.
- Fleischmann, M., et al. (2019). momepy: Urban Morphology Measuring Toolkit. *Journal of Open Source Software*, 4(43), 1807.
- Fleischmann, M., et al. (2021). Methodological Foundation of a Numerical Taxonomy of Urban Form. *Environment and Planning B: Urban Analytics and City Science*, 48(1), 45–66.
- Gil, J., et al. (2012). On the Discovery of Urban Typologies: Data Mining the Multi-dimensional Character of Neighbourhoods. *Urban Morphology*, 16(1), 27–40.
- Hillier, B. & Hanson, J. (1984). *The Social Logic of Space*. Cambridge University Press.
- Kropf, K. (2014). Ambiguity in the Definition of Built Form. *Urban Morphology*, 18(1), 41–57.
- Moudon, A.V. (1997). Urban Morphology as an Emerging Interdisciplinary Field. *Urban Morphology*, 1(1), 3–10.
- Porta, S., et al. (2006). The Network Analysis of Urban Streets: A Primal Approach. *Environment and Planning B*, 33(5), 705–725.

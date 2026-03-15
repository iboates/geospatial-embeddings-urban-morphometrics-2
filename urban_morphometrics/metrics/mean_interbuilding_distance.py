"""Mean interbuilding distance metric.

Mean distance between each building and all buildings within its combined
Delaunay + KNN neighbourhood. A broader spacing measure than neighbor_distance:
where neighbor_distance captures immediate adjacency, this captures the typical
separation across the whole local building cluster.

Requires neighbourhood context for the same reason as neighbor_distance and alignment.

Implementation note: momepy.mean_interbuilding_distance internally builds two sparse
matrices — one from the Delaunay adjacency via pandas to_coo(sort_labels=True), the
other from the KNN graph's sparse property — and indexes one with the other positionally.
This only works when both matrices share the same 0-based sequential positional ordering.
Because focal_plus_neighbourhood_buildings is indexed by OSM IDs (not sequential), we
reset to a RangeIndex before building the graphs and calling momepy, then map results
back to focal buildings via a positional mask.
"""

import pandas as pd
import momepy
from libpysal.graph import Graph

from urban_morphometrics.cell_context import CellContext
from urban_morphometrics.metrics import register
from urban_morphometrics.metrics.aggregation import aggregate_series


@register("mean_interbuilding_distance")
def compute(ctx: CellContext, num_quantiles: int) -> dict:
    """Mean distance from each building to all neighbours in its Delaunay+KNN cluster (m)."""
    b = ctx.buildings_ea
    if b.empty:
        return aggregate_series(pd.Series(dtype=float), "mean_interbuilding_distance", num_quantiles)

    all_b = ctx.focal_plus_neighbourhood_buildings
    if len(all_b) < 3:
        return aggregate_series(pd.Series(dtype=float), "mean_interbuilding_distance", num_quantiles)

    # momepy.mean_interbuilding_distance internally builds two sparse matrices — one
    # from the Delaunay adjacency via to_coo(sort_labels=True), the other from the
    # KNN sparse property — and indexes one with the other positionally. This only
    # works when both matrices share the same 0-based sequential ordering AND have
    # the same number of rows (i.e. both graphs contain all N buildings as nodes).
    #
    # Two issues arise with OSM data:
    # 1. Non-sequential OSM IDs cause different positional orderings in the two
    #    sparse matrices → fixed by resetting to a 0..N-1 RangeIndex.
    # 2. Near-coincident building centroids can cause Delaunay to silently drop nodes
    #    even with coplanar='jitter', making the distance matrix smaller than the KNN
    #    matrix → fixed by probing which nodes Delaunay retains, filtering to those,
    #    and re-building both graphs on the filtered 0..M-1 set.

    # Step 1: reset to sequential index
    all_b_pos = all_b.reset_index(drop=True)

    # Step 2: probe Delaunay to discover which nodes survive triangulation.
    # coplanar='clique' is more robust than 'jitter' for near-coincident centroids.
    delaunay_probe = Graph.build_triangulation(
        all_b_pos.centroid, method="delaunay", coplanar="clique"
    )
    del_positions = sorted(
        set(delaunay_probe._adjacency.index.get_level_values(0))
        | set(delaunay_probe._adjacency.index.get_level_values(1))
    )

    # Step 3: filter to surviving nodes and re-index to fresh 0..M-1
    surviving = all_b_pos.iloc[del_positions].reset_index(drop=True)
    if len(surviving) < 3:
        return aggregate_series(pd.Series(dtype=float), "mean_interbuilding_distance", num_quantiles)

    # Step 4: build final graphs on the clean sequential index
    k = min(ctx.config.knn_k, len(surviving) - 1)
    delaunay = Graph.build_triangulation(surviving.centroid, method="delaunay", coplanar="clique")
    knn = Graph.build_knn(surviving.centroid, k=k, coplanar="jitter")

    values_m = momepy.mean_interbuilding_distance(surviving, delaunay, knn)

    # Step 5: map positional results (0..M-1) back to original OSM IDs
    osm_ids_surviving = all_b.index[del_positions]
    values_mapped = pd.Series(values_m.values, index=osm_ids_surviving)
    return aggregate_series(values_mapped.reindex(b.index), "mean_interbuilding_distance", num_quantiles)

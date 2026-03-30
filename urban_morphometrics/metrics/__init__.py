"""Metric registry and computation entry point.

Each metric is a callable registered in REGISTRY under its name string.
A metric callable has the signature:

    def compute(ctx: CellContext, num_quantiles: int) -> dict[str, float]:
        ...

It receives the CellContext for the current cell and returns a flat dict of
{column_name: value} pairs ready to become columns in the results GeoDataFrame.
"""

import logging
from pathlib import Path
from typing import Callable

from urban_morphometrics.cell_context import CellContext

log = logging.getLogger(__name__)

# Maps metric name → compute function.
# Populated by individual metric modules via register().
REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a metric compute function under the given name."""
    def decorator(fn: Callable) -> Callable:
        REGISTRY[name] = fn
        return fn
    return decorator


# Import all metric modules to trigger @register decorators.
# Keep this list in alphabetical order; add new modules here as they are implemented.
def _load_metrics():
    from urban_morphometrics.metrics import (  # noqa: F401
        # Dimension metrics (step 5)
        courtyard_area,
        floor_area,
        longest_axis_length,
        perimeter_wall,
        volume,
        # Shape metrics (step 6)
        centroid_corner_distance,
        circular_compactness,
        compactness_weighted_axis,
        convexity,
        corners,
        courtyard_index,
        elongation,
        equivalent_rectangular_index,
        facade_ratio,
        form_factor,
        fractal_dimension,
        rectangularity,
        shape_index,
        square_compactness,
        squareness,
        # Distribution metrics (step 7)
        alignment,
        building_adjacency,
        cell_alignment,
        mean_interbuilding_distance,
        neighbor_distance,
        neighbors,
        orientation,
        shared_walls,
        street_alignment,
        # Intensity and street relationship metrics (step 8)
        courtyards,
        nearest_street_distance,
        street_profile,
        # Street connectivity — per-node metrics (step 10)
        betweenness_centrality,
        closeness_centrality,
        clustering,
        cyclomatic,
        degree,
        edge_node_ratio,
        gamma,
        mean_node_degree,
        mean_node_dist,
        meshedness,
        straightness_centrality,
        # Street connectivity — global graph-level metrics (step 11)
        cds_length_total,
        gamma_global,
        meshedness_global,
        proportion,
        # Land cover metrics (step 13)
        landuse_cover,
        water_cover,
        pedestrian_area_cover,
    )

_load_metrics()


def compute_metrics(
    ctx: CellContext,
    metric_names: list[str],
    num_quantiles: int,
    features_dir: Path | None = None,
) -> dict:
    """Compute all requested metrics for a single cell.

    Args:
        ctx: CellContext for the cell being processed.
        metric_names: List of metric names to compute, or ["all"] for every
            registered metric.
        num_quantiles: Passed through to each metric's aggregation step.
        features_dir: When provided, each metric writes a GeoPackage of its
            per-feature computed values to this directory.

    Returns:
        Flat dict of {column_name: value} for all computed metrics combined.
    """
    if metric_names == ["all"]:
        names = list(REGISTRY.keys())
    else:
        unknown = [n for n in metric_names if n not in REGISTRY]
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(unknown)}. Available: {', '.join(REGISTRY)}")
        names = metric_names

    row = {}
    for name in names:
        try:
            result = REGISTRY[name](ctx, num_quantiles, features_dir=features_dir)
            row.update(result)
        except Exception:
            log.warning("Metric '%s' failed for region %s", name, ctx.region_id, exc_info=True)
    return row

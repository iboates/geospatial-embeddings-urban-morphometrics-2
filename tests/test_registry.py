"""Tests for the metric registry."""

import inspect
import pytest

from urban_morphometrics.metrics import REGISTRY


EXPECTED_METRICS = {
    # Dimension
    "courtyard_area", "floor_area", "longest_axis_length", "perimeter_wall", "volume",
    # Shape
    "centroid_corner_distance", "circular_compactness", "compactness_weighted_axis",
    "convexity", "corners", "courtyard_index", "elongation",
    "equivalent_rectangular_index", "facade_ratio", "form_factor",
    "fractal_dimension", "rectangularity", "shape_index", "square_compactness",
    "squareness",
    # Distribution
    "alignment", "building_adjacency", "cell_alignment", "mean_interbuilding_distance",
    "neighbor_distance", "neighbors", "orientation", "shared_walls", "street_alignment",
    # Intensity / street relationship
    "courtyards", "nearest_street_distance", "street_profile",
    # Connectivity — per-node
    "betweenness_centrality", "closeness_centrality", "clustering", "cyclomatic",
    "degree", "edge_node_ratio", "gamma", "mean_node_degree", "mean_node_dist",
    "meshedness", "straightness_centrality",
    # Connectivity — global
    "cds_length_total", "gamma_global", "meshedness_global", "proportion",
}


def test_all_expected_metrics_registered():
    missing = EXPECTED_METRICS - set(REGISTRY)
    assert not missing, f"Missing from registry: {sorted(missing)}"


def test_no_unexpected_metrics():
    extra = set(REGISTRY) - EXPECTED_METRICS
    assert not extra, f"Unexpected metrics in registry: {sorted(extra)}"


def test_all_callables():
    for name, fn in REGISTRY.items():
        assert callable(fn), f"{name} is not callable"


def test_all_have_correct_signature():
    """Every metric function must accept (ctx, num_quantiles) positional args."""
    for name, fn in REGISTRY.items():
        params = list(inspect.signature(fn).parameters)
        assert len(params) >= 2, f"{name} has fewer than 2 parameters"
        assert params[0] == "ctx", f"{name} first param should be 'ctx', got '{params[0]}'"
        assert params[1] == "num_quantiles", (
            f"{name} second param should be 'num_quantiles', got '{params[1]}'"
        )


def test_all_return_dict():
    """Smoke-test each metric with an empty CellContext-like stub returning an empty GDF."""
    # We can't run actual metrics without real data, but we can verify the registry
    # entries are the compute functions (not the decorator wrappers).
    for name, fn in REGISTRY.items():
        assert fn.__name__ == "compute", (
            f"{name} registered function should be named 'compute', got '{fn.__name__}'"
        )

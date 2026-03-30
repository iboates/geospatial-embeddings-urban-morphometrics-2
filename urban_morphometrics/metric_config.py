"""Configuration for metric computation parameters."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class MetricConfig:
    """Metric computation configuration.

    All fields have sensible defaults so MetricConfig() works with no arguments.
    Pass an instance to compute_urban_morphometrics() via metric_config=, or load
    from a JSON file on the command line via --metric-config.

    Fields
    ------
    knn_k:
        Number of nearest neighbours for KNN spatial graphs. Used by the
        alignment, building_adjacency and mean_interbuilding_distance metrics.
        Default: 15.
    tessellation_buffer:
        Buffer around buildings used to clip the Voronoi tessellation. Pass a
        float for a fixed buffer in metres, or "adaptive" to let momepy choose
        the buffer per building based on the half-maximum Gabriel-graph neighbour
        distance. "adaptive" is recommended: it shrinks the clip in dense areas
        (avoiding runaway cells) and expands it in sparse areas.
        Default: "adaptive".
    tessellation_min_buffer:
        Minimum buffer distance (metres) enforced when tessellation_buffer is
        "adaptive". Has no effect for a fixed buffer. Keeps tessellation cells
        from collapsing to nothing in very dense clusters.
        Default: 0.
    tessellation_max_buffer:
        Maximum buffer distance (metres) enforced when tessellation_buffer is
        "adaptive". Has no effect for a fixed buffer. Prevents tessellation cells
        from extending unreasonably far in sparse areas.
        Default: 100.
    tessellation_shrink:
        Negative buffer distance (metres) applied to each building before Voronoi
        to create a small gap between adjacent cells. Default: 0.4.
    tessellation_segment:
        Maximum distance between points after polygon discretisation (metres).
        Smaller values increase tessellation precision at the cost of speed.
        Default: 0.5.
    street_alignment_max_distance:
        Maximum search radius (metres) when matching buildings to their nearest
        street for the street_alignment metric. Buildings with no street within
        this radius get a NaN alignment value. Set to null in JSON for no limit.
        Default: 500.0.
    network_subgraph_radius:
        Hop radius used when computing per-node subgraph metrics on the street
        network: meshedness, gamma, edge_node_ratio, cyclomatic, and
        mean_node_degree. Each node's metric is computed within its local
        subgraph of this many hops. Larger values capture wider context but
        increase computation time roughly quadratically.
        Default: 5.
    street_profile_distance:
        Spacing (metres) between the perpendicular ticks cast along each street
        segment when computing the street_profile metric. Smaller values give
        a denser, more precise profile at the cost of speed.
        Default: 10.
    street_profile_tick_length:
        Length (metres) of each perpendicular tick cast from the street
        centreline when computing the street_profile metric. Should be at
        least as wide as the widest street canyon expected; ticks that reach
        no building on one side count as open.
        Default: 50.
    graph_endpoint_snap_tolerance:
        Maximum distance (metres) between a street endpoint and another street
        segment for a topological split to be triggered. When an endpoint falls
        within this radius of another segment's interior (and is not already a
        shared endpoint), that segment is split at the projected point so the
        graph correctly represents the T-junction. Set to 0 to disable splitting.
        Default: 0.1.
    """

    knn_k: int = 15
    tessellation_buffer: float | str = "adaptive"
    tessellation_min_buffer: float = 0.0
    tessellation_max_buffer: float = 100.0
    tessellation_shrink: float = 0.4
    tessellation_segment: float = 0.5
    street_alignment_max_distance: float | None = 500.0
    network_subgraph_radius: int = 5
    street_profile_distance: float = 10.0
    street_profile_tick_length: float = 50.0
    graph_endpoint_snap_tolerance: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "MetricConfig":
        """Create a MetricConfig from a plain dict (e.g. loaded from JSON).

        Unknown keys raise ValueError to catch typos early.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"Unknown MetricConfig keys: {', '.join(sorted(unknown))}. "
                f"Valid keys: {', '.join(sorted(known))}"
            )
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> "MetricConfig":
        """Load a MetricConfig from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        """Serialise to a plain dict (suitable for JSON serialisation)."""
        return asdict(self)

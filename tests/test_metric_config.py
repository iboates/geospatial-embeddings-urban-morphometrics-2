"""Tests for MetricConfig loading and validation."""

import json
import pytest
from pathlib import Path

from urban_morphometrics.metric_config import MetricConfig


def test_defaults():
    cfg = MetricConfig()
    assert cfg.knn_k == 15
    assert cfg.tessellation_buffer == "adaptive"
    assert cfg.tessellation_min_buffer == 0.0
    assert cfg.tessellation_max_buffer == 100.0
    assert cfg.tessellation_shrink == 0.4
    assert cfg.tessellation_segment == 0.5
    assert cfg.street_alignment_max_distance == 500.0
    assert cfg.network_subgraph_radius == 5
    assert cfg.street_profile_distance == 10.0
    assert cfg.street_profile_tick_length == 50.0


def test_from_dict_partial():
    cfg = MetricConfig.from_dict({"knn_k": 20, "network_subgraph_radius": 3})
    assert cfg.knn_k == 20
    assert cfg.network_subgraph_radius == 3
    assert cfg.tessellation_buffer == "adaptive"  # default preserved


def test_from_dict_unknown_key():
    with pytest.raises(ValueError, match="Unknown MetricConfig keys"):
        MetricConfig.from_dict({"knn_k": 5, "typo_key": 99})


def test_from_dict_null_distance():
    cfg = MetricConfig.from_dict({"street_alignment_max_distance": None})
    assert cfg.street_alignment_max_distance is None


def test_to_dict_round_trip():
    cfg = MetricConfig(knn_k=7, network_subgraph_radius=3, street_profile_distance=5.0)
    d = cfg.to_dict()
    cfg2 = MetricConfig.from_dict(d)
    assert cfg2.knn_k == 7
    assert cfg2.network_subgraph_radius == 3
    assert cfg2.street_profile_distance == 5.0


def test_from_json(tmp_path):
    data = {"knn_k": 10, "tessellation_max_buffer": 200.0}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    cfg = MetricConfig.from_json(p)
    assert cfg.knn_k == 10
    assert cfg.tessellation_max_buffer == 200.0
    assert cfg.tessellation_min_buffer == 0.0  # default


def test_from_json_unknown_key(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"bad_key": 1}))
    with pytest.raises(ValueError, match="Unknown MetricConfig keys"):
        MetricConfig.from_json(p)

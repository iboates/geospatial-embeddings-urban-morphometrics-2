"""Tests for the aggregate_series helper."""

import math
import numpy as np
import pandas as pd
import pytest

from urban_morphometrics.metrics.aggregation import aggregate_series


def test_basic_statistics():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = aggregate_series(s, "x", num_quantiles=4)
    assert result["x_mean"] == pytest.approx(3.0)
    assert result["x_median"] == pytest.approx(3.0)
    assert result["x_std"] == pytest.approx(s.std())


def test_quantile_keys():
    result = aggregate_series(pd.Series([1.0, 2.0, 3.0, 4.0]), "v", num_quantiles=4)
    assert "v_q25" in result
    assert "v_q50" in result
    assert "v_q75" in result
    assert "v_q100" in result


def test_quantile_count():
    result = aggregate_series(pd.Series(range(10), dtype=float), "v", num_quantiles=10)
    quantile_keys = [k for k in result if k.startswith("v_q")]
    assert len(quantile_keys) == 10


def test_empty_series_returns_nan():
    result = aggregate_series(pd.Series(dtype=float), "x", num_quantiles=4)
    assert math.isnan(result["x_mean"])
    assert math.isnan(result["x_median"])
    assert math.isnan(result["x_std"])
    assert all(math.isnan(v) for k, v in result.items() if k.startswith("x_q"))


def test_all_nan_series_returns_nan():
    result = aggregate_series(pd.Series([np.nan, np.nan, np.nan]), "x", num_quantiles=2)
    assert math.isnan(result["x_mean"])


def test_nan_values_dropped():
    s = pd.Series([1.0, np.nan, 3.0])
    result = aggregate_series(s, "x", num_quantiles=2)
    assert result["x_mean"] == pytest.approx(2.0)


def test_prefix_applied():
    result = aggregate_series(pd.Series([1.0, 2.0]), "my_metric", num_quantiles=2)
    assert all(k.startswith("my_metric_") for k in result)


def test_single_value():
    result = aggregate_series(pd.Series([42.0]), "x", num_quantiles=4)
    assert result["x_mean"] == pytest.approx(42.0)
    assert result["x_median"] == pytest.approx(42.0)
    # std of a single value is NaN
    assert math.isnan(result["x_std"])

import numpy as np
import pytest

from src.evacuation.risk_mapper import _normalise


def test_normalise_scales_values():
    result = _normalise(np.array([10.0, 20.0, 30.0]))

    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_normalise_preserves_nan():
    result = _normalise(np.array([10.0, np.nan, 30.0]))

    assert result[0] == 0.0
    assert np.isnan(result[1])
    assert result[2] == 1.0


def test_normalise_constant_values():
    result = _normalise(np.array([5.0, 5.0, 5.0]))

    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_normalise_empty_values():
    result = _normalise(np.array([np.nan, np.nan]))

    assert np.isnan(result).all()

"""Pruebas de visualización de pesos sin iniciar Tkinter ni usar GPU."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minecraft_biomas"))

from weights_view import dense_weight_layers, draw_dense_connections


def test_detects_only_dense_layers():
    weights = {
        "0.weight": torch.ones(4, 3, 3, 3),
        "1.bias": torch.zeros(3),
        "2.weight": torch.arange(30.0).reshape(5, 6),
    }
    assert dense_weight_layers(weights) == [("2.weight", 6, 5)]


def test_draw_limits_connections_without_mutating_weights():
    weights = torch.arange(-300.0, 300.0).reshape(20, 30)
    before = weights.clone()
    fig = draw_dense_connections(weights, "10.weight", max_neurons=8, max_connections=15)
    assert torch.equal(weights, before)
    assert len(fig.axes) == 1
    assert 0 < len(fig.axes[0].lines) <= 15
    assert len(fig.axes[0].collections) == 16


def test_zero_weights_and_single_neuron_are_supported():
    fig = draw_dense_connections(torch.zeros(1, 1), "0.weight")
    assert len(fig.axes[0].lines) == 0
    assert len(fig.axes[0].collections) == 2


@pytest.mark.parametrize("bad", [torch.ones(2, 3, 4), torch.tensor([[float("nan")]])])
def test_rejects_invalid_weights(bad):
    with pytest.raises(ValueError):
        draw_dense_connections(bad, "bad.weight")

"""Generous CPU smoke budgets, not cross-machine speed benchmarks."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minecraft_biomas"))
from core import ConvNet

torch.set_num_threads(2)


@pytest.mark.performance
def test_cpu_inference_does_not_regress_catastrophically():
    model = ConvNet(image_size=32, num_classes=5).eval()
    batch = torch.randn(8, 3, 32, 32)
    with torch.inference_mode():
        for _ in range(3):
            model(batch)
        start = time.perf_counter()
        for _ in range(12):
            output = model(batch)
        seconds_per_batch = (time.perf_counter() - start) / 12
    assert output.shape == (8, 5)
    assert torch.isfinite(output).all()
    assert seconds_per_batch < 1.5, (
        f"CPU inference took {seconds_per_batch:.2f}s/batch; budget is 1.5s/batch"
    )


@pytest.mark.performance
def test_cpu_optimizer_steps_within_generous_budget():
    model = ConvNet(image_size=32, num_classes=5).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.arange(8) % 5

    def step():
        optimizer.zero_grad()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    step()
    start = time.perf_counter()
    for _ in range(8):
        final_loss = step()
    seconds_per_step = (time.perf_counter() - start) / 8
    assert torch.isfinite(final_loss)
    assert seconds_per_step < 2.0, (
        f"CPU training took {seconds_per_step:.2f}s/step; budget is 2.0s/step"
    )


@pytest.mark.performance
def test_default_model_stays_within_parameter_budget():
    model = ConvNet(image_size=32, num_classes=5)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 0 < parameters < 2_000_000, f"Unexpected parameter growth: {parameters}"

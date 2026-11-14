"""CPU-only regressions for model persistence, deterministic training and cancellation."""
from __future__ import annotations

import pickle
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minecraft_biomas"))
from core import Cancelled, Config, ConvNet, cap_samples, load_checkpoint, predict, split_samples, train

torch.set_num_threads(2)


@pytest.fixture
def synthetic_entries(tmp_path: Path) -> list[dict[str, str]]:
    rng = np.random.default_rng(20261114)
    entries = []
    for label in range(5):
        folder = tmp_path / f"biome_{label}"
        folder.mkdir()
        for index in range(12):
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{index:02d}.png")
        entries.append({"name": f"biome_{label}", "folder": str(folder)})
    return entries


def _minimal_checkpoint() -> dict:
    return {
        "state_dict": ConvNet(image_size=32, num_classes=5).state_dict(),
        "classes": [f"biome_{i}" for i in range(5)],
        "image_size": 32,
    }


def test_checkpoint_round_trip(tmp_path: Path, synthetic_entries: list[dict[str, str]]):
    path = tmp_path / "model.pth"
    original = _minimal_checkpoint()
    torch.save(original, path)
    restored = load_checkpoint(path)
    assert restored["classes"] == original["classes"]
    assert restored["image_size"] == 32
    for name, tensor in original["state_dict"].items():
        torch.testing.assert_close(tensor, restored["state_dict"][name], rtol=0, atol=0)
    sample = Path(synthetic_entries[0]["folder"]) / "00.png"
    probabilities = predict(restored, sample)
    assert [label for label, _ in probabilities] == original["classes"]
    assert sum(value for _, value in probabilities) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("broken", [
    "not_a_dict", "missing_weights", "missing_classes", "bad_image_size",
    "empty_classes", "wrong_weights",
])
def test_invalid_checkpoint_rejected(tmp_path: Path, broken: str):
    checkpoint = _minimal_checkpoint()
    if broken == "not_a_dict":
        checkpoint = ["not", "a", "checkpoint"]
    elif broken == "missing_weights":
        del checkpoint["state_dict"]
    elif broken == "missing_classes":
        del checkpoint["classes"]
    elif broken == "bad_image_size":
        checkpoint["image_size"] = 17
    elif broken == "empty_classes":
        checkpoint["classes"] = []
    elif broken == "wrong_weights":
        checkpoint["state_dict"] = {}
    path = tmp_path / "invalid.pth"
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(path)


def test_truncated_checkpoint_rejected(tmp_path: Path):
    path = tmp_path / "truncated.pth"
    path.write_bytes(b"not a valid PyTorch checkpoint")
    with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError)):
        load_checkpoint(path)


def test_sampling_and_splits_reproducible():
    samples = [
        (f"biome_{label}/{index:03d}.png", label, f"hash-{label}-{index}")
        for label in range(5) for index in range(30)
    ]
    selected = cap_samples(samples, limit=20, seed=51)
    assert selected == cap_samples(samples, limit=20, seed=51)
    assert selected != cap_samples(samples, limit=20, seed=52)
    assert all(sum(label == entry[1] for entry in selected) == 20 for label in range(5))
    cfg = Config(device="cpu", seed=51, validation_percent=20)
    first = split_samples(selected, cfg)
    assert first == split_samples(selected, cfg)
    assert first != split_samples(selected, Config(device="cpu", seed=52, validation_percent=20))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["val"]) & set(first["test"]))
    assert not (set(first["train"]) & set(first["val"]))
    assert set(first["train"] + first["val"] + first["test"]) == set(range(100))


def test_cpu_training_reproducible(tmp_path: Path, synthetic_entries: list[dict[str, str]]):
    cfg = Config(device="cpu", seed=2026, image_size=32, batch_size=8, epochs=1)
    first = train(synthetic_entries, cfg, tmp_path / "run-one")
    second = train(synthetic_entries, cfg, tmp_path / "run-two")
    assert first["checkpoint"]["history"] == second["checkpoint"]["history"]
    assert first["checkpoint"]["metrics"] == second["checkpoint"]["metrics"]
    weights_one = first["checkpoint"]["state_dict"]
    weights_two = second["checkpoint"]["state_dict"]
    assert weights_one.keys() == weights_two.keys()
    for name in weights_one:
        torch.testing.assert_close(weights_one[name], weights_two[name], rtol=0, atol=0)
    assert (
        Path(first["path"]).with_name("particiones.csv").read_text(encoding="utf-8")
        == Path(second["path"]).with_name("particiones.csv").read_text(encoding="utf-8")
    )


def test_cancel_after_first_training_batch_leaves_no_artifacts(
    tmp_path: Path, synthetic_entries: list[dict[str, str]]
):
    cancellation = threading.Event()
    events = []

    def on_event(kind, value):
        if kind == "progress":
            events.append(value)
            cancellation.set()

    destination = tmp_path / "cancelled-run"
    with pytest.raises(Cancelled):
        train(
            synthetic_entries,
            Config(device="cpu", image_size=32, batch_size=8, epochs=3),
            destination,
            emit=on_event,
            cancel=cancellation,
        )
    assert events, "Cancellation must happen after training has begun"
    assert not destination.exists(), "An interrupted run must not publish a checkpoint"

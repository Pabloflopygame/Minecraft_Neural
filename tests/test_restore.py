import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minecraft_biomas"))
from app import App
from architectures import ORIGINAL
from core import Config, initialize_weights, make_model, make_optimizer, train
from restore import load_workspace, save_copy


def test_load_and_train_independent_copy(tmp_path):
    entries = []
    for i in range(5):
        folder = tmp_path / f"folder_{i}"
        folder.mkdir()
        entries.append(dict(name=f"label_{i}", folder=str(folder)))
        for j in range(15):
            Image.new("RGB", (32, 32), (i, j, 0)).save(folder / f"{j}.png")
    cfg = Config(device="cpu", image_size=32, epochs=1, lr=0.003, validation_percent=20)
    original = Path(train(entries, cfg, tmp_path)["path"])
    digest = hashlib.sha256(original.read_bytes()).digest()
    workspace = load_workspace(original, tmp_path)
    cp = workspace["checkpoint"]
    assert workspace["config"]["lr"] == cfg.lr
    assert [row["name"] for row in workspace["rows"]] == cp["classes"]
    assert [row["count"] for row in workspace["rows"]] == [15] * 5

    # A previous training run's FP32 weights must reach the optimizer unmodified.
    def inspect(model, config):
        torch.testing.assert_close(model[0].weight, cp["state_dict"]["conv1.weight"])
        torch.testing.assert_close(model[9].bias, cp["state_dict"]["fc2.bias"])
        return make_optimizer(model, config)

    cfg.architecture = ORIGINAL
    cfg.lr = 0.001
    with patch("core.make_optimizer", side_effect=inspect):
        child = train(entries, cfg, tmp_path, parent=workspace)
    assert child["checkpoint"]["parent_checkpoint"] == str(original)
    assert Path(child["path"]).parent != original.parent
    assert hashlib.sha256(original.read_bytes()).digest() == digest
    loaded = load_workspace(child["path"], tmp_path)
    favorite = tmp_path / "favorita.pth"
    save_copy(loaded, favorite)
    with pytest.raises(ValueError, match="nombre nuevo"):
        save_copy(loaded, original)
    copied = load_workspace(favorite, tmp_path)
    assert copied["rows"] == loaded["rows"]
    assert copied["config"] == loaded["config"]
    torch.testing.assert_close(
        copied["checkpoint"]["state_dict"], loaded["checkpoint"]["state_dict"]
    )
    net = make_model(32, 5, ORIGINAL)
    initialize_weights(net, cfg, cp["classes"], loaded)
    for name, tensor in net.state_dict().items():
        torch.testing.assert_close(tensor, loaded["checkpoint"]["state_dict"][name])
    with pytest.raises(ValueError, match="orden"):
        initialize_weights(net, cfg, list(reversed(cp["classes"])), loaded)
    # An older .pth uses its sidecar and a relocated dataset, preserving renamed labels.
    del cp["entries"]
    torch.save(cp, original)
    sidecar = original.with_name("resultados.json")
    data = json.loads(sidecar.read_text())
    for row in data["entries"]:
        row["folder"] = "C:\\old\\" + Path(row["folder"]).name
    sidecar.write_text(json.dumps(data))
    restored = load_workspace(original, tmp_path)
    assert restored["rows"] == workspace["rows"]
    missing = load_workspace(original, tmp_path / "missing")
    assert all(row["count"] == 0 for row in missing["rows"])
    assert missing["notes"]


def test_loaded_event_restores_gui_fields_and_enables_weights():
    view = SimpleNamespace(
        initial_values={"lr": "0.001", "image_size": "64", "device": "cpu"},
        values={k: Mock() for k in ("lr", "image_size", "device")},
        arch_editor=Mock(),
        replace_rows=Mock(),
        reuse=Mock(),
    )
    workspace = dict(
        checkpoint={"image_size": 32, "architecture": ORIGINAL},
        config={"lr": 0.002, "device": "cuda"},
        rows=[{"name": "forest"}],
        notes=[],
    )
    with patch("torch.cuda.is_available", return_value=False):
        assert App.restore_training(view, workspace)
    view.values["lr"].set.assert_called_once_with("0.002")
    view.values["image_size"].set.assert_called_once_with("32")
    view.values["device"].set.assert_called_once_with("cpu")
    view.replace_rows.assert_called_once_with(workspace["rows"])
    assert json.loads(view.arch_editor.insert.call_args.args[1]) == ORIGINAL
    view.reuse.set.assert_called_once_with(True)

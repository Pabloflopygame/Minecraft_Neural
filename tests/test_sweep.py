import csv
import sys
import tempfile
import threading
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minecraft_biomas"))
from PIL import Image
from architectures import COMPACT
from core import Config
from sweep import axis_values, error_rates, run_sweep


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class SweepTests(TestCase):
    def test_decimal_range_and_rates(self):
        self.assertEqual(axis_values("lr", "0.1", "0.31", "0.1"), (0.1, 0.2, 0.3))
        with self.assertRaises(ValueError):
            axis_values("seed", "0", "2", "0.5")
        self.assertEqual(axis_values("optimizer", "SGD, Adam", "", ""), ("SGD", "Adam"))
        rate = list(error_rates([[8, 2], [1, 4]]))[0]
        self.assertEqual(rate, dict(tp=8, tn=4, fp=1, fn=2, fpr_pct=20, fnr_pct=20))

    def test_two_by_two_with_layer_parameter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = []
            for label in range(5):
                folder = root / str(label)
                folder.mkdir()
                for i in range(12):
                    Image.new("RGB", (32, 32), (label * 40, i, 0)).save(
                        folder / f"{i}.png"
                    )
                entries.append(dict(name=str(label), folder=str(folder)))
            cfg = Config(
                device="cpu",
                epochs=1,
                image_size=32,
                architecture=COMPACT,
                validation_percent=16,
            )
            path = Path(
                run_sweep(
                    entries,
                    cfg,
                    {"optimizer": ("Adam", "SGD"), "layers.16.p": (0.1, 0.2)},
                    root,
                    lambda *args: None,
                    threading.Event(),
                )
            )
            rows = read_csv(path)
            self.assertEqual(
                {(r["x_value"], r["y_value"]) for r in rows},
                {(o, p) for o in ("Adam", "SGD") for p in ("0.1", "0.2")},
            )
            self.assertEqual(len(read_csv(path.parent / "biomas.csv")), 20)
            for row in rows:
                model = path.parent / row["checkpoint"]
                self.assertTrue(model.is_file())
                predictions = read_csv(model.parent / "predicciones_test.csv")
                accuracy = sum(int(r["correcto"]) for r in predictions) / len(
                    predictions
                )
                self.assertAlmostEqual(accuracy, float(row["test_accuracy"]))
                self.assertTrue(row["val_macro_f1"])
            self.assertEqual(COMPACT["layers"][16]["p"], 0.25)

"""Barridos cartesianos y exportación incremental de experimentos."""

import csv
import json
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
from itertools import product
from pathlib import Path
from uuid import uuid4
from time import monotonic

OPTIONS = {
    "device": ("cuda", "cpu"),
    "optimizer": ("Adam", "AdamW", "SGD"),
    "scheduler": ("none", "plateau", "cosine"),
    "class_weighting": ("none", "balanced"),
    "selection_metric": ("val_loss", "val_macro_f1"),
}

LAYER_FIELDS = {
    "Conv2d": ("out_channels", "kernel_size", "stride", "padding"),
    "MaxPool2d": ("kernel_size", "stride"),
    "AvgPool2d": ("kernel_size", "stride"),
    "AdaptiveAvgPool2d": ("output_size",),
    "Dropout": ("p",),
    "Linear": ("out_features",),
}


def layer_parameters(spec):
    for index, layer in enumerate(spec["layers"]):
        for field in LAYER_FIELDS.get(layer["type"], ()):
            if layer.get(field) != "classes":
                yield f"layers.{index}.{field}"


def axis_values(key, start, stop, step):
    if key in OPTIONS:
        values = tuple(dict.fromkeys(v.strip() for v in start.split(",")))
        if any(v not in OPTIONS[key] for v in values):
            raise ValueError(f"{key}: elige valores de {', '.join(OPTIONS[key])}.")
        return values
    try:
        a, b, delta = map(Decimal, (start, stop, step))
    except InvalidOperation as exc:
        raise ValueError(f"{key}: introduce inicio, fin y paso numéricos.") from exc
    if not all(v.is_finite() for v in (a, b, delta)) or delta <= 0 or a > b:
        raise ValueError(
            f"{key}: el inicio debe ser ≤ fin y el paso debe ser positivo."
        )
    integer = key not in ("lr", "weight_decay") and not key.endswith(".p")
    if integer and any(v != int(v) for v in (a, b, delta)):
        raise ValueError(f"{key}: se necesitan valores enteros.")
    count = int((b - a) // delta) + 1
    if count > 10000:
        raise ValueError(f"{key}: hay más de 10000 valores. Aumenta el paso.")
    convert = int if integer else float
    return tuple(convert(a + i * delta) for i in range(count))


def configuration(base, settings):
    cfg = replace(base, architecture=deepcopy(base.architecture))
    for key, value in settings.items():
        if key.startswith("layers."):
            _, index, field = key.split(".")
            cfg.architecture["layers"][int(index)][field] = value
        else:
            setattr(cfg, key, value)
    return cfg


def error_rates(matrix):
    """Una clase frente al resto; filas reales y columnas predichas."""
    total = sum(map(sum, matrix))
    for i, row in enumerate(matrix):
        tp = row[i]
        fn = sum(row) - tp
        fp = sum(other[i] for other in matrix) - tp
        tn = total - tp - fn - fp
        yield dict(
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            fpr_pct=100 * fp / (fp + tn) if fp + tn else None,
            fnr_pct=100 * fn / (fn + tp) if fn + tp else None,
        )


def append_rows(path, rows):
    rows = iter(rows)
    first = next(rows)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        if new:
            writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def run_sweep(entries, base, axes, output, emit, cancel, *, parent=None):
    from core import check_cancel, collect, train
    from architectures import validate_architecture

    folder = Path(output) / ("barrido_" + uuid4().hex[:12])
    folder.mkdir(parents=True)
    summary = folder / "resumen.csv"
    keys = list(axes)
    total = len(axes[keys[0]]) * len(axes[keys[1]])
    started = monotonic()
    emit("sweep_progress", (0, f"Preparando datos · 0/{total} combinaciones · 0 %"))
    emit("log", f"Barrido de {total} redes: {folder}")
    collected = collect(entries, cancel, lambda msg: emit("log", msg))
    training_started = monotonic()
    for number, pair in enumerate(product(*axes.values()), 1):
        check_cancel(cancel)
        settings = dict(zip(keys, pair))
        cfg = configuration(base, settings)
        label = f"Red {number}/{total} · {settings}"
        run_started = monotonic()
        last_update = None

        def report(kind, value):
            nonlocal last_update
            if kind != "progress":
                emit(kind, value)
                return
            now = monotonic()
            if last_update is not None and now - last_update < 0.25 and value < 100:
                return
            last_update = now
            # El último tramo incluye la evaluación y escritura de los resultados.
            finished = number - 1 + min(value / 100, 0.999)
            percent = min(99.9, int(1000 * finished / total) / 10)
            remaining = "calculando…"
            if finished:
                remaining = duration(
                    (now - training_started) * (total - finished) / finished
                )
            emit(
                "sweep_progress",
                (
                    percent,
                    f"Combinación {number}/{total} · {percent:.1f} % del barrido · "
                    f"Transcurrido {duration(now - started)} · Restante estimado {remaining}",
                ),
            )

        report("progress", 0)
        emit("log", label)
        try:
            cfg.validate()
            validate_architecture(cfg.architecture, cfg.image_size, len(entries))
            result = train(
                entries, cfg, folder, report, cancel, collected=collected, parent=parent
            )
        except ValueError as exc:
            raise ValueError(
                f"{label}: {exc}. Resultados completados: {folder}"
            ) from exc
        cp = result["checkpoint"]
        metrics = cp["metrics"]
        epoch = cp["history"][metrics["selected_epoch"] - 1]
        common = dict(
            run=number,
            x_parameter=keys[0],
            x_value=pair[0],
            y_parameter=keys[1],
            y_value=pair[1],
        )
        rates = list(error_rates(metrics["confusion_matrix"]))
        append_rows(
            folder / "biomas.csv",
            (
                {**common, "bioma": name, **rate, **metrics["report"][f"class_{i}"]}
                for i, (name, rate) in enumerate(zip(cp["classes"], rates))
            ),
        )
        config = asdict(cfg)
        config["architecture"] = json.dumps(config["architecture"], ensure_ascii=False)
        row = {
            **common,
            **config,
            "checkpoint": str(Path(result["path"]).relative_to(folder)),
            "selected_epoch": metrics["selected_epoch"],
        }
        for metric in ("loss", "accuracy", "macro_f1", "balanced_accuracy"):
            row["test_" + metric] = metrics[metric]
            row["val_" + metric] = epoch.get("val_" + metric)
        for metric in ("fpr_pct", "fnr_pct"):
            row["macro_" + metric] = sum(r[metric] for r in rates) / len(rates)
        append_rows(summary, [row])
        elapsed = monotonic() - run_started
        emit("log", f"{label} completada en {duration(elapsed)} ({elapsed:.1f} s).")
        del result, cp
    elapsed = duration(monotonic() - started)
    emit(
        "sweep_progress",
        (
            100,
            f"Barrido completado · {total}/{total} combinaciones · 100 % · Total {elapsed}",
        ),
    )
    emit("log", f"Tiempo total del barrido: {elapsed}.")
    return str(summary)

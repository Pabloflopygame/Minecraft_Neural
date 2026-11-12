"""Arquitecturas secuenciales declarativas. JSON nunca ejecuta código Python."""

import copy
import math
import torch
from torch import nn

ORIGINAL = {
    "name": "CNN original",
    "layers": [
        {"type": "Conv2d", "out_channels": 16, "kernel_size": 5, "padding": 2},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Conv2d", "out_channels": 32, "kernel_size": 5, "padding": 2},
        {"type": "ReLU"},
        {"type": "MaxPool2d", "kernel_size": 2},
        {"type": "Flatten"},
        {"type": "Linear", "out_features": 120},
        {"type": "ReLU"},
        {"type": "Linear", "out_features": "classes"},
    ],
}
COMPACT = {
    "name": "CNN compacta · 3 convoluciones",
    "layers": [
        layer
        for channels in (16, 32, 64)
        for layer in (
            {
                "type": "Conv2d",
                "out_channels": channels,
                "kernel_size": 3,
                "padding": 1,
            },
            {"type": "BatchNorm2d"},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel_size": 2},
        )
    ]
    + [
        {"type": "AdaptiveAvgPool2d", "output_size": 2},
        {"type": "Flatten"},
        {"type": "Linear", "out_features": 128},
        {"type": "ReLU"},
        {"type": "Dropout", "p": 0.25},
        {"type": "Linear", "out_features": "classes"},
    ],
}
PRESETS = {"Original": ORIGINAL, "Compacta (3 convoluciones)": COMPACT}
FIELDS = {
    "Conv2d": {"out_channels", "kernel_size", "stride", "padding"},
    "BatchNorm2d": set(),
    "ReLU": set(),
    "MaxPool2d": {"kernel_size", "stride"},
    "AvgPool2d": {"kernel_size", "stride"},
    "AdaptiveAvgPool2d": {"output_size"},
    "Flatten": set(),
    "Linear": {"out_features"},
    "Dropout": {"p"},
}


def _integer(value, name, minimum=1, maximum=4096):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}: entero entre {minimum} y {maximum}.")
    return value


def _build(spec, size, classes, device):
    if not isinstance(spec, dict) or set(spec) != {"name", "layers"}:
        raise ValueError("El JSON debe contener solamente name y layers.")
    if not isinstance(spec["name"], str) or not 1 <= len(spec["name"].strip()) <= 120:
        raise ValueError("name debe ser texto de 1–120 caracteres.")
    layers = spec["layers"]
    if not isinstance(layers, list) or not 1 <= len(layers) <= 64:
        raise ValueError("layers debe contener entre 1 y 64 capas.")
    modules = []
    rows = []
    total = 0
    x = torch.empty(1, 3, size, size, device="meta")
    for i, layer in enumerate(layers, 1):
        try:
            if (
                not isinstance(layer, dict)
                or not isinstance(layer.get("type"), str)
                or layer["type"] not in FIELDS
            ):
                raise ValueError("Tipo de capa desconocido.")
            kind = layer["type"]
            p = {k: v for k, v in layer.items() if k != "type"}
            if set(p) - FIELDS[kind]:
                raise ValueError(
                    "Parámetros no admitidos: " + ", ".join(set(p) - FIELDS[kind])
                )
            if (
                kind
                in (
                    "Conv2d",
                    "BatchNorm2d",
                    "MaxPool2d",
                    "AvgPool2d",
                    "AdaptiveAvgPool2d",
                )
                and x.ndim != 4
            ):
                raise ValueError(
                    "Esta capa necesita mapas de imagen; colócala antes de Flatten."
                )
            for key, value in p.items():
                if key == "p":
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or not 0 <= value < 1
                    ):
                        raise ValueError(
                            "Dropout p: entre 0 (incluido) y 1 (excluido)."
                        )
                elif key == "out_features" and value == "classes":
                    p[key] = classes
                else:
                    _integer(
                        value,
                        key,
                        0 if key == "padding" else 1,
                        4096 if key in ("out_channels", "out_features") else 128,
                    )
            with torch.device(device):
                if kind == "Conv2d":
                    module = nn.Conv2d(x.shape[1], **p)
                elif kind == "BatchNorm2d":
                    if x.shape[2] * x.shape[3] == 1:
                        raise ValueError(
                            "BatchNorm necesita más de un valor espacial para batches de tamaño 1."
                        )
                    module = nn.BatchNorm2d(x.shape[1])
                elif kind == "Linear":
                    if x.ndim != 2:
                        raise ValueError("Añade Flatten antes de Linear.")
                    module = nn.Linear(x.shape[1], **p)
                elif kind == "Flatten":
                    module = nn.Flatten(1)
                else:
                    module = getattr(nn, kind)(**p)
            # El pase de validación usa exclusivamente tensores meta, sin reservar pesos reales.
            if device == "meta":
                module.eval()
                x = module(x)
                if any(d <= 0 for d in x.shape) or x.numel() > 4_000_000:
                    raise ValueError("Dimensiones inválidas o activaciones excesivas.")
            else:
                # Las formas ya se verificaron en el pase meta.
                with torch.device("meta"):
                    shadow = copy.deepcopy(module).to("meta").eval()
                    x = shadow(x)
            count = sum(v.numel() for v in module.parameters())
            total += count
            if total > 20_000_000:
                raise ValueError("Máximo permitido: 20 millones de parámetros.")
            rows.append(
                f"{i:02d}  {str(module):65s} → {tuple(x.shape[1:])}   [{count:,} parámetros]"
            )
            modules.append(module)
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            raise ValueError(f"Capa {i}: {exc}") from exc
    if layers[-1].get("type") != "Linear" or tuple(x.shape) != (1, classes):
        raise ValueError(
            'Termina con Linear y out_features: "classes" (logits, sin Softmax).'
        )
    return nn.Sequential(*modules), rows, total


def validate_architecture(spec, size=64, classes=10):
    _integer(size, "Tamaño", 1, 128)
    _integer(classes, "Clases", 2, 1000)
    _, rows, total = _build(spec, size, classes, "meta")
    return (
        f"{spec['name']}\nEntrada RGB: 3 × {size} × {size}\nSalidas: {classes}\nParámetros: {total:,}\n\n"
        + "\n".join(rows)
    )


def build_architecture(spec, size, classes):
    validate_architecture(spec, size, classes)
    model, _, _ = _build(spec, size, classes, "cpu")
    return model.train()

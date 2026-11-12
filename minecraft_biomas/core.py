"""Red y entrenamiento: sin ventanas, descargas ni ejecución al importar."""

from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import math
import random
import threading
import uuid
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from architectures import build_architecture

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Config:
    epochs: int = 20
    batch_size: int = 64
    image_size: int = 64
    lr: float = 0.001
    seed: int = 42
    validation_percent: int = 0  # Porcentaje del total; test siempre 20 %.
    device: str = "cuda"
    optimizer: str = "Adam"
    scheduler: str = "none"
    weight_decay: float = 0.0
    class_weighting: str = "none"
    max_images_per_class: int = (
        0  # 0: sin límite. Se aplica antes de separar train/val/test.
    )
    selection_metric: str = "val_loss"
    scheduler_patience: int = 2
    architecture: dict | None = None

    def validate(self):
        if not 1 <= self.epochs <= 10000:
            raise ValueError("Épocas: 1–10000.")
        if not 1 <= self.batch_size <= 4096:
            raise ValueError("Batch: 1–4096.")
        if self.image_size not in (32, 64, 96, 128):
            raise ValueError("Tamaño: 32, 64, 96 o 128.")
        if not math.isfinite(self.lr) or not 0 < self.lr <= 1:
            raise ValueError("Learning rate: mayor que 0 y hasta 1.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Semilla: entre 0 y 2^32−1.")
        if not 0 <= self.validation_percent <= 30:
            raise ValueError("Validación: 0–30 % del total.")
        if self.optimizer not in ("Adam", "AdamW", "SGD"):
            raise ValueError("Optimizador inválido.")
        if self.scheduler not in ("none", "plateau", "cosine"):
            raise ValueError("Plan de learning rate inválido.")
        if not math.isfinite(self.weight_decay) or not 0 <= self.weight_decay <= 1:
            raise ValueError("Weight decay: entre 0 y 1.")
        if self.class_weighting not in ("none", "balanced"):
            raise ValueError("Pesos CE inválidos.")
        if self.max_images_per_class != 0 and self.max_images_per_class < 10:
            raise ValueError("Máximo por bioma: 0 (sin límite) o al menos 10.")
        if self.selection_metric not in ("val_loss", "val_macro_f1"):
            raise ValueError("Criterio de selección inválido.")
        if not 0 <= self.scheduler_patience <= 1000:
            raise ValueError("Paciencia: entre 0 y 1000.")
        if self.scheduler == "plateau" and not self.validation_percent:
            raise ValueError(
                "Plateau necesita validación mayor que 0; nunca utiliza el test."
            )
        if self.selection_metric == "val_macro_f1" and not self.validation_percent:
            raise ValueError("Seleccionar por F1 requiere validación mayor que 0.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Dispositivo inválido.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA no disponible. Selecciona CPU o revisa tu instalación NVIDIA."
            )


class ConvNet(nn.Module):
    def __init__(self, image_size=64, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * (image_size // 4) ** 2, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))


def make_model(size, classes, architecture=None):
    return (
        ConvNet(size, classes)
        if architecture is None
        else build_architecture(architecture, size, classes)
    )


def preprocessing(size):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def image_files(folder):
    return sorted(
        p
        for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel.is_set():
        raise Cancelled("Operación cancelada; no se ha sustituido el modelo activo.")


def collect(entries, cancel, log):
    if not 5 <= len(entries) <= 10:
        raise ValueError("Selecciona entre 5 y 10 biomas para entrenar.")
    names = [e["name"].strip() for e in entries]
    if any(not n for n in names) or len({n.casefold() for n in names}) != len(names):
        raise ValueError("Cada bioma necesita un nombre único y no vacío.")
    paths = [Path(e["folder"]).resolve() for e in entries]
    for i, p in enumerate(paths):
        if not p.is_dir():
            raise ValueError(f"Carpeta inexistente: {p}")
        for q in paths[:i]:
            if p == q or p in q.parents or q in p.parents:
                raise ValueError(
                    "Las carpetas seleccionadas no pueden coincidir ni contenerse entre sí."
                )
    samples, seen, duplicates = [], {}, 0
    for label, folder in enumerate(paths):
        files = image_files(folder)
        log(f"Comprobando {names[label]}: {len(files)} archivos…")
        for path in files:
            check_cancel(cancel)
            try:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    digest = hashlib.sha256(
                        str(rgb.size).encode() + rgb.tobytes()
                    ).hexdigest()
            except Exception as exc:
                raise ValueError(f"Imagen ilegible: {path}") from exc
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(f"Imagen idéntica con biomas distintos: {path}")
                duplicates += 1
            else:
                seen[digest] = label
                samples.append((str(path), label, digest))
    counts = Counter(s[1] for s in samples)
    if any(counts[i] < 10 for i in range(len(names))):
        raise ValueError(
            "Cada bioma necesita al menos 10 imágenes distintas; recomendable muchas más."
        )
    log(f"{len(samples)} imágenes útiles; {duplicates} copias exactas descartadas.")
    return names, samples


def cap_samples(samples, limit, seed):
    """Muestreo sin reemplazo; no modifica archivos ni repite imágenes pequeñas."""
    if limit == 0:
        return list(samples)
    rng = random.Random(seed)
    chosen = []
    for label in sorted({s[1] for s in samples}):
        group = [s for s in samples if s[1] == label]
        indices = sorted(rng.sample(range(len(group)), min(limit, len(group))))
        chosen.extend(group[i] for i in indices)
    return chosen


def training_weights(samples, indices, num_classes, mode):
    """Los pesos solo dependen de la partición de entrenamiento efectiva."""
    if mode == "none":
        return None
    counts = np.bincount([samples[i][1] for i in indices], minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Todas las clases deben estar en entrenamiento.")
    return torch.tensor(len(indices) / (num_classes * counts), dtype=torch.float32)


def macro_metrics(truth, predicted, n):
    labels = list(range(n))
    return {
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            recall_score(
                truth, predicted, labels=labels, average="macro", zero_division=0
            )
        ),
    }


def make_optimizer(model, cfg):
    if cfg.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    cls = torch.optim.AdamW if cfg.optimizer == "AdamW" else torch.optim.Adam
    return cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def make_scheduler(optimizer, cfg):
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if cfg.selection_metric == "val_macro_f1" else "min",
            factor=0.5,
            patience=cfg.scheduler_patience,
            min_lr=min(1e-6, cfg.lr),
        )
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=min(1e-6, cfg.lr)
        )
    return None


def split_samples(samples, cfg):
    labels = np.array([s[1] for s in samples])
    indices = np.arange(len(samples))
    try:
        train, test = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=cfg.seed
        )
        val = np.array([], dtype=int)
        if cfg.validation_percent:
            train, val = train_test_split(
                train,
                test_size=cfg.validation_percent / 80,
                stratify=labels[train],
                random_state=cfg.seed,
            )
    except ValueError as exc:
        raise ValueError(
            "No hay suficientes imágenes para una separación estratificada con esa validación."
        ) from exc
    expected = set(labels)
    for part in (train, test, val):
        if len(part) and set(labels[part]) != expected:
            raise ValueError(
                "Alguna partición no contiene todas las clases. Añade imágenes o aumenta validación."
            )
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


class Images(Dataset):
    def __init__(self, samples, indices, size):
        self.samples = [samples[i] for i in indices]
        self.transform = preprocessing(size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        return x, label


def evaluate(model, loader, device, cancel):
    model.eval()
    loss_sum, truth, predicted = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            check_cancel(cancel)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y).item() * len(y)
            truth.extend(y.cpu().tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    return (
        loss_sum / len(truth),
        float(np.mean(np.array(truth) == predicted)),
        truth,
        predicted,
    )


def train(entries, cfg, output_root, emit=lambda kind, value: None, cancel=None):
    cancel = cancel or threading.Event()
    cfg.validate()
    log = lambda s: emit("log", s)
    names, samples = collect(entries, cancel, log)
    available = Counter(s[1] for s in samples)
    samples = cap_samples(samples, cfg.max_images_per_class, cfg.seed)
    selected = Counter(s[1] for s in samples)
    for i, name in enumerate(names):
        log(f"{name}: {available[i]} útiles → {selected[i]} seleccionadas")
    parts = split_samples(samples, cfg)
    log("Separación: " + ", ".join(f"{k}={len(v)}" for k, v in parts.items()))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loaders = {
        k: DataLoader(
            Images(samples, v, cfg.image_size),
            batch_size=cfg.batch_size,
            shuffle=k == "train",
            num_workers=0,
            pin_memory=cfg.device == "cuda",
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        for k, v in parts.items()
        if v
    }
    model = make_model(cfg.image_size, len(names), cfg.architecture).to(cfg.device)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    weights = training_weights(samples, parts["train"], len(names), cfg.class_weighting)
    if weights is not None:
        log(
            "Pesos CE calculados solo con train: "
            + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights))
        )
        weights = weights.to(cfg.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history, best_state, best_score, best_epoch = [], None, None, cfg.epochs
    data_summary = [
        {
            "name": name,
            "available": available[i],
            "selected": selected[i],
            **{k: sum(samples[j][1] == i for j in ids) for k, ids in parts.items()},
        }
        for i, name in enumerate(names)
    ]
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        objective_sum, objective_denominator = 0.0, 0.0
        train_truth, train_pred = [], []
        epoch_lr = optimizer.param_groups[0]["lr"]
        for batch, (x, y) in enumerate(loaders["train"]):
            check_cancel(cancel)
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            # Curvas comparables: CE sin ponderar en train, validación y test.
            # La pérdida ponderada que optimiza Adam/SGD se registra por separado.
            loss_sum += F.cross_entropy(logits.detach(), y, reduction="sum").item()
            denominator = weights[y].sum().item() if weights is not None else len(y)
            objective_sum += loss.item() * denominator
            objective_denominator += denominator
            train_truth.extend(y.cpu().tolist())
            train_pred.extend(logits.argmax(1).detach().cpu().tolist())
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            emit(
                "progress",
                100 * (epoch + (batch + 1) / len(loaders["train"])) / cfg.epochs,
            )
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / total,
            "accuracy": correct / total,
            "objective_loss": objective_sum / objective_denominator,
            "lr": epoch_lr,
            **macro_metrics(train_truth, train_pred, len(names)),
        }
        if "val" in loaders:
            vl, va, vt, vp = evaluate(model, loaders["val"], cfg.device, cancel)
            row.update(val_loss=vl, val_accuracy=va)
            row.update(
                {"val_" + k: v for k, v in macro_metrics(vt, vp, len(names)).items()}
            )
            score = row[cfg.selection_metric]
            improved = best_score is None or (
                score > best_score
                if cfg.selection_metric == "val_macro_f1"
                else score < best_score
            )
            if improved:
                best_score, best_epoch = score, epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        history.append(row)
        emit("epoch", row)
        if scheduler is not None:
            if cfg.scheduler == "plateau":
                scheduler.step(row[cfg.selection_metric])
            else:
                scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    log("Evaluando el test reservado (20 %)…")
    loss, accuracy, truth, predicted = evaluate(
        model, loaders["test"], cfg.device, cancel
    )
    report = classification_report(
        truth,
        predicted,
        labels=list(range(len(names))),
        target_names=[f"class_{i}" for i in range(len(names))],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(range(len(names)))
        ).tolist(),
        "classes": names,
        **macro_metrics(truth, predicted, len(names)),
        "selection_metric": cfg.selection_metric if parts["val"] else "last_epoch",
        "sizes": {k: len(v) for k, v in parts.items()},
        "selected_epoch": best_epoch,
    }
    check_cancel(cancel)
    run = Path(output_root) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    )
    run.mkdir(parents=True)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": cfg.architecture,
        "classes": names,
        "image_size": cfg.image_size,
        "config": asdict(cfg),
        "history": history,
        "metrics": metrics,
        "data_summary": data_summary,
        "class_weights": weights.cpu().tolist() if weights is not None else None,
        "normalization_mean": [0.5] * 3,
        "normalization_std": [0.5] * 3,
    }
    torch.save(checkpoint, run / "biomas_cnn.pth")
    (run / "resultados.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "entries": entries,
                "history": history,
                "metrics": metrics,
                "data_summary": data_summary,
                "class_weights": checkpoint["class_weights"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (run / "particiones.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ruta", "bioma", "sha256_rgb", "particion"])
        for kind, indices in parts.items():
            for i in indices:
                path, label, digest = samples[i]
                writer.writerow([path, names[label], digest, kind])
    log(f"Test: {accuracy:.2%}. Guardado en {run}")
    return {"checkpoint": checkpoint, "path": str(run / "biomas_cnn.pth")}


def load_checkpoint(path):
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(cp, dict)
        or not {"classes", "image_size", "state_dict"} <= cp.keys()
    ):
        raise ValueError(
            "Archivo incompatible: se necesitan pesos, clases y tamaño de imagen."
        )
    if not 2 <= len(cp["classes"]) <= 1000 or cp["image_size"] not in (32, 64, 96, 128):
        raise ValueError("Dimensiones del modelo no admitidas.")
    net = make_model(cp["image_size"], len(cp["classes"]), cp.get("architecture"))
    net.load_state_dict(cp["state_dict"])
    return cp


def predict(cp, path, device="cpu"):
    net = make_model(cp["image_size"], len(cp["classes"]), cp.get("architecture")).to(
        device
    )
    net.load_state_dict(cp["state_dict"])
    net.eval()
    with Image.open(path) as im:
        x = preprocessing(cp["image_size"])(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = net(x).softmax(1)[0].cpu().tolist()
    return list(zip(cp["classes"], probabilities))

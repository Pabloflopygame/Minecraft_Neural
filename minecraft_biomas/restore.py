"""Recuperación del contexto de entrenamiento de versiones nuevas y anteriores."""

import json
from pathlib import Path, PureWindowsPath

import torch

from core import image_files, load_checkpoint


def restore_rows(classes, entries, root, *, prefer_root=False):
    folders = {
        e["name"]: e["folder"]
        for e in entries
        if isinstance(e, dict)
        and isinstance(e.get("name"), str)
        and isinstance(e.get("folder"), str)
    }
    rows = []
    for name in classes:
        saved = folders.get(name)
        candidates = [Path(saved)] if saved else []
        basename = PureWindowsPath(saved or name).name
        if basename not in ("", ".", ".."):
            candidates.append(root / basename)
        if prefer_root:
            candidates.reverse()
        folder = next((path for path in candidates if path.is_dir()), None)
        rows.append(
            dict(
                name=name,
                folder=str(folder.resolve()) if folder else saved or "",
                count=len(image_files(folder)) if folder else 0,
            )
        )
    return rows


def load_workspace(path, dataset_root):
    cp = load_checkpoint(path)
    entries = cp.get("entries")
    config = cp.get("config", {})
    notes = []
    sidecar = Path(path).with_name("resultados.json")
    if entries is None and sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                data["metrics"]["classes"] == cp["classes"]
                and data["config"]["image_size"] == cp["image_size"]
                and (not config or config == data["config"])
            ):
                entries = data["entries"]
                config = data["config"]
        except (OSError, ValueError, KeyError, TypeError):
            notes.append("No se pudo recuperar la selección de resultados.json.")
    rows = restore_rows(cp["classes"], entries or [], dataset_root)
    if not config:
        notes.append(
            "El archivo solo conserva parte de la configuración. Se han usado los valores iniciales para los campos ausentes."
        )
    if any(not row["count"] for row in rows):
        notes.append(
            "Faltan imágenes. Usa «Vincular dataset» para indicar la carpeta raíz."
        )
    return dict(checkpoint=cp, path=str(path), config=config, rows=rows, notes=notes)


def save_copy(workspace, destination):
    checkpoint = dict(workspace["checkpoint"])
    if workspace.get("config"):
        checkpoint["config"] = workspace["config"]
    if "rows" in workspace:
        checkpoint["entries"] = workspace["rows"]
    path = Path(destination)
    try:
        stream = path.open("xb")
    except FileExistsError as exc:
        raise ValueError("Elige un nombre nuevo para guardar la copia.") from exc
    try:
        with stream:
            torch.save(checkpoint, stream)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return str(path)

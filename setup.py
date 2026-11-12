#!/usr/bin/env python3
"""Preparar y abrir la aplicación de biomas de Minecraft.

Uso (desde la raíz del repositorio):
    python setup.py
    python setup.py --install-only
    python setup.py --cpu
    python setup.py --cuda

No es un archivo de empaquetado setuptools: es un instalador/lanzador local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import venv

from scripts.prepare_dataset import DatasetError, prepare_dataset


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "minecraft_biomas"
APP = PROJECT / "app.py"
REQUIREMENTS = PROJECT / "requirements.txt"
ENV_DIR = ROOT / ".venv"
STATE_FILE = ENV_DIR / ".minecraft_setup.json"


class SetupError(Exception):
    """Error de instalación con una explicación para la persona usuaria."""


def say(message: str) -> None:
    print(message, flush=True)


def call(*args: str, cwd: Path = ROOT) -> None:
    say("$ " + subprocess.list2cmdline(list(args)))
    subprocess.run(args, cwd=cwd, check=True)


def env_python() -> Path:
    if os.name == "nt":
        return ENV_DIR / "Scripts" / "python.exe"
    return ENV_DIR / "bin" / "python"


def check_python() -> None:
    if not (sys.version_info[:2] >= (3, 11) and sys.version_info[:2] <= (3, 13)):
        raise SetupError(
            "Se necesita Python de 64 bits, versión 3.11, 3.12 o 3.13 "
            "(se recomienda 3.12). Ejemplo en Windows: py -3.12 setup.py"
        )
    if sys.maxsize <= 2**32:
        raise SetupError("Se necesita una instalación de Python de 64 bits.")


def check_prerequisites() -> None:
    check_python()
    if not APP.is_file() or not REQUIREMENTS.is_file():
        raise SetupError(
            "No encuentro minecraft_biomas/app.py o minecraft_biomas/requirements.txt. "
            "Ejecuta el setup.py desde una copia completa del repositorio."
        )
    try:
        import tkinter  # noqa: F401; Tk no se instala mediante pip.
    except ImportError as exc:
        if sys.platform.startswith("linux"):
            package_hint = (
                "Instala Tkinter con el gestor de paquetes de tu distribución:\n"
                "  Fedora: sudo dnf install python3-tkinter\n"
                "  Ubuntu/Debian: sudo apt install python3-tk\n"
                "  Arch Linux: sudo pacman -S tk tcl"
            )
        else:
            package_hint = (
                "Reinstala Python desde python.org y activa las opciones "
                "Tcl/Tk e IDLE del instalador."
            )
        raise SetupError("Falta Tkinter.\n" + package_hint) from exc


def detect_nvidia() -> tuple[bool, tuple[int, int] | None]:
    """nvidia-smi informa la versión *máxima* de CUDA admitida por el driver.

    No es necesario instalar por separado el CUDA Toolkit para usar ruedas
    binarias oficiales de PyTorch.
    """
    candidates = [shutil.which("nvidia-smi")]
    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        candidates.extend([
            str(system_root / "System32" / "nvidia-smi.exe"),
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ])
    for exe in filter(None, candidates):
        try:
            proc = subprocess.run(
                [exe], capture_output=True, text=True, errors="replace",
                timeout=12, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        # Ejemplo: "CUDA Version: 12.8"; esa versión indica soporte del driver.
        match = re.search(
            r"\bCUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.(\d+)",
            proc.stdout,
            re.I,
        )
        version = (int(match.group(1)), int(match.group(2))) if match else None
        return True, version
    return False, None


def select_torch_index(device: str) -> tuple[str, bool]:
    found, max_cuda = detect_nvidia()
    want_cuda = device == "cuda" or (device == "auto" and found)
    if not want_cuda:
        say("[Hardware] CPU: se instalará PyTorch sin CUDA.")
        if platform.system() == "Darwin":
            return "pypi", False
        return "https://download.pytorch.org/whl/cpu", False
    if not found:
        raise SetupError(
            "Se solicitó CUDA, pero nvidia-smi no responde. "
            "Comprueba la tarjeta NVIDIA y sus controladores, o usa --cpu."
        )
    if max_cuda is None:
        raise SetupError(
            "NVIDIA detectada, pero no se pudo leer el soporte CUDA de nvidia-smi. "
            "Actualiza los controladores o utiliza --cpu."
        )
    if max_cuda >= (12, 8):
        index = "cu128"
    elif max_cuda >= (12, 6):
        index = "cu126"
    else:
        raise SetupError(
            f"El controlador anuncia CUDA {max_cuda[0]}.{max_cuda[1]}; "
            "este instalador requiere soporte 12.6 o superior. "
            "Actualiza el driver NVIDIA o usa --cpu."
        )
    say(f"[Hardware] NVIDIA detectada: controlador compatible con CUDA "
        f"{max_cuda[0]}.{max_cuda[1]}. PyTorch {index}.")
    return f"https://download.pytorch.org/whl/{index}", True


def setup_signature(index: str) -> str:
    digest = hashlib.sha256()
    digest.update(REQUIREMENTS.read_bytes())
    digest.update(index.encode("utf-8"))
    digest.update(f"{sys.version_info.major}.{sys.version_info.minor}".encode("ascii"))
    digest.update(b"bootstrap-v1")
    return digest.hexdigest()


def environment_is_healthy(python: Path, cuda: bool) -> bool:
    if not python.is_file():
        return False
    checks = (
        "import tkinter, numpy, PIL, matplotlib, sklearn, torch, torchvision\n"
        "from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg\n"
        "assert not " + repr(cuda) + " or torch.cuda.is_available(), "
        "'PyTorch no puede utilizar CUDA'\n"
        "if " + repr(cuda) + ": assert torch.ones(1, device='cuda').sum().item() == 1\n"
        "print('PyTorch:', torch.__version__, '| CUDA disponible:', "
        "torch.cuda.is_available())\n"
        "if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))\n"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", checks], cwd=ROOT,
            capture_output=True, text=True, errors="replace", timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode == 0:
        say("[Verificación] " + result.stdout.strip().replace("\n", " · "))
        return True
    say("[Verificación] Dependencias ausentes/incompatibles: " +
        (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "verificación fallida"))
    return False


def check_environment_tk(python: Path) -> None:
    # Tkinter depende de bibliotecas del sistema; pip no puede repararlo.
    result = subprocess.run(
        [str(python), "-c", "import tkinter"], cwd=ROOT, capture_output=True,
        text=True, errors="replace", timeout=30, check=False,
    )
    if result.returncode:
        raise SetupError(
            "Tkinter no funciona dentro de .venv. Comprueba Tcl/Tk en la instalación "
            "de Python o el paquete python3-tkinter/python3-tk de tu distribución. "
            "No se descargan dependencias hasta corregirlo. Detalle: " + result.stderr.strip()
        )


def install(python: Path, index: str) -> None:
    call(str(python), "-m", "pip", "install", "--upgrade", "pip")
    if index == "pypi":
        call(str(python), "-m", "pip", "install", "--force-reinstall",
             "torch==2.10.0", "torchvision==0.25.0")
    else:
        call(str(python), "-m", "pip", "install", "--force-reinstall", "--index-url", index,
             "torch==2.10.0", "torchvision==0.25.0")
    call(str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS))


def main() -> int:
    parser = argparse.ArgumentParser(description="Instala dependencias y abre Minecraft Biomas.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cpu", action="store_true", help="Instalar PyTorch CPU.")
    selection.add_argument("--cuda", action="store_true", help="Exigir CUDA de NVIDIA.")
    parser.add_argument("--install-only", action="store_true",
                        help="Preparar y verificar sin abrir la interfaz.")
    parser.add_argument("--reinstall", action="store_true",
                        help="Reinstalar/actualizar dependencias en .venv.")
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--skip-dataset", action="store_true",
                         help="No descargar ni comprobar el dataset; usar datos propios.")
    dataset.add_argument("--dataset-source", type=Path, metavar="RUTA",
                         help="Importar un ZIP o carpeta descargada de Kaggle, sin modificar el origen.")
    parser.add_argument("--dataset-only", action="store_true",
                        help="Preparar solo el dataset, sin instalar PyTorch ni abrir la app.")
    args = parser.parse_args()
    if args.dataset_only and args.skip_dataset:
        parser.error("--dataset-only no se puede combinar con --skip-dataset")
    check_python()
    if not args.dataset_only:
        check_prerequisites()
    if not args.skip_dataset:
        prepare_dataset(PROJECT / "minecraft_biomas", args.dataset_source, report=say)
    if args.dataset_only:
        return 0
    (ROOT / "OutPut").mkdir(parents=True, exist_ok=True)
    index, want_cuda = select_torch_index("cpu" if args.cpu else "cuda" if args.cuda else "auto")
    signature = setup_signature(index)

    python = env_python()
    if not python.is_file():
        say("[1/3] Creando el entorno local .venv ...")
        venv.EnvBuilder(with_pip=True).create(ENV_DIR)
    else:
        say("[1/3] Reutilizando .venv existente.")

    check_environment_tk(python)
    try:
        previous = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    healthy = environment_is_healthy(python, want_cuda)
    if args.reinstall or not healthy or previous.get("signature") != signature:
        say("[2/3] Instalando dependencias (primera ejecución: puede descargar varios GB)...")
        install(python, index)
        if not environment_is_healthy(python, want_cuda):
            raise SetupError(
                "La instalación terminó, pero no se puede cargar el entorno correctamente. "
                "Revisa el mensaje anterior, actualiza el controlador NVIDIA "
                "o prueba 'python setup.py --cpu'."
            )
        STATE_FILE.write_text(
            json.dumps({"signature": signature, "index": index}, indent=2),
            encoding="utf-8",
        )
    else:
        say("[2/3] Todas las dependencias están listas; no hay que reinstalar.")

    if args.install_only:
        say("[3/3] Instalación terminada. Abre la app con: python setup.py")
        return 0
    say("[3/3] Abriendo la aplicación de Minecraft Biomas ...")
    return subprocess.run([str(python), str(APP)], cwd=PROJECT, check=False).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SetupError, DatasetError, subprocess.CalledProcessError, OSError) as exc:
        say(f"\nERROR: {exc}")
        sys.exit(1)


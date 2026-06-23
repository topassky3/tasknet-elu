#!/usr/bin/env python3
"""
00_phase0_check.py — Verificacion de la Fase 0 de TaskNet-ELU.

Comprueba que la estructura minima del repositorio existe, que el paquete
src/tasknet_elu es importable y que las dependencias principales estan
instaladas. NO ejecuta ningun experimento.

Uso:
    python scripts/00_phase0_check.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Raiz del proyecto = carpeta padre de este script (.../tasknet-elu/scripts -> tasknet-elu)
ROOT = Path(__file__).resolve().parent.parent

# ---- 1. Carpetas y archivos minimos que deben existir ----
REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "scripts",
    "src/tasknet_elu",
    "results/tables",
    "results/figures",
    "reports",
    "paper",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "scripts/00_phase0_check.py",
    "src/tasknet_elu/__init__.py",
    "src/tasknet_elu/datasets.py",
    "src/tasknet_elu/models.py",
    "src/tasknet_elu/receivers.py",
    "src/tasknet_elu/representations.py",
    "src/tasknet_elu/serialize.py",
    "src/tasknet_elu/metrics.py",
    "src/tasknet_elu/ssp.py",
    "src/tasknet_elu/plots.py",
]

# Dependencias principales (import_name, paquete_pip)
CORE_DEPS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
]
# Dependencias pesadas: se reportan, pero su ausencia NO bloquea la Fase 0.
HEAVY_DEPS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
]


def check_dirs() -> list[str]:
    return [d for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]


def check_files() -> list[str]:
    return [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]


def check_package_importable() -> str | None:
    """Devuelve None si el paquete importa; si no, el mensaje de error."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        pkg = importlib.import_module("tasknet_elu")
        _ = pkg.__version__  # noqa: F841
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def check_deps(deps: list[tuple[str, str]]) -> list[str]:
    """Devuelve la lista de paquetes pip faltantes."""
    missing = []
    for import_name, pip_name in deps:
        try:
            importlib.import_module(import_name)
        except Exception:  # noqa: BLE001
            missing.append(pip_name)
    return missing


def main() -> int:
    print("==> Verificando Fase 0 TaskNet-ELU...")
    print(f"Raiz del proyecto: {ROOT}")

    problems: list[str] = []

    missing_dirs = check_dirs()
    if missing_dirs:
        problems.append("Faltan carpetas: " + ", ".join(missing_dirs))
    else:
        print("Carpetas base: OK")

    missing_files = check_files()
    if missing_files:
        problems.append("Faltan archivos: " + ", ".join(missing_files))
    else:
        print("Archivos base: OK")

    pkg_err = check_package_importable()
    if pkg_err:
        problems.append("El paquete tasknet_elu no importa: " + pkg_err)
    else:
        print("Paquete tasknet_elu: importable")

    missing_core = check_deps(CORE_DEPS)
    if missing_core:
        problems.append("Faltan dependencias principales: " + ", ".join(missing_core))
    else:
        print("Dependencias principales: OK")

    # torch/torchvision: informativo, no bloqueante en Fase 0
    missing_heavy = check_deps(HEAVY_DEPS)
    if missing_heavy:
        print("Aviso: faltan (necesarias para la Fase 1, no para la Fase 0): "
              + ", ".join(missing_heavy))
    else:
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA disponible: {torch.cuda.is_available()}")
        except Exception:  # noqa: BLE001
            pass

    print("-" * 56)
    if problems:
        print("Fase 0 INCOMPLETA. Corrige lo siguiente:")
        for p in problems:
            print("  - " + p)
        return 1

    print("Fase 0 OK. Estructura, archivos base y entorno listos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

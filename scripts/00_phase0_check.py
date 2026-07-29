#!/usr/bin/env python3
"""Validate the current TaskNet-ELU repository structure without running experiments."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = [
    "scripts",
    "results/tables",
    "results/figures",
    "reports",
    "paper",
    "docs",
    "release",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "LICENSE",
    "LICENSES.md",
    "CITATION.cff",
    ".zenodo.json",
    "CHANGELOG.md",
    "scripts/00_phase0_check.py",
    "scripts/01_utility_per_bit_fashion_mnist.py",
    "scripts/02_learned_vs_classic.py",
    "scripts/03_cifar10_validation_v3.py",
    "scripts/04_elu_threshold_policy.py",
    "scripts/05_verify_release.py",
    "paper/TaskNet-ELU-preprint-v0.1.0.pdf",
    "results/tables/utility_per_byte_fashion_mnist.csv",
    "results/tables/learned_vs_classic_fashion_mnist.csv",
    "results/tables/cifar10_learned_vs_classic.csv",
    "results/tables/elu_policy_comparison.csv",
    "results/figures/accuracy_vs_bytes.png",
    "results/figures/phase2_learned_vs_classic_frontier.png",
    "results/figures/phase3_accuracy_vs_bytes.png",
    "results/figures/phase4_traffic_utility.png",
]

DEPENDENCIES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "torch",
    "torchvision",
    "tqdm",
    "Pillow",
]


def main() -> int:
    print("==> TaskNet-ELU v0.1.0: verificación estructural")
    print(f"Raíz: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Plataforma: {platform.platform()}")

    problems: list[str] = []

    for directory in REQUIRED_DIRS:
        if not (ROOT / directory).is_dir():
            problems.append(f"Falta carpeta obligatoria: {directory}")

    for filename in REQUIRED_FILES:
        if not (ROOT / filename).is_file():
            problems.append(f"Falta archivo obligatorio: {filename}")

    print("\nDependencias detectadas:")
    for distribution in DEPENDENCIES:
        try:
            version = importlib.metadata.version(distribution)
            print(f"  {distribution}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {distribution}: no instalada")

    print("-" * 64)
    if problems:
        print("Verificación INCOMPLETA:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Verificación estructural OK. No se ejecutaron experimentos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

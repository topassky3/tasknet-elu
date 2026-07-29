#!/usr/bin/env python3
"""Verify TaskNet-ELU v0.1.0 release metadata and artifact integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "release" / "MANIFEST.sha256"
PAPER_PARTS = [
    f"paper/TaskNet-ELU-preprint-v0.1.0-part-{index:02d}.tex"
    for index in range(1, 7)
]

REQUIRED = [
    "README.md",
    "LICENSE",
    "LICENSES.md",
    "CITATION.cff",
    ".zenodo.json",
    "CHANGELOG.md",
    "docs/release-audit-v0.1.0.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ZENODO_PUBLISHING.md",
    "release/RELEASE_NOTES_v0.1.0.md",
    "paper/TaskNet-ELU-preprint-v0.1.0.tex",
    "paper/TaskNet-ELU-preprint-v0.1.0.pdf",
    *PAPER_PARTS,
    "scripts/00_phase0_check.py",
    "scripts/01_utility_per_bit_fashion_mnist.py",
    "scripts/02_learned_vs_classic.py",
    "scripts/03_cifar10_validation_v3.py",
    "scripts/04_elu_threshold_policy.py",
    "results/tables/utility_per_byte_fashion_mnist.csv",
    "results/tables/learned_vs_classic_fashion_mnist.csv",
    "results/tables/cifar10_learned_vs_classic.csv",
    "results/tables/elu_policy_comparison.csv",
]

ARTIFACT_DIRS = ["paper", "results/tables", "results/figures", "scripts", "reports"]
EXCLUDED_SUFFIXES = {".aux", ".log", ".out", ".toc", ".synctex.gz"}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"ERROR: {message}")


def validate_metadata(errors: list[str]) -> None:
    try:
        metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f".zenodo.json inválido: {exc}", errors)
        return

    for key in (
        "title",
        "description",
        "creators",
        "version",
        "upload_type",
        "publication_type",
        "license",
    ):
        if not metadata.get(key):
            fail(f"Falta el campo Zenodo: {key}", errors)

    if metadata.get("version") != "0.1.0":
        fail("La versión de .zenodo.json no es 0.1.0", errors)
    if metadata.get("publication_type") != "preprint":
        fail("Zenodo debe declarar publication_type=preprint", errors)

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for token in (
        "cff-version: 1.2.0",
        "version: 0.1.0",
        "family-names: Orozco",
        "given-names: Juan Felipe",
    ):
        if token not in cff:
            fail(f"CITATION.cff no contiene: {token}", errors)

    suspicious_doi = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
    for path in (ROOT / "README.md", ROOT / "CITATION.cff", ROOT / ".zenodo.json"):
        if path.exists() and suspicious_doi.search(path.read_text(encoding="utf-8")):
            fail(f"Se detectó un DOI antes de la publicación: {path.relative_to(ROOT)}", errors)


def validate_paper_source(errors: list[str]) -> None:
    wrapper = ROOT / "paper/TaskNet-ELU-preprint-v0.1.0.tex"
    if not wrapper.exists():
        return
    text = wrapper.read_text(encoding="utf-8")
    for part in PAPER_PARTS:
        if part not in text:
            fail(f"La fuente principal no incluye {part}", errors)

    combined = "".join((ROOT / part).read_text(encoding="utf-8") for part in PAPER_PARTS)
    for token in (
        r"\documentclass",
        "TaskNet-ELU",
        "Versi\\'on 0.1.0",
        "Preprint no revisado por pares",
        r"\end{document}",
    ):
        if token not in combined:
            fail(f"La fuente LaTeX no contiene el marcador esperado: {token}", errors)


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(keys)


def find_row(rows: list[dict[str, str]], value: str, *columns: str) -> dict[str, str]:
    for row in rows:
        if any(row.get(column) == value for column in columns):
            return row
    raise KeyError(value)


def close(actual: float, expected: float, tolerance: float = 0.01) -> bool:
    return abs(actual - expected) <= tolerance


def validate_claims(errors: list[str]) -> None:
    try:
        phase1 = read_csv("results/tables/utility_per_byte_fashion_mnist.csv")
        original = find_row(phase1, "original", "representation")
        embedding = find_row(phase1, "embedding", "representation")
        if not close(as_float(original, "bytes"), 512.4, 1.0):
            fail("Bytes del original Fashion-MNIST no coinciden con el preprint", errors)
        if not close(as_float(original, "accuracy"), 0.8895, 0.01):
            fail("Exactitud del original Fashion-MNIST no coincide", errors)
        if not close(as_float(embedding, "bytes"), 50.4, 1.0):
            fail("Bytes del embedding Fashion-MNIST no coinciden", errors)
        if not close(as_float(embedding, "accuracy"), 0.8925, 0.01):
            fail("Exactitud del embedding Fashion-MNIST no coincide", errors)
    except Exception as exc:
        fail(f"No fue posible validar Fase 1: {exc}", errors)

    try:
        phase3 = read_csv("results/tables/cifar10_learned_vs_classic.csv")
        embedding = find_row(phase3, "embedding", "representation")
        if not close(as_float(embedding, "bytes"), 59.8, 1.0):
            fail("Bytes del embedding CIFAR-10 no coinciden", errors)
        if not close(as_float(embedding, "accuracy"), 0.7200, 0.01):
            fail("Exactitud del embedding CIFAR-10 no coincide", errors)
    except Exception as exc:
        fail(f"No fue posible validar Fase 3: {exc}", errors)

    try:
        phase4 = read_csv("results/tables/elu_policy_comparison.csv")
        elu_half = next(
            row
            for row in phase4
            if row.get("policy") == "elu" and "0.500" in str(row.get("param", ""))
        )
        conf_half = next(
            row
            for row in phase4
            if row.get("policy") == "confidence" and "0.500" in str(row.get("param", ""))
        )
        if as_float(elu_half, "utility") <= as_float(conf_half, "utility"):
            fail("ELU q=0.5 no supera a confianza q=0.5", errors)
        elu_rows = [row for row in phase4 if row.get("policy") == "elu"]
        if max(as_float(row, "utility") for row in elu_rows) < 0.86:
            fail("No se encontró un punto ELU compatible con el resultado principal", errors)
    except Exception as exc:
        fail(f"No fue posible validar Fase 4: {exc}", errors)


def artifact_files() -> list[Path]:
    files: list[Path] = []
    for directory in ARTIFACT_DIRS:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(str(path).endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
                continue
            files.append(path)

    for name in (
        "README.md",
        "LICENSE",
        "LICENSES.md",
        "CITATION.cff",
        ".zenodo.json",
        "CHANGELOG.md",
    ):
        path = ROOT / name
        if path.exists():
            files.append(path)

    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_text() -> str:
    return "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in artifact_files()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            fail(f"Falta artefacto obligatorio: {relative}", errors)

    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    if "TaskNet_ELU_preprint_v10" in readme:
        fail("README conserva una ruta obsoleta del paper", errors)
    if "no localizó todavía" in readme or "permanece bloqueada" in readme:
        fail("README contiene texto interno de auditoría", errors)

    validate_metadata(errors)
    validate_paper_source(errors)
    validate_claims(errors)

    expected = manifest_text()
    if args.write_manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(expected, encoding="utf-8")
        print(f"Manifiesto escrito: {MANIFEST.relative_to(ROOT)}")
    elif MANIFEST.exists():
        current = MANIFEST.read_text(encoding="utf-8")
        if current != expected:
            fail("release/MANIFEST.sha256 está desactualizado", errors)
    else:
        fail("Falta release/MANIFEST.sha256; ejecute --write-manifest", errors)

    if errors:
        print(f"Verificación fallida con {len(errors)} problema(s).")
        return 1

    print("Release v0.1.0 verificada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

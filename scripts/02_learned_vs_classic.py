#!/usr/bin/env python3
"""
02_learned_vs_classic.py — Fase 2 de TaskNet-ELU.

Compara representaciones APRENDIDAS (embedding cuantizado, etiqueta) contra
representaciones de COMPRESION CLASICA (PNG, JPEG a varias calidades, resize+JPEG)
en Fashion-MNIST, todas evaluadas con RECEPTOR INDEPENDIENTE y bytes homogeneos.

Pregunta de la fase:
    El embedding cuantizado, ¿domina a JPEG/PNG en utilidad por byte
    cuando se igualan las condiciones de medicion?

Protocolo heredado de Fase 1 v0.2:
  - receptor independiente por representacion accionable;
  - la etiqueta se mide contra verdad de terreno;
  - el silencio usa clase mayoritaria;
  - el embedding se cuantiza a uint8 y se reconstruye antes de evaluar;
  - se mide lo que realmente viajaria;
  - bytes explicitos por representacion;
  - SSP separado en: any, decision, semantic, visual;
  - latencia = inferencia del receptor por muestra, NO latencia de red.

Nuevo en Fase 2:
  - JPEG q90/q50/q20/q10/q5;
  - PNG;
  - resize16+JPEG;
  - compression_family y quality;
  - veredicto de dominancia mas honesto:
      * no deja ganar al embedding si no conserva utilidad dentro de epsilon;
      * compara clasicos dentro de epsilon cuando existen;
      * reporta tambien lectura Pareto.

Salidas:
    results/tables/learned_vs_classic_fashion_mnist.csv
    results/figures/phase2_accuracy_vs_bytes.png
    results/figures/phase2_utility_per_byte.png
    results/figures/phase2_learned_vs_classic_frontier.png
    reports/informe_fase_2.md
    reports/cierre_fase_2.md

Uso:
    python scripts/02_learned_vs_classic.py --epochs 1 --limit-train 5000 --limit-test 1000 --seeds 1
    python scripts/02_learned_vs_classic.py --epochs 3 --limit-train 15000 --limit-test 3000 --seeds 2
    python scripts/02_learned_vs_classic.py --epochs 5 --seeds 3
"""

from __future__ import annotations

import argparse
import gzip
import io
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EPS_SSP = 0.03
SAVINGS_MIN = 0.80
EMBED_DIM = 32
NUM_CLASSES = 10
BYTE_SAMPLE_LIMIT = 500
LATENCY_SAMPLE_LIMIT = 1000
JPEG_QUALITIES = [90, 50, 20, 10, 5]


# ==========================================================
# Utilidades generales
# ==========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dirs() -> tuple[Path, Path, Path]:
    tables = ROOT / "results" / "tables"
    figs = ROOT / "results" / "figures"
    reports = ROOT / "reports"

    for d in (tables, figs, reports):
        d.mkdir(parents=True, exist_ok=True)

    return tables, figs, reports


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ==========================================================
# Carga de datos
# ==========================================================

def load_fashion_mnist(limit_train: int, limit_test: int, seed: int):
    """
    Devuelve:
        Xtr, ytr, Xte, yte, source

    X:
        shape (N, 28, 28), float32, rango [0,1]
    """
    try:
        import torchvision
        import torchvision.transforms as T

        tf = T.Compose([T.ToTensor()])

        tr = torchvision.datasets.FashionMNIST(
            root=str(ROOT / "data" / "raw"),
            train=True,
            download=True,
            transform=tf,
        )

        te = torchvision.datasets.FashionMNIST(
            root=str(ROOT / "data" / "raw"),
            train=False,
            download=True,
            transform=tf,
        )

        Xtr = tr.data.numpy().astype("float32") / 255.0
        ytr = tr.targets.numpy().astype("int64")
        Xte = te.data.numpy().astype("float32") / 255.0
        yte = te.targets.numpy().astype("int64")
        source = "fashion_mnist"

    except Exception as exc:
        print(
            f"[AVISO] No se pudo cargar Fashion-MNIST ({type(exc).__name__}: {exc}).\n"
            "        Se usa dataset SINTETICO solo para validar la mecanica.\n"
            "        No uses esos numeros como resultado cientifico real."
        )
        rng = np.random.default_rng(seed)
        Xtr, ytr = synthetic_dataset(6000, rng)
        Xte, yte = synthetic_dataset(1500, rng)
        source = "synthetic"

    if limit_train and limit_train > 0:
        Xtr = Xtr[:limit_train]
        ytr = ytr[:limit_train]

    if limit_test and limit_test > 0:
        Xte = Xte[:limit_test]
        yte = yte[:limit_test]

    return Xtr, ytr, Xte, yte, source


def synthetic_dataset(n: int, rng: np.random.Generator):
    """
    Dataset sintetico 28x28 con 10 clases separables.
    Sirve solo para validar el pipeline si Fashion-MNIST no descarga.
    """
    y = rng.integers(0, NUM_CLASSES, size=n)
    X = np.zeros((n, 28, 28), dtype="float32")

    centers = [(7 + 6 * (c % 3), 7 + 6 * (c // 3)) for c in range(NUM_CLASSES)]
    yy, xx = np.mgrid[0:28, 0:28]

    for i in range(n):
        cy, cx = centers[int(y[i])]
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 22.0)
        noise = 0.05 * rng.standard_normal((28, 28))
        X[i] = (blob + noise).clip(0, 1)

    return X.astype("float32"), y.astype("int64")


# ==========================================================
# Representaciones visuales y serializacion
# ==========================================================

def uint8_array(X: np.ndarray) -> np.ndarray:
    return (np.clip(X, 0, 1) * 255).round().astype("uint8")


def resize_batch_pil(X: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    out = []
    resample = getattr(Image, "Resampling", Image).BILINEAR

    for img in X:
        im = Image.fromarray(uint8_array(img), mode="L")
        im = im.resize((size, size), resample)
        out.append(np.asarray(im).astype("float32") / 255.0)

    return np.stack(out).astype("float32")


def jpeg_roundtrip(img01: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """
    Comprime a JPEG y reconstruye.

    Devuelve:
        imagen_reconstruida en [0,1], bytes del payload JPEG
    """
    from PIL import Image

    im = Image.fromarray(uint8_array(img01), mode="L")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    payload = buf.getvalue()

    recon = np.asarray(Image.open(io.BytesIO(payload))).astype("float32") / 255.0
    return recon, len(payload)


def png_roundtrip(img01: np.ndarray) -> tuple[np.ndarray, int]:
    """
    PNG sin perdida.

    Devuelve:
        imagen_reconstruida en [0,1], bytes del payload PNG
    """
    from PIL import Image

    im = Image.fromarray(uint8_array(img01), mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    payload = buf.getvalue()

    recon = np.asarray(Image.open(io.BytesIO(payload))).astype("float32") / 255.0
    return recon, len(payload)


def jpeg_batch(X: np.ndarray, quality: int) -> tuple[np.ndarray, list[int]]:
    recons = []
    sizes = []

    for img in X:
        r, b = jpeg_roundtrip(img, quality)
        recons.append(r)
        sizes.append(b)

    return np.stack(recons).astype("float32"), sizes


def png_batch(X: np.ndarray) -> tuple[np.ndarray, list[int]]:
    recons = []
    sizes = []

    for img in X:
        r, b = png_roundtrip(img)
        recons.append(r)
        sizes.append(b)

    return np.stack(recons).astype("float32"), sizes


def gzip_len(payload: bytes) -> int:
    return len(gzip.compress(payload, compresslevel=6))


def bytes_uint8_gzip(arr01: np.ndarray) -> int:
    return gzip_len(uint8_array(arr01).tobytes())


def bytes_label_gzip() -> int:
    return gzip_len(np.array([0], dtype="uint8").tobytes())


def mean_bytes(sizes: list[int]) -> float:
    sample = sizes[:BYTE_SAMPLE_LIMIT]
    return float(np.mean(sample))


# ==========================================================
# Cuantizador del embedding
# ==========================================================

@dataclass
class Quantizer:
    minimum: np.ndarray
    maximum: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "Quantizer":
        return Quantizer(
            minimum=X.min(axis=0).astype("float32"),
            maximum=X.max(axis=0).astype("float32"),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = self.maximum - self.minimum
        denom = np.where(denom < 1e-8, 1.0, denom)
        q = np.round((X - self.minimum) / denom * 255.0)
        return np.clip(q, 0, 255).astype("uint8")

    def inverse_transform(self, Q: np.ndarray) -> np.ndarray:
        denom = self.maximum - self.minimum
        denom = np.where(denom < 1e-8, 1.0, denom)
        return (Q.astype("float32") / 255.0) * denom + self.minimum


# ==========================================================
# Backend torch
# ==========================================================

def make_image_clf_torch(in_hw: int, Xtr: np.ndarray, ytr: np.ndarray, epochs: int, seed: int):
    import torch
    import torch.nn as nn

    set_seed(seed)

    class CNN(nn.Module):
        def __init__(self, hw: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            flat = 32 * (hw // 4) * (hw // 4)
            self.embed = nn.Linear(flat, EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x, return_embed: bool = False):
            z = self.features(x).flatten(1)
            e = torch.relu(self.embed(z))
            out = self.head(e)
            if return_embed:
                return out, e
            return out

    model = CNN(in_hw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(Xtr[:, None, :, :], dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)

    model.train()
    batch = 128

    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    return model


def predict_image_torch(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X[:, None, :, :], dtype=torch.float32))

    return out.argmax(1).numpy()


def extract_embedding_torch(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        _, e = model(
            torch.tensor(X[:, None, :, :], dtype=torch.float32),
            return_embed=True,
        )

    return e.numpy().astype("float32")


def make_vector_clf_torch(Xtr: np.ndarray, ytr: np.ndarray, epochs: int, seed: int):
    import torch
    import torch.nn as nn

    set_seed(seed)

    class MLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, NUM_CLASSES),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)

    model.train()
    batch = 128

    for _ in range(max(epochs, 3)):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    return model


def predict_vector_torch(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))

    return out.argmax(1).numpy()


# ==========================================================
# Backend sklearn de respaldo
# ==========================================================

def make_image_clf_sklearn(Xtr: np.ndarray, ytr: np.ndarray, epochs: int, seed: int):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=40 + 10 * epochs,
        random_state=seed,
    )
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return clf


def predict_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X.reshape(len(X), -1))


def make_vector_clf_sklearn(Xtr: np.ndarray, ytr: np.ndarray, epochs: int, seed: int):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=40 + 10 * epochs,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    return clf


def predict_vector_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


# ==========================================================
# Evaluacion
# ==========================================================

def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def timed_predict(predict_fn, model, Xeval: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Mide SOLO el tiempo de inferencia del receptor por muestra.

    No incluye:
        - entrenamiento;
        - compresion;
        - serializacion;
        - transmision de red.

    Es proxy de computo de recepcion.
    """
    n = min(LATENCY_SAMPLE_LIMIT, len(Xeval))

    t0 = now_ms()
    _ = predict_fn(model, Xeval[:n])
    latency_ms = (now_ms() - t0) / max(1, n)

    pred_full = predict_fn(model, Xeval)
    return pred_full, float(latency_ms)


def add_row(
    rows: list[dict[str, Any]],
    representation: str,
    representation_group: str,
    compression_family: str,
    quality: str,
    acc: float,
    bytes_value: float,
    raw: float,
    method: str,
    receiver: str,
    latency_ms: float,
) -> None:
    rows.append(
        {
            "representation": representation,
            "representation_group": representation_group,
            "compression_family": compression_family,
            "quality": quality,
            "accuracy": float(acc),
            "bytes": float(bytes_value),
            "raw": float(raw),
            "method": method,
            "receiver": receiver,
            "latency_ms": float(latency_ms),
        }
    )


def evaluate_all(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    epochs: int,
    seed: int,
):
    """
    Devuelve:
        rows, backend

    rows:
        lista de diccionarios, uno por representacion.
    """
    torch_ok = has_torch()
    rows: list[dict[str, Any]] = []

    if torch_ok:
        pfn_img = predict_image_torch
        pfn_vec = predict_vector_torch

        tx = make_image_clf_torch(28, Xtr, ytr, epochs, seed)
        y_label = predict_image_torch(tx, Xte)
        Etr = extract_embedding_torch(tx, Xtr)
        Ete = extract_embedding_torch(tx, Xte)

    else:
        pfn_img = predict_sklearn
        pfn_vec = predict_vector_sklearn

        tx = make_image_clf_sklearn(Xtr, ytr, epochs, seed)
        y_label = predict_sklearn(tx, Xte)

        rng = np.random.default_rng(seed)
        W = rng.standard_normal((28 * 28, EMBED_DIM)).astype("float32")
        Etr = np.maximum(Xtr.reshape(len(Xtr), -1) @ W, 0)
        Ete = np.maximum(Xte.reshape(len(Xte), -1) @ W, 0)

    # ------------------------------------------------------
    # Original PNG / baseline visual
    # ------------------------------------------------------
    if torch_ok:
        rx_full = make_image_clf_torch(28, Xtr, ytr, epochs, seed + 100)
    else:
        rx_full = make_image_clf_sklearn(Xtr, ytr, epochs, seed + 100)

    pred_full, lat_full = timed_predict(pfn_img, rx_full, Xte)
    _, png_sizes = png_batch(Xte)

    add_row(
        rows=rows,
        representation="original_png",
        representation_group="visual",
        compression_family="png",
        quality="lossless",
        acc=accuracy(pred_full, yte),
        bytes_value=mean_bytes(png_sizes),
        raw=float(28 * 28),
        method="png",
        receiver="independent_full_image",
        latency_ms=lat_full,
    )

    # ------------------------------------------------------
    # JPEG a varias calidades
    # ------------------------------------------------------
    for q in JPEG_QUALITIES:
        Jtr, _ = jpeg_batch(Xtr, q)
        Jte, jsizes = jpeg_batch(Xte, q)

        if torch_ok:
            rx_jpeg = make_image_clf_torch(28, Jtr, ytr, epochs, seed + q)
        else:
            rx_jpeg = make_image_clf_sklearn(Jtr, ytr, epochs, seed + q)

        pred_jpeg, lat_jpeg = timed_predict(pfn_img, rx_jpeg, Jte)

        add_row(
            rows=rows,
            representation=f"jpeg_q{q}",
            representation_group="visual",
            compression_family="jpeg",
            quality=str(q),
            acc=accuracy(pred_jpeg, yte),
            bytes_value=mean_bytes(jsizes),
            raw=float(28 * 28),
            method="jpeg",
            receiver="independent_jpeg",
            latency_ms=lat_jpeg,
        )

    # ------------------------------------------------------
    # Resize 16x16 + JPEG q50
    # ------------------------------------------------------
    X16tr = resize_batch_pil(Xtr, 16)
    X16te = resize_batch_pil(Xte, 16)

    RJtr, _ = jpeg_batch(X16tr, 50)
    RJte, rjsizes = jpeg_batch(X16te, 50)

    if torch_ok:
        rx_rj = make_image_clf_torch(16, RJtr, ytr, epochs, seed + 700)
    else:
        rx_rj = make_image_clf_sklearn(RJtr, ytr, epochs, seed + 700)

    pred_rj, lat_rj = timed_predict(pfn_img, rx_rj, RJte)

    add_row(
        rows=rows,
        representation="resize16_jpeg_q50",
        representation_group="visual",
        compression_family="resize",
        quality="16+q50",
        acc=accuracy(pred_rj, yte),
        bytes_value=mean_bytes(rjsizes),
        raw=float(16 * 16),
        method="jpeg",
        receiver="independent_resize_jpeg",
        latency_ms=lat_rj,
    )

    # ------------------------------------------------------
    # Resize 8x8 puro + uint8+gzip
    # ------------------------------------------------------
    X8tr = resize_batch_pil(Xtr, 8)
    X8te = resize_batch_pil(Xte, 8)

    if torch_ok:
        rx_8 = make_image_clf_torch(8, X8tr, ytr, epochs, seed + 800)
    else:
        rx_8 = make_image_clf_sklearn(X8tr, ytr, epochs, seed + 800)

    pred_8, lat_8 = timed_predict(pfn_img, rx_8, X8te)

    s8 = [
        bytes_uint8_gzip(X8te[i])
        for i in range(min(BYTE_SAMPLE_LIMIT, len(X8te)))
    ]

    add_row(
        rows=rows,
        representation="resize_8x8",
        representation_group="visual",
        compression_family="resize",
        quality="8",
        acc=accuracy(pred_8, yte),
        bytes_value=float(np.mean(s8)),
        raw=float(8 * 8),
        method="uint8+gzip",
        receiver="independent_resize",
        latency_ms=lat_8,
    )

    # ------------------------------------------------------
    # Embedding cuantizado + receptor independiente
    # ------------------------------------------------------
    qz = Quantizer.fit(Etr)
    Etr_q = qz.transform(Etr)
    Ete_q = qz.transform(Ete)

    Etr_rx = qz.inverse_transform(Etr_q)
    Ete_rx = qz.inverse_transform(Ete_q)

    if torch_ok:
        rx_emb = make_vector_clf_torch(Etr_rx, ytr, epochs, seed + 900)
    else:
        rx_emb = make_vector_clf_sklearn(Etr_rx, ytr, epochs, seed + 900)

    pred_emb, lat_emb = timed_predict(pfn_vec, rx_emb, Ete_rx)

    s_emb = [
        gzip_len(Ete_q[i].tobytes())
        for i in range(min(BYTE_SAMPLE_LIMIT, len(Ete_q)))
    ]

    add_row(
        rows=rows,
        representation="embedding",
        representation_group="semantic",
        compression_family="learned",
        quality="uint8",
        acc=accuracy(pred_emb, yte),
        bytes_value=float(np.mean(s_emb)),
        raw=float(EMBED_DIM),
        method="uint8+gzip",
        receiver="independent_embedding_quantized",
        latency_ms=lat_emb,
    )

    # ------------------------------------------------------
    # Label
    # ------------------------------------------------------
    add_row(
        rows=rows,
        representation="label",
        representation_group="decision",
        compression_family="decision",
        quality="argmax",
        acc=accuracy(y_label, yte),
        bytes_value=float(bytes_label_gzip()),
        raw=1.0,
        method="uint8+gzip",
        receiver="ground_truth",
        latency_ms=0.0,
    )

    # ------------------------------------------------------
    # Silence
    # ------------------------------------------------------
    _, counts = np.unique(yte, return_counts=True)
    acc_silence = float(counts.max() / counts.sum())

    add_row(
        rows=rows,
        representation="silence",
        representation_group="silence",
        compression_family="silence",
        quality="none",
        acc=acc_silence,
        bytes_value=0.0,
        raw=0.0,
        method="none",
        receiver="majority_class",
        latency_ms=0.0,
    )

    backend = "torch" if torch_ok else "sklearn"
    return rows, backend


# ==========================================================
# Corridas y agregacion
# ==========================================================

def run_once(args: argparse.Namespace, seed: int):
    set_seed(seed)

    Xtr, ytr, Xte, yte, source = load_fashion_mnist(
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        seed=seed,
    )

    rows, backend = evaluate_all(
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        epochs=args.epochs,
        seed=seed,
    )

    return rows, source, backend


def aggregate(all_runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Promedia accuracy y latencia entre semillas.
    Bytes se toman de la primera corrida para cada representacion.
    """
    by_rep: dict[str, list[dict[str, Any]]] = {}

    for run in all_runs:
        for row in run:
            by_rep.setdefault(row["representation"], []).append(row)

    aggregated = []

    for rep, rows in by_rep.items():
        base = dict(rows[0])

        accs = np.array([r["accuracy"] for r in rows], dtype="float64")
        lats = np.array([r["latency_ms"] for r in rows], dtype="float64")

        base["accuracy"] = float(accs.mean())
        base["accuracy_std"] = float(accs.std(ddof=0))
        base["latency_ms"] = float(lats.mean())
        base["latency_ms_std"] = float(lats.std(ddof=0))

        aggregated.append(base)

    return aggregated


def build_table(agg: list[dict[str, Any]], seed_count: int):
    original_rows = [r for r in agg if r["representation"] == "original_png"]

    if not original_rows:
        raise RuntimeError("No existe original_png para calcular ahorro de bytes.")

    b_original = original_rows[0]["bytes"]
    u_max = max(r["accuracy"] for r in agg)

    rows = []

    for r in agg:
        b = float(r["bytes"])
        u = float(r["accuracy"])
        is_silence = r["representation_group"] == "silence"

        utility_per_byte = np.nan if is_silence or b == 0 else u / b
        byte_savings = 0.0 if r["representation"] == "original_png" else max(0.0, 1.0 - b / b_original)

        within_epsilon = u >= (u_max - EPS_SSP)
        enough_savings = byte_savings >= SAVINGS_MIN

        rows.append(
            {
                "representation": r["representation"],
                "representation_group": r["representation_group"],
                "compression_family": r["compression_family"],
                "quality": r["quality"],
                "bytes_raw": round(float(r["raw"]), 4),
                "bytes": round(b, 4),
                "bits": round(8.0 * b, 4),
                "serialization_method": r["method"],
                "accuracy": round(u, 6),
                "accuracy_std": round(float(r["accuracy_std"]), 6),
                "latency_ms": round(float(r["latency_ms"]), 6),
                "latency_ms_std": round(float(r["latency_ms_std"]), 6),
                "utility_per_byte": np.nan if np.isnan(utility_per_byte) else round(utility_per_byte, 8),
                "byte_savings": round(byte_savings, 6),
                "within_epsilon": "yes" if within_epsilon else "no",
                "enough_savings": "yes" if enough_savings else "no",
                "is_ssp_any": "no",
                "is_ssp_decision": "no",
                "is_ssp_semantic": "no",
                "is_ssp_visual": "no",
                "receiver": r["receiver"],
                "is_silence_baseline": "yes" if is_silence else "no",
                "seed_count": seed_count,
            }
        )

    df = pd.DataFrame(rows)

    def mark(mask: pd.Series, col: str) -> None:
        candidates = df[
            mask
            & (df["is_silence_baseline"] == "no")
            & (df["within_epsilon"] == "yes")
            & (df["enough_savings"] == "yes")
        ].copy()

        if candidates.empty:
            return

        winner = candidates.sort_values("bytes", ascending=True).iloc[0]["representation"]
        df.loc[df["representation"] == winner, col] = "yes"

    mark(pd.Series([True] * len(df)), "is_ssp_any")
    mark(df["representation_group"] == "decision", "is_ssp_decision")
    mark(df["representation_group"] == "semantic", "is_ssp_semantic")
    mark(df["representation_group"] == "visual", "is_ssp_visual")

    return df, u_max


# ==========================================================
# Dominancia
# ==========================================================

def dominance_verdict(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compara embedding vs compresion clasica de forma defendible.

    Lecturas:
      1. El embedding solo puede "ganar" si tambien conserva utilidad dentro de epsilon.
      2. La comparacion de utilidad/byte se hace contra clasicos dentro de epsilon si existen.
      3. Se reporta dominancia Pareto por separado.
    """
    emb = df[df["representation"] == "embedding"]
    classic = df[df["compression_family"].isin(["jpeg", "png", "resize"])]

    out = {
        "verdict": "indeterminado",
        "emb_within_epsilon": None,
        "emb_upb": None,
        "emb_accuracy": None,
        "emb_bytes": None,
        "best_classic": None,
        "pareto": None,
        "note": "",
    }

    if emb.empty or classic.empty:
        out["note"] = "faltan embedding o representaciones clasicas"
        return out

    e = emb.iloc[0]

    emb_within = str(e["within_epsilon"]) == "yes"
    emb_upb = e["utility_per_byte"]

    out["emb_within_epsilon"] = emb_within
    out["emb_upb"] = None if pd.isna(emb_upb) else float(emb_upb)
    out["emb_accuracy"] = float(e["accuracy"])
    out["emb_bytes"] = float(e["bytes"])

    classic_valid = classic[classic["within_epsilon"] == "yes"]

    if not classic_valid.empty:
        pool = classic_valid
        out["note"] = "comparado contra clasicos dentro de epsilon"
    else:
        pool = classic
        out["note"] = "ningun clasico quedo dentro de epsilon; se reporta mejor clasico global solo como referencia"

    # Pareto
    emb_u = float(e["accuracy"])
    emb_b = float(e["bytes"])

    classic_dominates_embedding = classic[
        (classic["accuracy"] >= emb_u)
        & (classic["bytes"] <= emb_b)
        & (
            (classic["accuracy"] > emb_u)
            | (classic["bytes"] < emb_b)
        )
    ]

    embedding_dominates_classic = classic[
        (classic["accuracy"] <= emb_u)
        & (classic["bytes"] >= emb_b)
        & (
            (classic["accuracy"] < emb_u)
            | (classic["bytes"] > emb_b)
        )
    ]

    if not classic_dominates_embedding.empty:
        row = classic_dominates_embedding.sort_values("bytes").iloc[0]
        out["pareto"] = f"un clasico domina al embedding en Pareto: {row['representation']}"
    elif len(embedding_dominates_classic) == len(classic):
        out["pareto"] = "el embedding domina en Pareto a todos los clasicos"
    else:
        out["pareto"] = "no hay dominancia Pareto clara; la frontera es mixta"

    if pool.empty:
        out["verdict"] = "indeterminado"
        return out

    best_classic = pool.sort_values("utility_per_byte", ascending=False).iloc[0]

    bc_name = str(best_classic["representation"])
    bc_upb = float(best_classic["utility_per_byte"])
    bc_acc = float(best_classic["accuracy"])
    bc_bytes = float(best_classic["bytes"])
    bc_within = str(best_classic["within_epsilon"]) == "yes"

    out["best_classic"] = {
        "representation": bc_name,
        "utility_per_byte": bc_upb,
        "accuracy": bc_acc,
        "bytes": bc_bytes,
        "within_epsilon": bc_within,
    }

    # Regla importante:
    # El embedding NO puede declararse ganador si no conserva utilidad dentro de epsilon.
    if not emb_within:
        out["verdict"] = "embedding_no_conserva_utilidad"
        return out

    # Si el embedding conserva utilidad y ningun clasico queda dentro de epsilon,
    # eso ya es evidencia fuerte a favor del embedding como representacion util.
    if classic_valid.empty:
        out["verdict"] = "embedding_domina_utilidad_conservada"
        return out

    if out["emb_upb"] is None:
        out["verdict"] = "indeterminado"
    elif out["emb_upb"] > bc_upb * 1.05:
        out["verdict"] = "embedding_domina"
    elif bc_upb > out["emb_upb"] * 1.05:
        out["verdict"] = "clasico_domina"
    else:
        out["verdict"] = "empate_tecnico"

    return out


# ==========================================================
# Graficas
# ==========================================================

def make_plots(df: pd.DataFrame, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["is_silence_baseline"] == "no"].copy()

    fam_color = {
        "learned": "#145c3a",
        "jpeg": "#14345c",
        "png": "#7a4f9c",
        "resize": "#a06010",
        "decision": "#999999",
    }

    # ------------------------------------------------------
    # 1. Accuracy vs bytes
    # ------------------------------------------------------
    plt.figure(figsize=(8.5, 5.2))

    for fam, sub in d.groupby("compression_family"):
        plt.scatter(
            sub["bytes"],
            sub["accuracy"],
            s=60,
            color=fam_color.get(fam, "#333333"),
            label=fam,
            zorder=3,
        )

        for _, row in sub.iterrows():
            label = row["representation"]
            if row["is_ssp_any"] == "yes":
                label += " (SSP)"
            plt.annotate(
                label,
                (row["bytes"], row["accuracy"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

    plt.xlabel("Bytes transmitidos serializados")
    plt.ylabel("Exactitud con receptor independiente")
    plt.title("Fase 2: utilidad vs bytes — aprendido vs clasico")
    plt.grid(alpha=0.3)
    plt.legend(title="familia")
    plt.tight_layout()
    plt.savefig(figdir / "phase2_accuracy_vs_bytes.png", dpi=140)
    plt.close()

    # ------------------------------------------------------
    # 2. Utilidad por byte
    # ------------------------------------------------------
    plt.figure(figsize=(8.5, 5.2))

    d2 = d.dropna(subset=["utility_per_byte"]).sort_values("utility_per_byte", ascending=False)
    colors = [fam_color.get(f, "#333333") for f in d2["compression_family"]]

    plt.bar(d2["representation"], d2["utility_per_byte"], color=colors)
    plt.ylabel("Utilidad por byte")
    plt.title("Fase 2: utilidad por byte por representacion")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "phase2_utility_per_byte.png", dpi=140)
    plt.close()

    # ------------------------------------------------------
    # 3. Frontera learned vs classic
    # ------------------------------------------------------
    plt.figure(figsize=(8.5, 5.2))

    learned = d[d["compression_family"].isin(["learned", "decision"])].sort_values("bytes")
    classic = d[d["compression_family"].isin(["jpeg", "png", "resize"])].sort_values("bytes")

    if not classic.empty:
        plt.plot(
            classic["bytes"],
            classic["accuracy"],
            "o-",
            color="#14345c",
            label="clasico",
        )

    if not learned.empty:
        plt.plot(
            learned["bytes"],
            learned["accuracy"],
            "s--",
            color="#145c3a",
            label="aprendido/decision",
        )

    for _, row in d.iterrows():
        plt.annotate(
            row["representation"],
            (row["bytes"], row["accuracy"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("Bytes transmitidos serializados")
    plt.ylabel("Exactitud")
    plt.title("Fase 2: frontera aprendido vs clasico")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "phase2_learned_vs_classic_frontier.png", dpi=140)
    plt.close()


# ==========================================================
# Informes
# ==========================================================

def table_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def verdict_text(verdict: str) -> str:
    mapping = {
        "embedding_domina": "El EMBEDDING domina a la compresion clasica en utilidad por byte entre representaciones que conservan utilidad.",
        "embedding_domina_utilidad_conservada": "El EMBEDDING conserva utilidad dentro de epsilon y ningun clasico lo logra; se favorece al embedding como representacion util.",
        "clasico_domina": "La COMPRESION CLASICA domina al embedding en utilidad por byte entre representaciones que conservan utilidad.",
        "empate_tecnico": "EMPATE TECNICO entre embedding y compresion clasica.",
        "embedding_no_conserva_utilidad": "El embedding NO conserva utilidad dentro de epsilon; no puede declararse ganador aunque tenga buena utilidad por byte.",
        "indeterminado": "Resultado indeterminado por datos insuficientes.",
    }
    return mapping.get(verdict, "Resultado indeterminado.")


def write_reports(
    df: pd.DataFrame,
    u_max: float,
    args: argparse.Namespace,
    source: str,
    backend: str,
    reports_dir: Path,
) -> str:
    dv = dominance_verdict(df)
    verdict = dv["verdict"]
    text = verdict_text(verdict)

    lines = []
    lines.append("# Informe Fase 2 — TaskNet-ELU (aprendido vs clasico)")
    lines.append("")
    lines.append("## Configuracion")
    lines.append(f"- Fuente de datos: **{source}**")
    lines.append(f"- Backend: **{backend}**")
    lines.append(f"- Epocas: **{args.epochs}**")
    lines.append(f"- Semillas: **{args.seeds}**")
    lines.append(f"- limit_train: **{args.limit_train}**")
    lines.append(f"- limit_test: **{args.limit_test}**")
    lines.append(f"- U_max = **{u_max:.4f}**")
    lines.append(f"- Epsilon SSP = **{EPS_SSP}**")
    lines.append(f"- Ahorro minimo = **{SAVINGS_MIN:.0%}**")
    lines.append("")

    if source == "synthetic":
        lines.append("> **AVISO:** se uso dataset sintetico. Valida mecanica, no resultado real.")
        lines.append("")

    if backend == "sklearn":
        lines.append("> **AVISO:** sin torch, el embedding es proxy por proyeccion aleatoria.")
        lines.append("")

    lines.append("## Veredicto de dominancia")
    lines.append("")
    lines.append(f"**{text}**")
    lines.append("")
    lines.append(f"- Lectura de eficiencia: {dv['note']}.")
    lines.append(f"- Embedding dentro de epsilon: **{dv['emb_within_epsilon']}**")

    if dv["emb_upb"] is not None:
        lines.append(f"- Utilidad por byte del embedding: **{dv['emb_upb']:.8f}**")
        lines.append(f"- Accuracy embedding: **{dv['emb_accuracy']:.4f}**")
        lines.append(f"- Bytes embedding: **{dv['emb_bytes']:.2f}**")

    if dv["best_classic"] is not None:
        bc = dv["best_classic"]
        lines.append(
            f"- Mejor clasico considerado: **`{bc['representation']}`** "
            f"(utility/byte={bc['utility_per_byte']:.8f}, "
            f"accuracy={bc['accuracy']:.4f}, bytes={bc['bytes']:.2f}, "
            f"within_epsilon={bc['within_epsilon']})"
        )

    lines.append(f"- Lectura Pareto: {dv['pareto']}.")
    lines.append("")

    lines.append("## Resultados")
    lines.append("")
    lines.append(table_md(df))
    lines.append("")

    lines.append("## Notas de honestidad")
    lines.append("")
    lines.append(
        "- El embedding sale de un transmisor entrenado en la tarea. Por tanto, es una cota "
        "optimista. La comparacion es valida como experimento controlado, no como conclusion universal."
    )
    lines.append(
        "- JPEG en imagenes 28x28 en escala de grises no es su escenario ideal. Si pierde, no significa "
        "que JPEG sea malo; significa que en este entorno controlado no fue la mejor representacion para la tarea."
    )
    lines.append(
        "- Cuantizador del embedding: los parametros min/max se asumen compartidos de antemano entre transmisor "
        "y receptor como metadatos de calibracion. No se cuentan como bytes por muestra. Solo se cuentan los bytes "
        "del vector cuantizado que viaja en cada transmision."
    )
    lines.append(
        "- Latencia: es el tiempo de inferencia del receptor por muestra. No incluye entrenamiento, compresion, "
        "serializacion ni latencia de red."
    )
    lines.append(
        "- La utilidad por byte no basta por si sola. Por eso el veredicto exige primero conservar utilidad dentro "
        "de epsilon para declarar ganador al embedding o a un clasico."
    )

    (reports_dir / "informe_fase_2.md").write_text("\n".join(lines), encoding="utf-8")

    cierre = []
    cierre.append("# Cierre de Fase 2 — TaskNet-ELU")
    cierre.append("")
    cierre.append("## Estado")

    if source == "fashion_mnist" and backend == "torch":
        cierre.append("Fase 2 ejecutada con Fashion-MNIST real y backend torch.")
    else:
        cierre.append("Fase 2 ejecutada en modo no principal. Repetir con Fashion-MNIST real y torch.")

    cierre.append("")
    cierre.append("## Comando ejecutado")
    cierre.append(
        f"`python scripts/02_learned_vs_classic.py --epochs {args.epochs} "
        f"--seeds {args.seeds}`"
    )
    cierre.append("")
    cierre.append("## Archivos generados")
    cierre.append("- `results/tables/learned_vs_classic_fashion_mnist.csv`")
    cierre.append("- `results/figures/phase2_accuracy_vs_bytes.png`")
    cierre.append("- `results/figures/phase2_utility_per_byte.png`")
    cierre.append("- `results/figures/phase2_learned_vs_classic_frontier.png`")
    cierre.append("- `reports/informe_fase_2.md`")
    cierre.append("- `reports/cierre_fase_2.md`")
    cierre.append("")
    cierre.append("## Veredicto")
    cierre.append(f"- {text}")
    cierre.append(f"- Pareto: {dv['pareto']}")
    cierre.append("")
    cierre.append("## Decision siguiente")
    cierre.append(
        "Pasar a Fase 3 con CIFAR-10 para evaluar si el comportamiento persiste en imagenes mas complejas."
    )

    (reports_dir / "cierre_fase_2.md").write_text("\n".join(cierre), encoding="utf-8")

    return verdict


# ==========================================================
# Validacion
# ==========================================================

def validate(df: pd.DataFrame) -> None:
    required = {
        "representation",
        "representation_group",
        "compression_family",
        "quality",
        "bytes_raw",
        "bytes",
        "bits",
        "serialization_method",
        "accuracy",
        "accuracy_std",
        "latency_ms",
        "latency_ms_std",
        "utility_per_byte",
        "byte_savings",
        "within_epsilon",
        "enough_savings",
        "is_ssp_any",
        "is_ssp_decision",
        "is_ssp_semantic",
        "is_ssp_visual",
        "receiver",
        "is_silence_baseline",
        "seed_count",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas obligatorias: {sorted(missing)}")

    expected_reps = {
        "original_png",
        "jpeg_q90",
        "jpeg_q50",
        "jpeg_q20",
        "jpeg_q10",
        "jpeg_q5",
        "resize16_jpeg_q50",
        "resize_8x8",
        "embedding",
        "label",
        "silence",
    }

    found_reps = set(df["representation"].astype(str))
    missing_reps = expected_reps - found_reps

    if missing_reps:
        raise RuntimeError(f"Faltan representaciones: {sorted(missing_reps)}")

    if not ((df["accuracy"] >= 0) & (df["accuracy"] <= 1)).all():
        raise RuntimeError("accuracy fuera de [0,1]")

    if not (df["bytes"] >= 0).all():
        raise RuntimeError("bytes negativos")

    if not (df["bits"] >= 0).all():
        raise RuntimeError("bits negativos")

    if not (df["latency_ms"] >= 0).all():
        raise RuntimeError("latencias negativas")

    upb_non_na = df["utility_per_byte"].dropna()

    if not (upb_non_na >= 0).all():
        raise RuntimeError("utility_per_byte negativa")

    # Toda representacion accionable visual/semantica debe tener receptor independiente.
    actionable = df[
        df["representation_group"].isin(["visual", "semantic"])
        & (df["is_silence_baseline"] == "no")
    ]

    for _, row in actionable.iterrows():
        if not str(row["receiver"]).startswith("independent"):
            raise RuntimeError(
                f"{row['representation']} no tiene receptor independiente: {row['receiver']}"
            )

    label = df[df["representation"] == "label"].iloc[0]
    if label["receiver"] != "ground_truth":
        raise RuntimeError("label debe evaluarse contra ground_truth")

    silence = df[df["representation"] == "silence"].iloc[0]
    if silence["is_silence_baseline"] != "yes":
        raise RuntimeError("silence debe estar marcado como baseline")

    silence_upb = silence["utility_per_byte"]
    if not pd.isna(silence_upb):
        raise RuntimeError("silence debe tener utility_per_byte NaN")


# ==========================================================
# CLI
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TaskNet-ELU Fase 2: aprendido vs clasico en Fashion-MNIST."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=0, dest="limit_train")
    parser.add_argument("--limit-test", type=int, default=0, dest="limit_test")
    parser.add_argument("--seeds", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tables_dir, figs_dir, reports_dir = ensure_dirs()

    all_runs = []
    source = None
    backend = None

    for seed in range(args.seeds):
        print(f"==> Corrida semilla {seed}")
        rows, source, backend = run_once(args, seed)
        all_runs.append(rows)

    agg = aggregate(all_runs)
    df, u_max = build_table(agg, args.seeds)

    validate(df)

    csv_path = tables_dir / "learned_vs_classic_fashion_mnist.csv"
    df.to_csv(csv_path, index=False)

    make_plots(df, figs_dir)

    verdict = write_reports(
        df=df,
        u_max=u_max,
        args=args,
        source=str(source),
        backend=str(backend),
        reports_dir=reports_dir,
    )

    print("\n=== Resultados Fase 2 ===")
    print(df.to_string(index=False))
    print("")
    print(f"U_max = {u_max:.6f}")
    print(f"Veredicto = {verdict}")
    print(f"CSV = {csv_path}")
    print(f"Figura accuracy vs bytes = {figs_dir / 'phase2_accuracy_vs_bytes.png'}")
    print(f"Figura utility per byte = {figs_dir / 'phase2_utility_per_byte.png'}")
    print(f"Figura frontera = {figs_dir / 'phase2_learned_vs_classic_frontier.png'}")
    print(f"Informe = {reports_dir / 'informe_fase_2.md'}")
    print(f"Cierre = {reports_dir / 'cierre_fase_2.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
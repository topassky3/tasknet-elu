#!/usr/bin/env python3
"""
01_utility_per_bit_fashion_mnist.py — Fase 1 de TaskNet-ELU.

Experimento corregido v0.2:
- Evalua utilidad vs bytes en Fashion-MNIST.
- Usa receptor independiente para original, resize_16x16, resize_8x8 y embedding.
- La etiqueta se mide contra verdad de terreno, no contra si misma.
- El silencio usa clase mayoritaria.
- El resize es resize real con PIL, no recorte.
- El embedding se cuantiza a uint8, se serializa, se reconstruye y el receptor
  se entrena/evalua sobre la version reconstruida. Es decir: la utilidad se mide
  sobre lo que realmente viajaria.
- Se separan candidatos SSP:
    * is_ssp_any: candidato general de menor byte.
    * is_ssp_decision: etiqueta/decision final.
    * is_ssp_semantic: representacion reutilizable, por ejemplo embedding.
    * is_ssp_visual: representacion visual, por ejemplo resize.
- La latencia es proxy de computo por representacion, no latencia de red.

Salidas:
    results/tables/utility_per_bit_fashion_mnist.csv
    results/tables/utility_per_byte_fashion_mnist.csv
    results/figures/accuracy_vs_bytes.png
    results/figures/utility_per_byte.png
    reports/informe_fase_1.md
    reports/cierre_fase_1.md

Uso:
    python scripts/01_utility_per_bit_fashion_mnist.py --epochs 1 --limit-train 5000 --limit-test 1000
    python scripts/01_utility_per_bit_fashion_mnist.py --epochs 5 --seeds 3
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
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


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return a / b


# ==========================================================
# Carga de datos
# ==========================================================

def load_fashion_mnist(limit_train: int, limit_test: int, seed: int):
    """
    Devuelve Xtr, ytr, Xte, yte, source.
    X tiene shape (N, 28, 28), float32 en [0,1].
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
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]

    if limit_test and limit_test > 0:
        Xte, yte = Xte[:limit_test], yte[:limit_test]

    return Xtr, ytr, Xte, yte, source


def synthetic_dataset(n: int, rng: np.random.Generator):
    """
    Dataset sintetico 28x28 con 10 clases separables.
    Sirve solo para validar la mecanica del pipeline.
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
# Representaciones y serializacion
# ==========================================================

def resize_batch_pil(X: np.ndarray, size: int) -> np.ndarray:
    """
    Resize real usando PIL. No recorta.
    Entrada: X shape (N,H,W), float32 [0,1].
    Salida: shape (N,size,size), float32 [0,1].
    """
    from PIL import Image

    out = []
    resample = getattr(Image, "Resampling", Image).BILINEAR

    for img in X:
        arr = (np.clip(img, 0, 1) * 255).astype("uint8")
        im = Image.fromarray(arr, mode="L")
        im = im.resize((size, size), resample)
        out.append(np.asarray(im).astype("float32") / 255.0)

    return np.stack(out).astype("float32")


def uint8_array(X: np.ndarray) -> np.ndarray:
    return (np.clip(X, 0, 1) * 255).round().astype("uint8")


def gzip_len_from_bytes(payload: bytes) -> int:
    return len(gzip.compress(payload, compresslevel=6))


def bytes_png(img01: np.ndarray) -> tuple[int, str]:
    """
    PNG para baseline visual original. Se reporta como png.
    """
    from PIL import Image

    arr = uint8_array(img01)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return len(buf.getvalue()), "png"


def bytes_uint8_gzip(arr01: np.ndarray) -> tuple[int, str]:
    q = uint8_array(arr01)
    return gzip_len_from_bytes(q.tobytes()), "uint8+gzip"


def bytes_from_uint8_payload(q: np.ndarray, method: str = "uint8+gzip") -> tuple[int, str]:
    return gzip_len_from_bytes(np.asarray(q, dtype="uint8").tobytes()), method


def bytes_label_gzip() -> tuple[int, str]:
    return gzip_len_from_bytes(np.array([0], dtype="uint8").tobytes()), "uint8+gzip"


@dataclass
class Quantizer:
    minimum: np.ndarray
    maximum: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "Quantizer":
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        return Quantizer(minimum=mn.astype("float32"), maximum=mx.astype("float32"))

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

def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def run_torch_backend(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, epochs: int, seed: int):
    import torch
    import torch.nn as nn

    set_seed(seed)
    device = torch.device("cpu")

    class ImageCNN(nn.Module):
        def __init__(self, in_hw: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            flat = 32 * (in_hw // 4) * (in_hw // 4)
            self.embed = nn.Linear(flat, EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x, return_embed: bool = False):
            z = self.features(x).flatten(1)
            e = torch.relu(self.embed(z))
            out = self.head(e)
            if return_embed:
                return out, e
            return out

    class EmbeddingMLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, NUM_CLASSES),
            )

        def forward(self, x):
            return self.net(x)

    def to_image_tensor(X: np.ndarray):
        return torch.tensor(X[:, None, :, :], dtype=torch.float32, device=device)

    def to_vector_tensor(X: np.ndarray):
        return torch.tensor(X, dtype=torch.float32, device=device)

    def fit_image_model(model: nn.Module, X: np.ndarray, y: np.ndarray, ep: int) -> nn.Module:
        model.train()
        Xten = to_image_tensor(X)
        yten = torch.tensor(y, dtype=torch.long, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        batch = 128

        for _ in range(ep):
            perm = torch.randperm(len(Xten))
            for i in range(0, len(Xten), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = loss_fn(model(Xten[idx]), yten[idx])
                loss.backward()
                opt.step()

        return model

    def fit_vector_model(model: nn.Module, X: np.ndarray, y: np.ndarray, ep: int) -> nn.Module:
        model.train()
        Xten = to_vector_tensor(X)
        yten = torch.tensor(y, dtype=torch.long, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        batch = 128

        for _ in range(max(ep, 3)):
            perm = torch.randperm(len(Xten))
            for i in range(0, len(Xten), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = loss_fn(model(Xten[idx]), yten[idx])
                loss.backward()
                opt.step()

        return model

    @torch.no_grad()
    def predict_image(model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        out = model(to_image_tensor(X))
        return out.argmax(1).cpu().numpy()

    @torch.no_grad()
    def predict_vector(model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        out = model(to_vector_tensor(X))
        return out.argmax(1).cpu().numpy()

    @torch.no_grad()
    def extract_embedding(model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        _, emb = model(to_image_tensor(X), return_embed=True)
        return emb.cpu().numpy().astype("float32")

    def timed_image_pred(model: nn.Module, X: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = now_ms()
        pred = predict_image(model, X)
        elapsed = now_ms() - t0
        return pred, elapsed / max(1, len(X))

    def timed_vector_pred(model: nn.Module, X: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = now_ms()
        pred = predict_vector(model, X)
        elapsed = now_ms() - t0
        return pred, elapsed / max(1, len(X))

    # ------------------------------------------------------
    # Transmisor: se usa para label y embedding.
    # ------------------------------------------------------
    tx = fit_image_model(ImageCNN(28).to(device), Xtr, ytr, epochs)

    # ------------------------------------------------------
    # Receptor independiente para original.
    # ------------------------------------------------------
    rx_original = fit_image_model(ImageCNN(28).to(device), Xtr, ytr, epochs)

    X16tr = resize_batch_pil(Xtr, 16)
    X16te = resize_batch_pil(Xte, 16)
    X8tr = resize_batch_pil(Xtr, 8)
    X8te = resize_batch_pil(Xte, 8)

    rx_16 = fit_image_model(ImageCNN(16).to(device), X16tr, ytr, epochs)
    rx_8 = fit_image_model(ImageCNN(8).to(device), X8tr, ytr, epochs)

    # ------------------------------------------------------
    # Label predicha por transmisor, medida contra verdad.
    # ------------------------------------------------------
    y_label = predict_image(tx, Xte)

    # ------------------------------------------------------
    # Embedding cuantizado/reconstruido.
    # La utilidad se mide sobre el embedding reconstruido.
    # ------------------------------------------------------
    Etr_float = extract_embedding(tx, Xtr)
    Ete_float = extract_embedding(tx, Xte)
    qz = Quantizer.fit(Etr_float)
    Etr_q = qz.transform(Etr_float)
    Ete_q = qz.transform(Ete_float)
    Etr_rx = qz.inverse_transform(Etr_q)
    Ete_rx = qz.inverse_transform(Ete_q)

    rx_emb = fit_vector_model(EmbeddingMLP(EMBED_DIM).to(device), Etr_rx, ytr, epochs)

    # ------------------------------------------------------
    # Predicciones y latencias.
    # ------------------------------------------------------
    latency_sample = min(LATENCY_SAMPLE_LIMIT, len(Xte))

    pred_original, lat_original = timed_image_pred(rx_original, Xte[:latency_sample])
    pred_16, lat_16 = timed_image_pred(rx_16, X16te[:latency_sample])
    pred_8, lat_8 = timed_image_pred(rx_8, X8te[:latency_sample])
    pred_emb, lat_emb = timed_vector_pred(rx_emb, Ete_rx[:latency_sample])

    # Para accuracy usamos test completo.
    pred_original_full = predict_image(rx_original, Xte)
    pred_16_full = predict_image(rx_16, X16te)
    pred_8_full = predict_image(rx_8, X8te)
    pred_emb_full = predict_vector(rx_emb, Ete_rx)

    acc = {
        "original": float((pred_original_full == yte).mean()),
        "resize_16x16": float((pred_16_full == yte).mean()),
        "resize_8x8": float((pred_8_full == yte).mean()),
        "embedding": float((pred_emb_full == yte).mean()),
        "label": float((y_label == yte).mean()),
    }

    latencies = {
        "original": float(lat_original),
        "resize_16x16": float(lat_16),
        "resize_8x8": float(lat_8),
        "embedding": float(lat_emb),
        "label": 0.0,
        "silence": 0.0,
    }

    artifacts = {
        "X16te": X16te,
        "X8te": X8te,
        "Ete_q": Ete_q,
        "embedding_quantizer_min_mean": float(qz.minimum.mean()),
        "embedding_quantizer_max_mean": float(qz.maximum.mean()),
    }

    return acc, latencies, artifacts, "torch"


# ==========================================================
# Backend sklearn de respaldo
# ==========================================================

def run_sklearn_backend(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, epochs: int, seed: int):
    from sklearn.neural_network import MLPClassifier

    set_seed(seed)

    def fit_mlp(Xa: np.ndarray, ya: np.ndarray, hidden=(64,)):
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=40 + 10 * epochs,
            random_state=seed,
            early_stopping=False,
        )
        clf.fit(Xa.reshape(len(Xa), -1), ya)
        return clf

    def pred_time(clf: Any, Xa: np.ndarray) -> tuple[np.ndarray, float]:
        n = min(LATENCY_SAMPLE_LIMIT, len(Xa))
        t0 = now_ms()
        pred = clf.predict(Xa[:n].reshape(n, -1))
        elapsed = now_ms() - t0
        return pred, elapsed / max(1, n)

    tx = fit_mlp(Xtr, ytr)
    rx_original = fit_mlp(Xtr, ytr)

    X16tr = resize_batch_pil(Xtr, 16)
    X16te = resize_batch_pil(Xte, 16)
    X8tr = resize_batch_pil(Xtr, 8)
    X8te = resize_batch_pil(Xte, 8)

    rx_16 = fit_mlp(X16tr, ytr)
    rx_8 = fit_mlp(X8tr, ytr)

    y_label = tx.predict(Xte.reshape(len(Xte), -1))

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((28 * 28, EMBED_DIM)).astype("float32")
    Etr_float = np.maximum(Xtr.reshape(len(Xtr), -1) @ W, 0)
    Ete_float = np.maximum(Xte.reshape(len(Xte), -1) @ W, 0)

    qz = Quantizer.fit(Etr_float)
    Etr_q = qz.transform(Etr_float)
    Ete_q = qz.transform(Ete_float)
    Etr_rx = qz.inverse_transform(Etr_q)
    Ete_rx = qz.inverse_transform(Ete_q)

    rx_emb = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=40 + 10 * epochs,
        random_state=seed,
    )
    rx_emb.fit(Etr_rx, ytr)

    pred_original_full = rx_original.predict(Xte.reshape(len(Xte), -1))
    pred_16_full = rx_16.predict(X16te.reshape(len(X16te), -1))
    pred_8_full = rx_8.predict(X8te.reshape(len(X8te), -1))
    pred_emb_full = rx_emb.predict(Ete_rx)

    _, lat_original = pred_time(rx_original, Xte)
    _, lat_16 = pred_time(rx_16, X16te)
    _, lat_8 = pred_time(rx_8, X8te)

    n = min(LATENCY_SAMPLE_LIMIT, len(Ete_rx))
    t0 = now_ms()
    _ = rx_emb.predict(Ete_rx[:n])
    lat_emb = (now_ms() - t0) / max(1, n)

    acc = {
        "original": float((pred_original_full == yte).mean()),
        "resize_16x16": float((pred_16_full == yte).mean()),
        "resize_8x8": float((pred_8_full == yte).mean()),
        "embedding": float((pred_emb_full == yte).mean()),
        "label": float((y_label == yte).mean()),
    }

    latencies = {
        "original": float(lat_original),
        "resize_16x16": float(lat_16),
        "resize_8x8": float(lat_8),
        "embedding": float(lat_emb),
        "label": 0.0,
        "silence": 0.0,
    }

    artifacts = {
        "X16te": X16te,
        "X8te": X8te,
        "Ete_q": Ete_q,
        "embedding_quantizer_min_mean": float(qz.minimum.mean()),
        "embedding_quantizer_max_mean": float(qz.maximum.mean()),
    }

    return acc, latencies, artifacts, "sklearn"


# ==========================================================
# Medicion de bytes
# ==========================================================

def mean_bytes_for_representations(Xte: np.ndarray, artifacts: dict[str, Any]):
    n = min(BYTE_SAMPLE_LIMIT, len(Xte))
    idx = np.arange(n)

    X16te = artifacts["X16te"]
    X8te = artifacts["X8te"]
    Ete_q = artifacts["Ete_q"]

    original_vals = [bytes_png(Xte[i])[0] for i in idx]
    resize16_vals = [bytes_uint8_gzip(X16te[i])[0] for i in idx]
    resize8_vals = [bytes_uint8_gzip(X8te[i])[0] for i in idx]
    embedding_vals = [bytes_from_uint8_payload(Ete_q[i], "uint8+gzip")[0] for i in idx]
    label_bytes = bytes_label_gzip()[0]

    ser = {
        "original": float(np.mean(original_vals)),
        "resize_16x16": float(np.mean(resize16_vals)),
        "resize_8x8": float(np.mean(resize8_vals)),
        "embedding": float(np.mean(embedding_vals)),
        "label": float(label_bytes),
        "silence": 0.0,
    }

    raw = {
        "original": float(28 * 28),
        "resize_16x16": float(16 * 16),
        "resize_8x8": float(8 * 8),
        "embedding": float(EMBED_DIM),
        "label": 1.0,
        "silence": 0.0,
    }

    method = {
        "original": "png",
        "resize_16x16": "uint8+gzip",
        "resize_8x8": "uint8+gzip",
        "embedding": "uint8+gzip",
        "label": "uint8+gzip",
        "silence": "none",
    }

    return ser, raw, method


# ==========================================================
# Corrida por semilla y agregacion
# ==========================================================

def run_once(args: argparse.Namespace, seed: int):
    set_seed(seed)

    Xtr, ytr, Xte, yte, source = load_fashion_mnist(
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        seed=seed,
    )

    backend_fn = run_torch_backend if has_torch() else run_sklearn_backend
    acc, latencies, artifacts, backend = backend_fn(Xtr, ytr, Xte, yte, args.epochs, seed)

    vals, counts = np.unique(yte, return_counts=True)
    acc_silence = float(counts.max() / counts.sum())
    acc["silence"] = acc_silence

    ser, raw, method = mean_bytes_for_representations(Xte, artifacts)

    return {
        "acc": acc,
        "latencies": latencies,
        "ser": ser,
        "raw": raw,
        "method": method,
        "source": source,
        "backend": backend,
        "artifacts": artifacts,
    }


def aggregate_runs(runs: list[dict[str, Any]]):
    reps = ["original", "resize_16x16", "resize_8x8", "embedding", "label", "silence"]

    acc_mean = {}
    acc_std = {}
    latency_mean = {}
    latency_std = {}

    for r in reps:
        acc_values = np.array([run["acc"][r] for run in runs], dtype="float64")
        lat_values = np.array([run["latencies"][r] for run in runs], dtype="float64")
        acc_mean[r] = float(acc_values.mean())
        acc_std[r] = float(acc_values.std(ddof=0))
        latency_mean[r] = float(lat_values.mean())
        latency_std[r] = float(lat_values.std(ddof=0))

    ser = runs[0]["ser"]
    raw = runs[0]["raw"]
    method = runs[0]["method"]
    source = runs[0]["source"]
    backend = runs[0]["backend"]

    return reps, acc_mean, acc_std, latency_mean, latency_std, ser, raw, method, source, backend


# ==========================================================
# Tabla final y SSP
# ==========================================================

def representation_group(rep: str) -> str:
    if rep == "label":
        return "decision"
    if rep == "embedding":
        return "semantic"
    if rep in {"original", "resize_16x16", "resize_8x8"}:
        return "visual"
    if rep == "silence":
        return "silence"
    return "other"


def receiver_name(rep: str) -> str:
    return {
        "original": "independent_full_image",
        "resize_16x16": "independent_resize",
        "resize_8x8": "independent_resize",
        "embedding": "independent_embedding_quantized",
        "label": "ground_truth",
        "silence": "majority_class",
    }[rep]


def build_results_table(
    reps: list[str],
    acc_mean: dict[str, float],
    acc_std: dict[str, float],
    latency_mean: dict[str, float],
    latency_std: dict[str, float],
    ser: dict[str, float],
    raw: dict[str, float],
    method: dict[str, str],
    seed_count: int,
):
    u_max = max(acc_mean.values())
    b_original = ser["original"]

    rows = []
    for rep in reps:
        b = ser[rep]
        u = acc_mean[rep]
        is_silence = rep == "silence"

        utility_per_byte = np.nan if is_silence or b == 0 else u / b
        byte_savings = 0.0 if rep == "original" else max(0.0, 1.0 - b / b_original)
        within_eps = u >= (u_max - EPS_SSP)
        enough_savings = byte_savings >= SAVINGS_MIN
        candidate = (not is_silence) and within_eps and enough_savings

        rows.append(
            {
                "representation": rep,
                "representation_group": representation_group(rep),
                "bytes_raw": round(raw[rep], 4),
                "bytes": round(b, 4),
                "bits": round(8.0 * b, 4),
                "serialization_method": method[rep],
                "accuracy": round(u, 6),
                "accuracy_std": round(acc_std[rep], 6),
                "latency_ms": round(latency_mean[rep], 6),
                "latency_ms_std": round(latency_std[rep], 6),
                "utility_per_byte": np.nan if np.isnan(utility_per_byte) else round(utility_per_byte, 8),
                "byte_savings": round(byte_savings, 6),
                "within_epsilon": "yes" if within_eps else "no",
                "enough_savings": "yes" if enough_savings else "no",
                "is_ssp_any": "no",
                "is_ssp_decision": "no",
                "is_ssp_semantic": "no",
                "is_ssp_visual": "no",
                "receiver": receiver_name(rep),
                "is_silence_baseline": "yes" if is_silence else "no",
                "seed_count": seed_count,
            }
        )

    df = pd.DataFrame(rows)

    def mark_winner(group_filter: pd.Series, col_name: str) -> str | None:
        candidates = df[
            group_filter
            & (df["is_silence_baseline"] == "no")
            & (df["within_epsilon"] == "yes")
            & (df["enough_savings"] == "yes")
        ].copy()

        if candidates.empty:
            return None

        winner_rep = candidates.sort_values("bytes", ascending=True).iloc[0]["representation"]
        df.loc[df["representation"] == winner_rep, col_name] = "yes"
        return str(winner_rep)

    mark_winner(pd.Series([True] * len(df)), "is_ssp_any")
    mark_winner(df["representation_group"] == "decision", "is_ssp_decision")
    mark_winner(df["representation_group"] == "semantic", "is_ssp_semantic")
    mark_winner(df["representation_group"] == "visual", "is_ssp_visual")

    return df, u_max


# ==========================================================
# Graficas
# ==========================================================

def monotone_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estima la envolvente superior empirica: best-so-far al ordenar por bytes.
    Los puntos crudos pueden no ser monotonos; la frontera estimada si.
    """
    d = df[df["is_silence_baseline"] == "no"].copy()
    d = d.sort_values("bytes", ascending=True)
    d["frontier_accuracy"] = d["accuracy"].cummax()
    return d


def make_plots(df: pd.DataFrame, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df["is_silence_baseline"] == "no"].sort_values("bytes")

    plt.figure(figsize=(8, 5))
    plt.plot(d["bytes"], d["accuracy"], "o-", label="puntos observados")
    front = monotone_frontier(df)
    plt.plot(front["bytes"], front["frontier_accuracy"], "--", label="frontera empirica best-so-far")

    for _, row in d.iterrows():
        label = str(row["representation"])
        if row["is_ssp_any"] == "yes":
            label += " (SSP)"
        plt.annotate(
            label,
            (row["bytes"], row["accuracy"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Bytes transmitidos serializados")
    plt.ylabel("Exactitud")
    plt.title("TaskNet-ELU Fase 1: utilidad vs bytes")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "accuracy_vs_bytes.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    d2 = d.dropna(subset=["utility_per_byte"])
    plt.bar(d2["representation"], d2["utility_per_byte"])
    plt.ylabel("Utilidad por byte")
    plt.title("TaskNet-ELU Fase 1: utilidad por byte")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "utility_per_byte.png", dpi=140)
    plt.close()


# ==========================================================
# Informes
# ==========================================================

def table_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def selected_ssp(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[df[col] == "yes"]


def report_ssp_line(df: pd.DataFrame, col: str, label: str) -> str:
    sel = selected_ssp(df, col)
    if sel.empty:
        return f"- **{label}:** no encontrado bajo epsilon={EPS_SSP} y ahorro minimo={SAVINGS_MIN:.0%}."
    r = sel.iloc[0]
    return (
        f"- **{label}: `{r['representation']}`** "
        f"(grupo={r['representation_group']}, accuracy={r['accuracy']:.4f}, "
        f"bytes={r['bytes']:.2f}, ahorro={r['byte_savings']:.1%})."
    )


def write_reports(
    df: pd.DataFrame,
    u_max: float,
    args: argparse.Namespace,
    source: str,
    backend: str,
    reports_dir: Path,
) -> None:
    lines = []
    lines.append("# Informe Fase 1 — TaskNet-ELU")
    lines.append("")
    lines.append("## Configuracion")
    lines.append(f"- Fuente de datos: **{source}**")
    lines.append(f"- Backend de modelos: **{backend}**")
    lines.append(f"- Epocas: **{args.epochs}**")
    lines.append(f"- Semillas: **{args.seeds}**")
    lines.append(f"- limit_train: **{args.limit_train}**")
    lines.append(f"- limit_test: **{args.limit_test}**")
    lines.append(f"- Tolerancia SSP epsilon: **{EPS_SSP}**")
    lines.append(f"- Ahorro minimo SSP: **{SAVINGS_MIN:.0%}**")
    lines.append("")

    if source == "synthetic":
        lines.append("> **AVISO:** se uso dataset sintetico. Los numeros validan mecanica, no resultado cientifico real.")
        lines.append("")

    if backend == "sklearn":
        lines.append("> **AVISO:** sin torch, el embedding es proxy por proyeccion aleatoria. Usar torch para resultado principal.")
        lines.append("")

    lines.append("## Resultados")
    lines.append("")
    lines.append(table_to_markdown(df))
    lines.append("")

    lines.append("## Lectura SSP")
    lines.append("")
    lines.append(f"- Utilidad maxima observada: **U_max = {u_max:.4f}**")
    lines.append(report_ssp_line(df, "is_ssp_any", "SSP general de menor byte"))
    lines.append(report_ssp_line(df, "is_ssp_decision", "SSP de decision"))
    lines.append(report_ssp_line(df, "is_ssp_semantic", "SSP semantico reutilizable"))
    lines.append(report_ssp_line(df, "is_ssp_visual", "SSP visual"))
    lines.append("")

    lines.append("## Interpretacion tecnica")
    lines.append("")
    lines.append(
        "La etiqueta (`label`) puede aparecer como SSP general porque representa una decision ya tomada "
        "por el transmisor. Esto es valido si la red solo necesita transportar la decision final, pero no "
        "sirve como evidencia reutilizable ni como representacion rica del dato."
    )
    lines.append("")
    lines.append(
        "El `embedding` es la representacion semantica mas relevante para TaskNet-ELU porque conserva informacion "
        "reutilizable para un receptor independiente. En esta version se evalua despues de cuantizarlo y reconstruirlo, "
        "por lo que la utilidad se mide sobre lo que realmente viajaria."
    )
    lines.append("")
    lines.append(
        "Los puntos crudos de utilidad vs bytes no tienen que ser monotonos. La frontera empirica se interpreta como "
        "la envolvente superior best-so-far al ordenar por bytes."
    )
    lines.append("")
    lines.append(
        "La latencia reportada es un proxy de computo por representacion, no una medicion de red."
    )

    (reports_dir / "informe_fase_1.md").write_text("\n".join(lines), encoding="utf-8")

    cierre = []
    cierre.append("# Cierre de Fase 1 — TaskNet-ELU")
    cierre.append("")
    cierre.append("## Estado")
    if source == "fashion_mnist" and backend == "torch":
        cierre.append("Fase 1 ejecutada con Fashion-MNIST real y backend torch.")
    else:
        cierre.append("Fase 1 ejecutada en modo no principal. Repetir con Fashion-MNIST real y torch.")
    cierre.append("")
    cierre.append("## Comando ejecutado")
    cierre.append(f"`python scripts/01_utility_per_bit_fashion_mnist.py --epochs {args.epochs} --seeds {args.seeds}`")
    cierre.append("")
    cierre.append("## Archivos generados")
    cierre.append("- `results/tables/utility_per_bit_fashion_mnist.csv`")
    cierre.append("- `results/tables/utility_per_byte_fashion_mnist.csv`")
    cierre.append("- `results/figures/accuracy_vs_bytes.png`")
    cierre.append("- `results/figures/utility_per_byte.png`")
    cierre.append("- `reports/informe_fase_1.md`")
    cierre.append("")
    cierre.append("## Resultado SSP")
    cierre.append(report_ssp_line(df, "is_ssp_any", "SSP general"))
    cierre.append(report_ssp_line(df, "is_ssp_semantic", "SSP semantico"))
    cierre.append("")
    cierre.append("## Decision siguiente")
    cierre.append(
        "Si los archivos existen, las columnas pasan validacion y los resultados se interpretan sin forzar, "
        "esta corrida puede considerarse Fase 1 v0.2. La siguiente decision es validar visualmente graficas y "
        "pasar a Fase 2 o repetir con mas epocas/semillas."
    )

    (reports_dir / "cierre_fase_1.md").write_text("\n".join(cierre), encoding="utf-8")


# ==========================================================
# Validaciones automaticas
# ==========================================================

def validate_output(df: pd.DataFrame) -> None:
    required = {
        "representation",
        "representation_group",
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

    expected_reps = {"original", "resize_16x16", "resize_8x8", "embedding", "label", "silence"}
    found_reps = set(df["representation"].astype(str))
    missing_reps = expected_reps - found_reps
    if missing_reps:
        raise RuntimeError(f"Faltan representaciones: {sorted(missing_reps)}")

    if not ((df["accuracy"] >= 0) & (df["accuracy"] <= 1)).all():
        raise RuntimeError("Accuracy fuera de rango [0,1].")

    if not (df["bytes"] >= 0).all():
        raise RuntimeError("Hay bytes negativos.")

    if not (df["latency_ms"] >= 0).all():
        raise RuntimeError("Hay latencias negativas.")

    upb_non_na = df["utility_per_byte"].dropna()
    if not (upb_non_na >= 0).all():
        raise RuntimeError("Hay utilidad por byte negativa.")

    na_rows = df[df["utility_per_byte"].isna()]
    if not (na_rows["is_silence_baseline"] == "yes").all():
        raise RuntimeError("utility_per_byte=NA solo se permite en silence.")

    for rep in ["original", "resize_16x16", "resize_8x8", "embedding"]:
        row = df[df["representation"] == rep].iloc[0]
        if not str(row["receiver"]).startswith("independent"):
            raise RuntimeError(f"{rep} no fue evaluado con receptor independiente.")

    label_row = df[df["representation"] == "label"].iloc[0]
    if label_row["receiver"] != "ground_truth":
        raise RuntimeError("label debe evaluarse contra ground_truth.")

    silence_row = df[df["representation"] == "silence"].iloc[0]
    if silence_row["is_silence_baseline"] != "yes":
        raise RuntimeError("silence debe estar marcado como baseline.")


# ==========================================================
# CLI principal
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TaskNet-ELU Fase 1: utilidad vs bytes en Fashion-MNIST."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=0, dest="limit_train")
    parser.add_argument("--limit-test", type=int, default=0, dest="limit_test")
    parser.add_argument("--seeds", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tables_dir, figs_dir, reports_dir = ensure_dirs()

    runs = []
    for seed in range(args.seeds):
        print(f"==> Corrida semilla {seed}")
        run = run_once(args, seed)
        runs.append(run)

    (
        reps,
        acc_mean,
        acc_std,
        latency_mean,
        latency_std,
        ser,
        raw,
        method,
        source,
        backend,
    ) = aggregate_runs(runs)

    df, u_max = build_results_table(
        reps=reps,
        acc_mean=acc_mean,
        acc_std=acc_std,
        latency_mean=latency_mean,
        latency_std=latency_std,
        ser=ser,
        raw=raw,
        method=method,
        seed_count=args.seeds,
    )

    validate_output(df)

    csv_bit = tables_dir / "utility_per_bit_fashion_mnist.csv"
    csv_byte = tables_dir / "utility_per_byte_fashion_mnist.csv"
    df.to_csv(csv_bit, index=False)
    df.to_csv(csv_byte, index=False)

    make_plots(df, figs_dir)
    write_reports(df, u_max, args, source, backend, reports_dir)

    print("\n=== Resultados Fase 1 v0.2 ===")
    print(df.to_string(index=False))
    print("")
    print(f"U_max: {u_max:.6f}")
    print(f"CSV bit:  {csv_bit}")
    print(f"CSV byte: {csv_byte}")
    print(f"Figura 1: {figs_dir / 'accuracy_vs_bytes.png'}")
    print(f"Figura 2: {figs_dir / 'utility_per_byte.png'}")
    print(f"Informe:  {reports_dir / 'informe_fase_1.md'}")
    print(f"Cierre:   {reports_dir / 'cierre_fase_1.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
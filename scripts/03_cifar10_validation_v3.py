#!/usr/bin/env python3
"""
03_cifar10_validation.py — Fase 3 de TaskNet-ELU.

Version fusionada v3:
- Base practica/compacta para CIFAR-10.
- Arquitectura CNN liviana para CPU, pero con soporte opcional de CUDA.
- Mantiene el protocolo defendible de Fases 1-2:
    * receptor independiente por representacion accionable;
    * embedding cuantizado a uint8 y reconstruido antes de evaluar;
    * bytes medidos sobre lo que viaja;
    * SSP separado: general, decision, semantico, visual;
    * dominancia honesta: primero conservar utilidad dentro de epsilon;
    * latencia = inferencia del receptor por muestra, no latencia de red.
- Agrega elementos de investigacion:
    * columna dataset/source;
    * validacion fuerte del CSV;
    * comparacion automatica contra Fase 2 si existe el CSV de Fashion-MNIST;
    * informe y cierre de Fase 3.

Pregunta de Fase 3:
    ¿El comportamiento observado en Fashion-MNIST persiste en CIFAR-10,
    un dataset mas complejo, a color, con texturas y fondos?

Salidas:
    results/tables/cifar10_learned_vs_classic.csv
    results/figures/phase3_accuracy_vs_bytes.png
    results/figures/phase3_utility_per_byte.png
    results/figures/phase3_learned_vs_classic_frontier.png
    reports/informe_fase_3.md
    reports/cierre_fase_3.md

Uso recomendado:
    python scripts/03_cifar10_validation.py --epochs 1 --limit-train 5000 --limit-test 1000 --seeds 1
    python scripts/03_cifar10_validation.py --epochs 3 --limit-train 15000 --limit-test 3000 --seeds 2
    python scripts/03_cifar10_validation.py --epochs 5 --limit-train 30000 --limit-test 5000 --seeds 3

Corrida fuerte, si la maquina aguanta:
    python scripts/03_cifar10_validation.py --epochs 10 --seeds 3
"""

from __future__ import annotations

import argparse
import gc
import gzip
import io
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ==========================================================
# Constantes del experimento
# ==========================================================
EPS_SSP = 0.03
SAVINGS_MIN = 0.80
NUM_CLASSES = 10
IMG_HW = 32
IMG_CH = 3
EMBED_DIM = 64
BYTE_SAMPLE_LIMIT = 500
LATENCY_SAMPLE_LIMIT = 1000
JPEG_QUALITIES = [90, 50, 20, 10, 5]
BATCH_SIZE = 128


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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
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


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    # auto
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def clear_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ==========================================================
# Carga de datos: CIFAR-10
# ==========================================================
def load_cifar10(limit_train: int, limit_test: int, seed: int):
    """
    Devuelve Xtr, ytr, Xte, yte, source.

    X tiene shape (N, 32, 32, 3), dtype float32, rango [0,1].
    """
    try:
        import torchvision
        import torchvision.transforms as T

        tf = T.Compose([T.ToTensor()])
        tr = torchvision.datasets.CIFAR10(
            root=str(ROOT / "data" / "raw"),
            train=True,
            download=True,
            transform=tf,
        )
        te = torchvision.datasets.CIFAR10(
            root=str(ROOT / "data" / "raw"),
            train=False,
            download=True,
            transform=tf,
        )

        # CIFAR-10 ya viene como uint8 (N,32,32,3)
        Xtr = tr.data.astype("float32") / 255.0
        ytr = np.asarray(tr.targets, dtype="int64")
        Xte = te.data.astype("float32") / 255.0
        yte = np.asarray(te.targets, dtype="int64")
        source = "cifar10"
    except Exception as exc:
        print(
            f"[AVISO] No se pudo cargar CIFAR-10 ({type(exc).__name__}: {exc}).\n"
            "        Se usa dataset SINTETICO a color solo para validar la mecanica.\n"
            "        No uses esos numeros como resultado cientifico real."
        )
        rng = np.random.default_rng(seed)
        Xtr, ytr = synthetic_color_dataset(6000, rng)
        Xte, yte = synthetic_color_dataset(1500, rng)
        source = "synthetic_color"

    if limit_train and limit_train > 0:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test and limit_test > 0:
        Xte, yte = Xte[:limit_test], yte[:limit_test]

    return Xtr, ytr, Xte, yte, source


def synthetic_color_dataset(n: int, rng: np.random.Generator):
    """Dataset sintetico 32x32x3 con 10 clases. Solo valida pipeline."""
    y = rng.integers(0, NUM_CLASSES, size=n)
    X = np.zeros((n, IMG_HW, IMG_HW, IMG_CH), dtype="float32")
    centers = [(8 + 7 * (c % 3), 8 + 7 * (c // 3)) for c in range(NUM_CLASSES)]
    yy, xx = np.mgrid[0:IMG_HW, 0:IMG_HW]

    for i in range(n):
        cy, cx = centers[int(y[i])]
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 30.0)
        main_channel = int(y[i]) % IMG_CH
        for c in range(IMG_CH):
            base = blob if c == main_channel else 0.30 * blob
            noise = 0.05 * rng.standard_normal((IMG_HW, IMG_HW))
            X[i, :, :, c] = (base + noise).clip(0, 1)

    return X.astype("float32"), y.astype("int64")


# ==========================================================
# Serializacion y transformaciones visuales
# ==========================================================
def uint8_img(X: np.ndarray) -> np.ndarray:
    return (np.clip(X, 0, 1) * 255.0).round().astype("uint8")


def resize_batch_pil(X: np.ndarray, size: int) -> np.ndarray:
    """Resize real a color: (N,H,W,3) -> (N,size,size,3)."""
    from PIL import Image

    out = []
    resample = getattr(Image, "Resampling", Image).BILINEAR
    for img in X:
        im = Image.fromarray(uint8_img(img), mode="RGB")
        im = im.resize((size, size), resample)
        out.append(np.asarray(im).astype("float32") / 255.0)
    return np.stack(out).astype("float32")


def jpeg_roundtrip(img01: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """Comprime una imagen RGB a JPEG y la reconstruye."""
    from PIL import Image

    im = Image.fromarray(uint8_img(img01), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    payload = buf.getvalue()
    recon = np.asarray(Image.open(io.BytesIO(payload))).astype("float32") / 255.0
    return recon, len(payload)


def png_roundtrip(img01: np.ndarray) -> tuple[np.ndarray, int]:
    """Comprime una imagen RGB a PNG y la reconstruye."""
    from PIL import Image

    im = Image.fromarray(uint8_img(img01), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    payload = buf.getvalue()
    recon = np.asarray(Image.open(io.BytesIO(payload))).astype("float32") / 255.0
    return recon, len(payload)


def jpeg_batch(X: np.ndarray, quality: int) -> tuple[np.ndarray, list[int]]:
    recons: list[np.ndarray] = []
    sizes: list[int] = []
    for img in X:
        r, b = jpeg_roundtrip(img, quality)
        recons.append(r)
        sizes.append(b)
    return np.stack(recons).astype("float32"), sizes


def png_batch(X: np.ndarray) -> tuple[np.ndarray, list[int]]:
    recons: list[np.ndarray] = []
    sizes: list[int] = []
    for img in X:
        r, b = png_roundtrip(img)
        recons.append(r)
        sizes.append(b)
    return np.stack(recons).astype("float32"), sizes


def gzip_len(payload: bytes) -> int:
    return len(gzip.compress(payload, compresslevel=6))


def bytes_uint8_gzip(arr01: np.ndarray) -> int:
    return gzip_len(uint8_img(arr01).tobytes())


def bytes_label_gzip() -> int:
    return gzip_len(np.array([0], dtype="uint8").tobytes())


def mean_sample_bytes(sizes: list[int]) -> float:
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
def make_image_clf_torch(
    in_hw: int,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    epochs: int,
    seed: int,
    device_name: str,
):
    """CNN compacta para imagen RGB. Sirve para 32x32, 16x16 y 8x8."""
    import torch
    import torch.nn as nn

    set_seed(seed)
    device = torch.device(device_name)

    class CNN(nn.Module):
        def __init__(self, hw: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(IMG_CH, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            pooled = max(1, hw // 8)
            flat = 64 * pooled * pooled
            self.embed = nn.Linear(flat, EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x, return_embed: bool = False):
            e = torch.relu(self.embed(self.features(x).flatten(1)))
            out = self.head(e)
            if return_embed:
                return out, e
            return out

    model = CNN(in_hw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(np.transpose(Xtr, (0, 3, 1, 2)), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    return model


def predict_image_torch(model, X: np.ndarray, device_name: str) -> np.ndarray:
    import torch

    device = torch.device(device_name)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(
                np.transpose(X[i : i + BATCH_SIZE], (0, 3, 1, 2)),
                dtype=torch.float32,
                device=device,
            )
            out = model(xb)
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def extract_embedding_torch(model, X: np.ndarray, device_name: str) -> np.ndarray:
    import torch

    device = torch.device(device_name)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(
                np.transpose(X[i : i + BATCH_SIZE], (0, 3, 1, 2)),
                dtype=torch.float32,
                device=device,
            )
            _, e = model(xb, return_embed=True)
            chunks.append(e.cpu().numpy().astype("float32"))
    return np.concatenate(chunks, axis=0)


def make_vector_clf_torch(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    epochs: int,
    seed: int,
    device_name: str,
):
    import torch
    import torch.nn as nn

    set_seed(seed)
    device = torch.device(device_name)

    class MLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Linear(128, NUM_CLASSES),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)

    model.train()
    for _ in range(max(epochs, 3)):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    return model


def predict_vector_torch(model, X: np.ndarray, device_name: str) -> np.ndarray:
    import torch

    device = torch.device(device_name)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i : i + BATCH_SIZE], dtype=torch.float32, device=device)
            out = model(xb)
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ==========================================================
# Backend sklearn de respaldo
# ==========================================================
def make_image_clf_sklearn(Xtr: np.ndarray, ytr: np.ndarray, epochs: int, seed: int):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(128,),
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
        hidden_layer_sizes=(128,),
        max_iter=40 + 10 * epochs,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    return clf


def predict_vector_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


# ==========================================================
# Evaluacion de representaciones
# ==========================================================
def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def timed_predict(
    predict_fn: Callable[..., np.ndarray],
    model: Any,
    Xeval: np.ndarray,
    device_name: str | None = None,
) -> tuple[np.ndarray, float]:
    """Mide SOLO inferencia del receptor por muestra."""
    n = min(LATENCY_SAMPLE_LIMIT, len(Xeval))
    t0 = now_ms()
    if device_name is None:
        _ = predict_fn(model, Xeval[:n])
    else:
        _ = predict_fn(model, Xeval[:n], device_name)
    latency_ms = (now_ms() - t0) / max(1, n)

    if device_name is None:
        pred_full = predict_fn(model, Xeval)
    else:
        pred_full = predict_fn(model, Xeval, device_name)
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
    device_name: str,
):
    rows: list[dict[str, Any]] = []
    torch_ok = has_torch()

    if torch_ok:
        backend = f"torch:{device_name}"
        tx = make_image_clf_torch(IMG_HW, Xtr, ytr, epochs, seed, device_name)
        y_label = predict_image_torch(tx, Xte, device_name)
        Etr = extract_embedding_torch(tx, Xtr, device_name)
        Ete = extract_embedding_torch(tx, Xte, device_name)
    else:
        backend = "sklearn"
        tx = make_image_clf_sklearn(Xtr, ytr, epochs, seed)
        y_label = predict_sklearn(tx, Xte)
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((IMG_HW * IMG_HW * IMG_CH, EMBED_DIM)).astype("float32")
        Etr = np.maximum(Xtr.reshape(len(Xtr), -1) @ W, 0)
        Ete = np.maximum(Xte.reshape(len(Xte), -1) @ W, 0)

    # ------------------------------------------------------
    # Original PNG / baseline visual
    # ------------------------------------------------------
    if torch_ok:
        rx_full = make_image_clf_torch(IMG_HW, Xtr, ytr, epochs, seed + 100, device_name)
        pred_full, lat_full = timed_predict(predict_image_torch, rx_full, Xte, device_name)
    else:
        rx_full = make_image_clf_sklearn(Xtr, ytr, epochs, seed + 100)
        pred_full, lat_full = timed_predict(predict_sklearn, rx_full, Xte)
    _, png_sizes = png_batch(Xte)
    add_row(
        rows,
        "original_png",
        "visual",
        "png",
        "lossless",
        accuracy(pred_full, yte),
        mean_sample_bytes(png_sizes),
        float(IMG_HW * IMG_HW * IMG_CH),
        "png",
        "independent_full_image",
        lat_full,
    )
    clear_memory()

    # ------------------------------------------------------
    # JPEG a varias calidades
    # ------------------------------------------------------
    for q in JPEG_QUALITIES:
        Jtr, _ = jpeg_batch(Xtr, q)
        Jte, jsizes = jpeg_batch(Xte, q)
        if torch_ok:
            rx_jpeg = make_image_clf_torch(IMG_HW, Jtr, ytr, epochs, seed + q, device_name)
            pred_jpeg, lat_jpeg = timed_predict(predict_image_torch, rx_jpeg, Jte, device_name)
        else:
            rx_jpeg = make_image_clf_sklearn(Jtr, ytr, epochs, seed + q)
            pred_jpeg, lat_jpeg = timed_predict(predict_sklearn, rx_jpeg, Jte)
        add_row(
            rows,
            f"jpeg_q{q}",
            "visual",
            "jpeg",
            str(q),
            accuracy(pred_jpeg, yte),
            mean_sample_bytes(jsizes),
            float(IMG_HW * IMG_HW * IMG_CH),
            "jpeg",
            "independent_jpeg",
            lat_jpeg,
        )
        del Jtr, Jte, jsizes
        clear_memory()

    # ------------------------------------------------------
    # Resize 16x16 + JPEG q50
    # ------------------------------------------------------
    X16tr = resize_batch_pil(Xtr, 16)
    X16te = resize_batch_pil(Xte, 16)
    RJtr, _ = jpeg_batch(X16tr, 50)
    RJte, rjsizes = jpeg_batch(X16te, 50)
    if torch_ok:
        rx_rj = make_image_clf_torch(16, RJtr, ytr, epochs, seed + 700, device_name)
        pred_rj, lat_rj = timed_predict(predict_image_torch, rx_rj, RJte, device_name)
    else:
        rx_rj = make_image_clf_sklearn(RJtr, ytr, epochs, seed + 700)
        pred_rj, lat_rj = timed_predict(predict_sklearn, rx_rj, RJte)
    add_row(
        rows,
        "resize16_jpeg_q50",
        "visual",
        "resize",
        "16+q50",
        accuracy(pred_rj, yte),
        mean_sample_bytes(rjsizes),
        float(16 * 16 * IMG_CH),
        "jpeg",
        "independent_resize_jpeg",
        lat_rj,
    )
    del RJtr, RJte, rjsizes
    clear_memory()

    # ------------------------------------------------------
    # Resize 8x8 puro + uint8+gzip
    # ------------------------------------------------------
    X8tr = resize_batch_pil(Xtr, 8)
    X8te = resize_batch_pil(Xte, 8)
    if torch_ok:
        rx_8 = make_image_clf_torch(8, X8tr, ytr, epochs, seed + 800, device_name)
        pred_8, lat_8 = timed_predict(predict_image_torch, rx_8, X8te, device_name)
    else:
        rx_8 = make_image_clf_sklearn(X8tr, ytr, epochs, seed + 800)
        pred_8, lat_8 = timed_predict(predict_sklearn, rx_8, X8te)
    s8 = [bytes_uint8_gzip(X8te[i]) for i in range(min(BYTE_SAMPLE_LIMIT, len(X8te)))]
    add_row(
        rows,
        "resize_8x8",
        "visual",
        "resize",
        "8",
        accuracy(pred_8, yte),
        float(np.mean(s8)),
        float(8 * 8 * IMG_CH),
        "uint8+gzip",
        "independent_resize",
        lat_8,
    )
    clear_memory()

    # ------------------------------------------------------
    # Embedding cuantizado + receptor independiente
    # ------------------------------------------------------
    qz = Quantizer.fit(Etr)
    Etr_q = qz.transform(Etr)
    Ete_q = qz.transform(Ete)
    Etr_rx = qz.inverse_transform(Etr_q)
    Ete_rx = qz.inverse_transform(Ete_q)

    if torch_ok:
        rx_emb = make_vector_clf_torch(Etr_rx, ytr, epochs, seed + 900, device_name)
        pred_emb, lat_emb = timed_predict(predict_vector_torch, rx_emb, Ete_rx, device_name)
    else:
        rx_emb = make_vector_clf_sklearn(Etr_rx, ytr, epochs, seed + 900)
        pred_emb, lat_emb = timed_predict(predict_vector_sklearn, rx_emb, Ete_rx)
    s_emb = [gzip_len(Ete_q[i].tobytes()) for i in range(min(BYTE_SAMPLE_LIMIT, len(Ete_q)))]
    add_row(
        rows,
        "embedding",
        "semantic",
        "learned",
        "uint8",
        accuracy(pred_emb, yte),
        float(np.mean(s_emb)),
        float(EMBED_DIM),
        "uint8+gzip",
        "independent_embedding_quantized",
        lat_emb,
    )

    # ------------------------------------------------------
    # Label y silence
    # ------------------------------------------------------
    add_row(
        rows,
        "label",
        "decision",
        "decision",
        "argmax",
        accuracy(y_label, yte),
        float(bytes_label_gzip()),
        1.0,
        "uint8+gzip",
        "ground_truth",
        0.0,
    )

    _, counts = np.unique(yte, return_counts=True)
    add_row(
        rows,
        "silence",
        "silence",
        "silence",
        "none",
        float(counts.max() / counts.sum()),
        0.0,
        0.0,
        "none",
        "majority_class",
        0.0,
    )

    return rows, backend


# ==========================================================
# Corridas, agregacion y tabla final
# ==========================================================
def run_once(args: argparse.Namespace, seed: int):
    set_seed(seed)
    Xtr, ytr, Xte, yte, source = load_cifar10(args.limit_train, args.limit_test, seed)
    device_name = resolve_device(args.device)
    rows, backend = evaluate_all(Xtr, ytr, Xte, yte, args.epochs, seed, device_name)
    return rows, source, backend


def aggregate(all_runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_rep: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        for row in run:
            by_rep.setdefault(row["representation"], []).append(row)

    out: list[dict[str, Any]] = []
    for rep, rows in by_rep.items():
        base = dict(rows[0])
        accs = np.array([r["accuracy"] for r in rows], dtype="float64")
        lats = np.array([r["latency_ms"] for r in rows], dtype="float64")
        base["accuracy"] = float(accs.mean())
        base["accuracy_std"] = float(accs.std(ddof=0))
        base["latency_ms"] = float(lats.mean())
        base["latency_ms_std"] = float(lats.std(ddof=0))
        out.append(base)
    return out


def build_table(agg: list[dict[str, Any]], seed_count: int, source: str) -> tuple[pd.DataFrame, float]:
    original_rows = [r for r in agg if r["representation"] == "original_png"]
    if not original_rows:
        raise RuntimeError("No existe original_png para calcular ahorro de bytes.")

    b_original = float(original_rows[0]["bytes"])
    u_max = max(float(r["accuracy"]) for r in agg)
    rows: list[dict[str, Any]] = []

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
                "dataset": source,
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

    mark(pd.Series([True] * len(df), index=df.index), "is_ssp_any")
    mark(df["representation_group"] == "decision", "is_ssp_decision")
    mark(df["representation_group"] == "semantic", "is_ssp_semantic")
    mark(df["representation_group"] == "visual", "is_ssp_visual")

    return df, u_max


# ==========================================================
# Dominancia y comparacion con Fase 2
# ==========================================================
def dominance_verdict(df: pd.DataFrame) -> dict[str, Any]:
    emb = df[df["representation"] == "embedding"]
    classic = df[df["compression_family"].isin(["jpeg", "png", "resize"])]

    out: dict[str, Any] = {
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
        out["note"] = "ningun clasico quedo dentro de epsilon; mejor clasico global solo como referencia"

    emb_u = float(e["accuracy"])
    emb_b = float(e["bytes"])
    classic_dominates_embedding = classic[
        (classic["accuracy"] >= emb_u)
        & (classic["bytes"] <= emb_b)
        & ((classic["accuracy"] > emb_u) | (classic["bytes"] < emb_b))
    ]
    embedding_dominates_classic = classic[
        (classic["accuracy"] <= emb_u)
        & (classic["bytes"] >= emb_b)
        & ((classic["accuracy"] < emb_u) | (classic["bytes"] > emb_b))
    ]

    if not classic_dominates_embedding.empty:
        row = classic_dominates_embedding.sort_values("bytes").iloc[0]
        out["pareto"] = f"un clasico domina al embedding en Pareto: {row['representation']}"
    elif len(embedding_dominates_classic) == len(classic):
        out["pareto"] = "el embedding domina en Pareto a todos los clasicos"
    else:
        out["pareto"] = "no hay dominancia Pareto clara; la frontera es mixta"

    if pool.empty:
        return out

    best_classic = pool.sort_values("utility_per_byte", ascending=False).iloc[0]
    out["best_classic"] = {
        "representation": str(best_classic["representation"]),
        "utility_per_byte": float(best_classic["utility_per_byte"]),
        "accuracy": float(best_classic["accuracy"]),
        "bytes": float(best_classic["bytes"]),
        "within_epsilon": str(best_classic["within_epsilon"]) == "yes",
    }

    if not emb_within:
        out["verdict"] = "embedding_no_conserva_utilidad"
        return out
    if classic_valid.empty:
        out["verdict"] = "embedding_domina_utilidad_conservada"
        return out
    if out["emb_upb"] is None:
        out["verdict"] = "indeterminado"
    elif out["emb_upb"] > float(best_classic["utility_per_byte"]) * 1.05:
        out["verdict"] = "embedding_domina"
    elif float(best_classic["utility_per_byte"]) > out["emb_upb"] * 1.05:
        out["verdict"] = "clasico_domina"
    else:
        out["verdict"] = "empate_tecnico"
    return out


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


def load_phase2_summary() -> dict[str, Any] | None:
    path = ROOT / "results" / "tables" / "learned_vs_classic_fashion_mnist.csv"
    if not path.is_file():
        return None
    try:
        df2 = pd.read_csv(path)
        emb = df2[df2["representation"] == "embedding"]
        best_classic = df2[df2["compression_family"].isin(["jpeg", "png", "resize"])]
        if emb.empty or best_classic.empty:
            return None
        best_classic = best_classic.sort_values("utility_per_byte", ascending=False).iloc[0]
        e = emb.iloc[0]
        return {
            "path": str(path),
            "embedding_accuracy": float(e["accuracy"]),
            "embedding_bytes": float(e["bytes"]),
            "embedding_utility_per_byte": float(e["utility_per_byte"]),
            "best_classic": str(best_classic["representation"]),
            "best_classic_accuracy": float(best_classic["accuracy"]),
            "best_classic_bytes": float(best_classic["bytes"]),
            "best_classic_utility_per_byte": float(best_classic["utility_per_byte"]),
        }
    except Exception:
        return None


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
            label = str(row["representation"])
            if row["is_ssp_any"] == "yes":
                label += " (SSP)"
            plt.annotate(label, (row["bytes"], row["accuracy"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Bytes transmitidos serializados")
    plt.ylabel("Exactitud con receptor independiente")
    plt.title("Fase 3 (CIFAR-10): utilidad vs bytes")
    plt.grid(alpha=0.3)
    plt.legend(title="familia")
    plt.tight_layout()
    plt.savefig(figdir / "phase3_accuracy_vs_bytes.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    d2 = d.dropna(subset=["utility_per_byte"]).sort_values("utility_per_byte", ascending=False)
    colors = [fam_color.get(f, "#333333") for f in d2["compression_family"]]
    plt.bar(d2["representation"], d2["utility_per_byte"], color=colors)
    plt.ylabel("Utilidad por byte")
    plt.title("Fase 3 (CIFAR-10): utilidad por byte")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "phase3_utility_per_byte.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    learned = d[d["compression_family"].isin(["learned", "decision"])].sort_values("bytes")
    classic = d[d["compression_family"].isin(["jpeg", "png", "resize"])].sort_values("bytes")
    if not classic.empty:
        plt.plot(classic["bytes"], classic["accuracy"], "o-", color="#14345c", label="clasico")
    if not learned.empty:
        plt.plot(learned["bytes"], learned["accuracy"], "s--", color="#145c3a", label="aprendido/decision")
    for _, row in d.iterrows():
        plt.annotate(str(row["representation"]), (row["bytes"], row["accuracy"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Bytes transmitidos serializados")
    plt.ylabel("Exactitud")
    plt.title("Fase 3 (CIFAR-10): frontera aprendido vs clasico")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "phase3_learned_vs_classic_frontier.png", dpi=140)
    plt.close()


# ==========================================================
# Informes
# ==========================================================
def table_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def ssp_line(df: pd.DataFrame, col: str, label: str) -> str:
    sel = df[df[col] == "yes"]
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
) -> str:
    dv = dominance_verdict(df)
    verdict = str(dv["verdict"])
    text = verdict_text(verdict)
    phase2 = load_phase2_summary()

    lines: list[str] = []
    lines.append("# Informe Fase 3 — TaskNet-ELU (CIFAR-10)")
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

    if source != "cifar10":
        lines.append("> **AVISO:** esta corrida no uso CIFAR-10 real. Valida mecanica, no resultado cientifico principal.")
        lines.append("")
    if not str(backend).startswith("torch"):
        lines.append("> **AVISO:** sin torch, el embedding es proxy por proyeccion aleatoria. Usar torch para resultado principal.")
        lines.append("")

    lines.append("## Lectura SSP")
    lines.append(f"- Utilidad maxima observada: **U_max = {u_max:.4f}**")
    lines.append(ssp_line(df, "is_ssp_any", "SSP general"))
    lines.append(ssp_line(df, "is_ssp_decision", "SSP de decision"))
    lines.append(ssp_line(df, "is_ssp_semantic", "SSP semantico"))
    lines.append(ssp_line(df, "is_ssp_visual", "SSP visual"))
    lines.append("")

    lines.append("## Veredicto de dominancia")
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

    lines.append("## Comparacion con Fase 2 (Fashion-MNIST)")
    if phase2 is None:
        lines.append("No se encontro `results/tables/learned_vs_classic_fashion_mnist.csv`; no se pudo comparar automaticamente contra Fase 2.")
    else:
        lines.append(f"- Fase 2 embedding accuracy: **{phase2['embedding_accuracy']:.4f}**")
        lines.append(f"- Fase 2 embedding bytes: **{phase2['embedding_bytes']:.2f}**")
        lines.append(f"- Fase 2 embedding utilidad/byte: **{phase2['embedding_utility_per_byte']:.8f}**")
        lines.append(f"- Fase 2 mejor clasico por utilidad/byte: **`{phase2['best_classic']}`** "
                     f"(acc={phase2['best_classic_accuracy']:.4f}, bytes={phase2['best_classic_bytes']:.2f}, "
                     f"u/byte={phase2['best_classic_utility_per_byte']:.8f})")
        emb3 = df[df["representation"] == "embedding"].iloc[0]
        lines.append(f"- Fase 3 embedding accuracy: **{float(emb3['accuracy']):.4f}**")
        lines.append(f"- Fase 3 embedding bytes: **{float(emb3['bytes']):.2f}**")
        lines.append(f"- Fase 3 embedding utilidad/byte: **{float(emb3['utility_per_byte']):.8f}**")
    lines.append("")

    lines.append("## Resultados")
    lines.append(table_md(df))
    lines.append("")

    lines.append("## Notas de honestidad")
    lines.append("- CIFAR-10 es mas complejo que Fashion-MNIST: color, texturas, fondos y clases mas ambiguas. Exactitudes absolutas mas bajas son esperables.")
    lines.append("- JPEG opera aqui en un escenario mas natural que en Fashion-MNIST; si JPEG mejora, eso no refuta ELU, muestra dependencia de la fuente y la tarea.")
    lines.append("- El embedding sale de un transmisor entrenado en la tarea; sigue siendo una cota optimista aunque el receptor sea independiente.")
    lines.append("- Los parametros min/max del cuantizador se asumen compartidos como metadatos de calibracion y no se cuentan por muestra.")
    lines.append("- La latencia es inferencia del receptor por muestra. No incluye entrenamiento, compresion, serializacion ni red.")
    lines.append("- La utilidad por byte no basta por si sola; el veredicto exige conservar utilidad dentro de epsilon.")

    (reports_dir / "informe_fase_3.md").write_text("\n".join(lines), encoding="utf-8")

    cierre: list[str] = []
    cierre.append("# Cierre de Fase 3 — TaskNet-ELU")
    cierre.append("")
    cierre.append("## Estado")
    if source == "cifar10" and str(backend).startswith("torch"):
        cierre.append("Fase 3 ejecutada con CIFAR-10 real y backend torch.")
    else:
        cierre.append("Fase 3 ejecutada en modo no principal. Repetir con CIFAR-10 real y backend torch.")
    cierre.append("")
    cierre.append("## Comando ejecutado")
    cierre.append(f"`python scripts/03_cifar10_validation.py --epochs {args.epochs} --seeds {args.seeds}`")
    cierre.append("")
    cierre.append("## Archivos generados")
    cierre.append("- `results/tables/cifar10_learned_vs_classic.csv`")
    cierre.append("- `results/figures/phase3_accuracy_vs_bytes.png`")
    cierre.append("- `results/figures/phase3_utility_per_byte.png`")
    cierre.append("- `results/figures/phase3_learned_vs_classic_frontier.png`")
    cierre.append("- `reports/informe_fase_3.md`")
    cierre.append("- `reports/cierre_fase_3.md`")
    cierre.append("")
    cierre.append("## Veredicto")
    cierre.append(f"- {text}")
    cierre.append(f"- Pareto: {dv['pareto']}")
    cierre.append("")
    cierre.append("## Decision siguiente")
    cierre.append("Comparar el resultado contra Fase 2. Si el patron persiste, pasar a Fase 4: politica de transmision por umbral. Si cambia, documentar como evidencia de dependencia del SSP respecto a la complejidad de la fuente.")
    (reports_dir / "cierre_fase_3.md").write_text("\n".join(cierre), encoding="utf-8")

    return verdict


# ==========================================================
# Validacion fuerte
# ==========================================================
def validate(df: pd.DataFrame) -> None:
    required = {
        "dataset",
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
    missing_reps = expected_reps - set(df["representation"].astype(str))
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

    upb = df["utility_per_byte"].dropna()
    if not (upb >= 0).all():
        raise RuntimeError("utility_per_byte negativa")

    actionable = df[
        df["representation_group"].isin(["visual", "semantic"])
        & (df["is_silence_baseline"] == "no")
    ]
    for _, row in actionable.iterrows():
        if not str(row["receiver"]).startswith("independent"):
            raise RuntimeError(f"{row['representation']} no tiene receptor independiente: {row['receiver']}")

    label = df[df["representation"] == "label"].iloc[0]
    if label["receiver"] != "ground_truth":
        raise RuntimeError("label debe evaluarse contra ground_truth")

    silence = df[df["representation"] == "silence"].iloc[0]
    if silence["is_silence_baseline"] != "yes":
        raise RuntimeError("silence debe estar marcado como baseline")
    if not pd.isna(silence["utility_per_byte"]):
        raise RuntimeError("silence debe tener utility_per_byte NaN")


# ==========================================================
# CLI
# ==========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TaskNet-ELU Fase 3: CIFAR-10 aprendido vs clasico."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=0, dest="limit_train")
    parser.add_argument("--limit-test", type=int, default=0, dest="limit_test")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tables_dir, figs_dir, reports_dir = ensure_dirs()

    all_runs: list[list[dict[str, Any]]] = []
    source: str | None = None
    backend: str | None = None

    for seed in range(args.seeds):
        print(f"==> Corrida semilla {seed}")
        rows, source, backend = run_once(args, seed)
        all_runs.append(rows)

    agg = aggregate(all_runs)
    df, u_max = build_table(agg, args.seeds, str(source))
    validate(df)

    csv_path = tables_dir / "cifar10_learned_vs_classic.csv"
    df.to_csv(csv_path, index=False)
    make_plots(df, figs_dir)
    verdict = write_reports(df, u_max, args, str(source), str(backend), reports_dir)

    print("\n=== Resultados Fase 3 (CIFAR-10) ===")
    print(df.to_string(index=False))
    print("")
    print(f"U_max = {u_max:.6f}")
    print(f"Veredicto = {verdict}")
    print(f"CSV = {csv_path}")
    print(f"Figura accuracy vs bytes = {figs_dir / 'phase3_accuracy_vs_bytes.png'}")
    print(f"Figura utility per byte = {figs_dir / 'phase3_utility_per_byte.png'}")
    print(f"Figura frontera = {figs_dir / 'phase3_learned_vs_classic_frontier.png'}")
    print(f"Informe = {reports_dir / 'informe_fase_3.md'}")
    print(f"Cierre = {reports_dir / 'cierre_fase_3.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

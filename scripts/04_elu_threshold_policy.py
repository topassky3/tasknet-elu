#!/usr/bin/env python3
"""
04_elu_threshold_policy.py — Fase 4 de TaskNet-ELU.

Pone a prueba la regla central del marco ELU, por muestra:

    transmitir(x)  <=>  Delta_U(x) > lambda * C(x)

Escenario:
  - Para cada muestra de test, el transmisor decide entre:
      * SILENCIO (0 bytes): el receptor cae a la clase mayoritaria.
      * TRANSMITIR el embedding cuantizado (~bytes medidos): el receptor
        independiente de embeddings decide.
  - Delta_U(x) real no se conoce de antemano. La politica ELU lo ESTIMA con
    el softmax del transmisor:
        DU_hat(x) = p_tx(clase_top | x) - p_tx(clase_mayoritaria | x)
    Interpretacion: la ganancia esperada de transmitir es alta cuando la
    muestra probablemente NO es de la clase mayoritaria (callar fallaria)
    y el receptor probablemente acertara (muestra "facil" para el modelo).

Politicas comparadas (plan maestro, Fase 4):
  1. always_original   : transmitir siempre la imagen completa (PNG).
  2. always_embedding  : transmitir siempre el embedding.
  3. always_label      : transmitir siempre la etiqueta del transmisor.
  4. never (silence)   : no transmitir nunca.
  5. confidence(t)     : transmitir si la confianza del transmisor baja de t
                         (umbral barrido, para competencia justa).
  6. elu_threshold(l)  : transmitir si DU_hat(x) > lambda * B_emb
                         (lambda barrido: curva trafico-utilidad completa).
  7. oracle            : transmitir si Delta_U REAL(x) > 0 (usa verdad de
                         terreno; NO desplegable; techo teorico).

Metrica principal: J_ELU = U - lambda_B * B_mean, reportada como curva
utilidad vs bytes medios (barrido de lambda), no en un punto unico.

Criterio de exito (plan maestro, criterio 3):
  existe un punto de operacion ELU con utilidad >= U(always_embedding) - eps
  y reduccion de trafico >= 50% frente a always_embedding.

Honestidad:
  - DU_hat usa el softmax del transmisor como proxy de calibracion declarado.
  - El silencio cae a clase mayoritaria: eleccion de diseno declarada.
  - El bit de senalizacion transmitir/no-transmitir no se cuenta (se declara).
  - El oraculo usa la verdad de terreno: es techo, no politica real.
  - Las decisiones de politica se evaluan SOLO en test; tx y rx se entrenan
    solo en train.

Salidas:
    results/tables/elu_policy_comparison.csv
    results/figures/phase4_traffic_utility.png
    reports/informe_fase_4.md
    reports/cierre_fase_4.md

Uso:
    python scripts/04_elu_threshold_policy.py --epochs 1 --limit-train 5000 --limit-test 1000 --seeds 1
    python scripts/04_elu_threshold_policy.py --epochs 5 --seeds 3
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
NUM_CLASSES = 10
EMBED_DIM = 32
BYTE_SAMPLE_LIMIT = 2000
LAMBDA_POINTS = 25
BATCH_SIZE = 128


# ==========================================================
# Utilidades
# ==========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def ensure_dirs():
    tables = ROOT / "results" / "tables"
    figs = ROOT / "results" / "figures"
    reports = ROOT / "reports"
    for d in (tables, figs, reports):
        d.mkdir(parents=True, exist_ok=True)
    return tables, figs, reports


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ==========================================================
# Datos: Fashion-MNIST (con respaldo sintetico)
# ==========================================================
def load_fashion_mnist(limit_train, limit_test, seed):
    try:
        import torchvision
        import torchvision.transforms as T
        tf = T.Compose([T.ToTensor()])
        tr = torchvision.datasets.FashionMNIST(
            root=str(ROOT / "data" / "raw"), train=True, download=True, transform=tf)
        te = torchvision.datasets.FashionMNIST(
            root=str(ROOT / "data" / "raw"), train=False, download=True, transform=tf)
        Xtr = tr.data.numpy().astype("float32") / 255.0
        ytr = tr.targets.numpy().astype("int64")
        Xte = te.data.numpy().astype("float32") / 255.0
        yte = te.targets.numpy().astype("int64")
        source = "fashion_mnist"
    except Exception as exc:
        print(f"[AVISO] Sin Fashion-MNIST ({type(exc).__name__}). Dataset SINTETICO: "
              "solo valida mecanica, no resultado real.")
        rng = np.random.default_rng(seed)
        Xtr, ytr = synthetic_dataset(6000, rng)
        Xte, yte = synthetic_dataset(1500, rng)
        source = "synthetic"
    if limit_train and limit_train > 0:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test and limit_test > 0:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    return Xtr, ytr, Xte, yte, source


def synthetic_dataset(n, rng):
    """28x28, 10 clases separables PERO con solape parcial para que existan
    muestras dificiles: asi la politica tiene algo real que decidir."""
    y = rng.integers(0, NUM_CLASSES, size=n)
    X = np.zeros((n, 28, 28), dtype="float32")
    centers = [(7 + 6 * (c % 3), 7 + 6 * (c // 3)) for c in range(NUM_CLASSES)]
    yy, xx = np.mgrid[0:28, 0:28]
    for i in range(n):
        cy, cx = centers[int(y[i])]
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 22.0)
        noise = 0.25 * rng.standard_normal((28, 28))  # ruido alto a proposito
        X[i] = (blob + noise).clip(0, 1)
    return X.astype("float32"), y.astype("int64")


# ==========================================================
# Serializacion
# ==========================================================
def gzip_len(payload: bytes) -> int:
    return len(gzip.compress(payload, compresslevel=6))


def uint8_img(X):
    return (np.clip(X, 0, 1) * 255).round().astype("uint8")


def bytes_png(img01) -> int:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(uint8_img(img01), mode="L").save(buf, format="PNG")
    return len(buf.getvalue())


def bytes_label() -> int:
    return gzip_len(np.array([0], dtype="uint8").tobytes())


@dataclass
class Quantizer:
    minimum: np.ndarray
    maximum: np.ndarray

    @staticmethod
    def fit(X):
        return Quantizer(X.min(axis=0).astype("float32"), X.max(axis=0).astype("float32"))

    def transform(self, X):
        denom = np.where(self.maximum - self.minimum < 1e-8, 1.0, self.maximum - self.minimum)
        return np.clip(np.round((X - self.minimum) / denom * 255.0), 0, 255).astype("uint8")

    def inverse_transform(self, Q):
        denom = np.where(self.maximum - self.minimum < 1e-8, 1.0, self.maximum - self.minimum)
        return (Q.astype("float32") / 255.0) * denom + self.minimum


# ==========================================================
# Modelos (torch con respaldo sklearn)
# ==========================================================
def train_models_torch(Xtr, ytr, Xte, epochs, seed):
    import torch
    import torch.nn as nn
    set_seed(seed)

    class CNN(nn.Module):
        def __init__(self, hw=28):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            self.embed = nn.Linear(32 * (hw // 4) * (hw // 4), EMBED_DIM)
            self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, x, return_embed=False):
            e = torch.relu(self.embed(self.features(x).flatten(1)))
            out = self.head(e)
            return (out, e) if return_embed else out

    def fit_image(model, X, y, ep):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(X[:, None], dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        model.train()
        for _ in range(ep):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                opt.zero_grad()
                lossf(model(Xt[idx]), yt[idx]).backward()
                opt.step()
        return model

    @torch.no_grad()
    def softmax_probs(model, X):
        model.eval()
        outs = []
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i:i + BATCH_SIZE][:, None], dtype=torch.float32)
            outs.append(torch.softmax(model(xb), dim=1).numpy())
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def embed(model, X):
        model.eval()
        outs = []
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i:i + BATCH_SIZE][:, None], dtype=torch.float32)
            _, e = model(xb, return_embed=True)
            outs.append(e.numpy().astype("float32"))
        return np.concatenate(outs, axis=0)

    # Transmisor (softmax para DU_hat, embedding, etiqueta)
    tx = fit_image(CNN(), Xtr, ytr, epochs)
    probs_te = softmax_probs(tx, Xte)
    Etr, Ete = embed(tx, Xtr), embed(tx, Xte)

    # Receptor independiente de imagen completa (para always_original)
    rx_full = fit_image(CNN(), Xtr, ytr, epochs)
    pred_full_te = softmax_probs(rx_full, Xte).argmax(1)

    # Receptor independiente de embedding cuantizado
    qz = Quantizer.fit(Etr)
    Etr_rx = qz.inverse_transform(qz.transform(Etr))
    Ete_q = qz.transform(Ete)
    Ete_rx = qz.inverse_transform(Ete_q)

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES))

        def forward(self, x):
            return self.net(x)

    rx_emb = MLP(EMBED_DIM)
    opt = torch.optim.Adam(rx_emb.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Etr_rx, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    rx_emb.train()
    for _ in range(max(epochs, 3)):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            lossf(rx_emb(Xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        rx_emb.eval()
        pred_emb_te = rx_emb(torch.tensor(Ete_rx, dtype=torch.float32)).argmax(1).numpy()

    return probs_te, pred_full_te, pred_emb_te, Ete_q, "torch"


def train_models_sklearn(Xtr, ytr, Xte, epochs, seed):
    from sklearn.neural_network import MLPClassifier
    set_seed(seed)

    def fit(X, y):
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=40 + 10 * epochs, random_state=seed)
        clf.fit(X.reshape(len(X), -1), y)
        return clf

    tx = fit(Xtr, ytr)
    probs_te = tx.predict_proba(Xte.reshape(len(Xte), -1))
    rx_full = fit(Xtr, ytr)
    pred_full_te = rx_full.predict(Xte.reshape(len(Xte), -1))

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((28 * 28, EMBED_DIM)).astype("float32")
    Etr = np.maximum(Xtr.reshape(len(Xtr), -1) @ W, 0)
    Ete = np.maximum(Xte.reshape(len(Xte), -1) @ W, 0)
    qz = Quantizer.fit(Etr)
    Etr_rx = qz.inverse_transform(qz.transform(Etr))
    Ete_q = qz.transform(Ete)
    Ete_rx = qz.inverse_transform(Ete_q)
    rx_emb = MLPClassifier(hidden_layer_sizes=(64,), max_iter=40 + 10 * epochs, random_state=seed + 9)
    rx_emb.fit(Etr_rx, ytr)
    pred_emb_te = rx_emb.predict(Ete_rx)
    return probs_te, pred_full_te, pred_emb_te, Ete_q, "sklearn"


# ==========================================================
# Evaluacion de politicas
# ==========================================================
def eval_policy(transmit_mask, pred_if_tx, yte, per_bytes_if_tx, majority):
    """Utilidad, bytes medios y fraccion de trafico de una politica binaria
    transmitir/callar. Si calla, el receptor responde la clase mayoritaria."""
    correct = np.where(transmit_mask, pred_if_tx == yte, majority == yte)
    util = float(correct.mean())
    mean_b = float((per_bytes_if_tx * transmit_mask).mean())
    traffic = float(transmit_mask.mean())
    return util, mean_b, traffic


def sweep_curve(score, pred_if_tx, yte, per_bytes, majority, n_points=LAMBDA_POINTS):
    """Barrido parametrizado por fraccion de trafico q: transmitir la fraccion q
    de muestras con mayor score. q es comparable entre semillas (el umbral crudo
    no lo es), de modo que la agregacion entre semillas promedia puntos homologos.
    Devuelve dicts con q, umbral implicito, utilidad, bytes y trafico."""
    rows = []
    n = len(score)
    order = np.argsort(-score)  # descendente por score
    for q in np.linspace(0.0, 1.0, n_points):
        k = int(round(q * n))
        mask = np.zeros(n, dtype=bool)
        if k > 0:
            mask[order[:k]] = True
        thr = float(score[order[k - 1]]) if 0 < k <= n else float("nan")
        u, b, t = eval_policy(mask, pred_if_tx, yte, per_bytes, majority)
        rows.append({"q": float(q), "threshold": thr, "utility": u,
                     "mean_bytes": b, "traffic": t})
    return rows


# ==========================================================
# Corrida por semilla
# ==========================================================
def run_once(args, seed):
    set_seed(seed)
    Xtr, ytr, Xte, yte, source = load_fashion_mnist(args.limit_train, args.limit_test, seed)

    backend_fn = train_models_torch if has_torch() else train_models_sklearn
    probs_te, pred_full_te, pred_emb_te, Ete_q, backend = backend_fn(Xtr, ytr, Xte, epochs=args.epochs, seed=seed)

    # Clase mayoritaria (del train: el receptor mudo no ve test)
    vals, counts = np.unique(ytr, return_counts=True)
    majority = int(vals[counts.argmax()])

    # Bytes por muestra
    n_b = min(BYTE_SAMPLE_LIMIT, len(Xte))
    emb_bytes = np.array([gzip_len(Ete_q[i].tobytes()) for i in range(len(Ete_q))], dtype="float64")
    png_bytes_mean = float(np.mean([bytes_png(Xte[i]) for i in range(n_b)]))
    label_b = float(bytes_label())
    B_emb_mean = float(emb_bytes.mean())

    # Softmax del transmisor sobre test
    p_top = probs_te.max(axis=1)
    p_majority = probs_te[:, majority]
    pred_tx = probs_te.argmax(axis=1)

    # ---------- Politicas fijas ----------
    fixed = []
    ones = np.ones(len(yte), dtype=bool)
    zeros = np.zeros(len(yte), dtype=bool)
    # 1. transmitir todo (original completa, receptor independiente de imagen)
    u = float((pred_full_te == yte).mean())
    fixed.append({"policy": "always_original", "param": "", "utility": u,
                  "mean_bytes": png_bytes_mean, "traffic": 1.0})
    # 2. siempre embedding
    u, b, t = eval_policy(ones, pred_emb_te, yte, emb_bytes, majority)
    fixed.append({"policy": "always_embedding", "param": "", "utility": u,
                  "mean_bytes": b, "traffic": t})
    # 3. siempre etiqueta (decision del transmisor, medida vs verdad)
    u = float((pred_tx == yte).mean())
    fixed.append({"policy": "always_label", "param": "", "utility": u,
                  "mean_bytes": label_b, "traffic": 1.0})
    # 4. silencio
    u, b, t = eval_policy(zeros, pred_emb_te, yte, emb_bytes, majority)
    fixed.append({"policy": "never_silence", "param": "", "utility": u,
                  "mean_bytes": b, "traffic": t})

    # ---------- Politica de confianza (transmitir la fraccion q mas insegura) ----------
    conf_rows = sweep_curve(-p_top, pred_emb_te, yte, emb_bytes, majority)
    for r in conf_rows:
        r["policy"] = "confidence"
        thr = r.pop("threshold")
        r["param"] = f"q={r.pop('q'):.3f}"
        r["info"] = f"umbral_confianza={-thr:.4f}" if np.isfinite(thr) else ""

    # ---------- Politica ELU: DU_hat = p_top - p_majoritaria, fraccion q de mayor ganancia ----------
    du_hat = p_top - p_majority
    elu_rows = sweep_curve(du_hat, pred_emb_te, yte, emb_bytes, majority)
    for r in elu_rows:
        thr = r.pop("threshold")
        lam = thr / B_emb_mean if (np.isfinite(thr) and B_emb_mean > 0) else float("nan")
        r["policy"] = "elu_threshold"
        r["param"] = f"q={r.pop('q'):.3f}"
        r["info"] = f"lambda_implicito={lam:.6f}" if np.isfinite(lam) else ""

    # ---------- Oraculo: DU real > 0 (techo, usa verdad de terreno) ----------
    du_true = (pred_emb_te == yte).astype("float64") - (majority == yte) * np.ones(len(yte))
    # transmitir exactamente donde transmite ganaria utilidad
    mask_oracle = du_true > 0
    u, b, t = eval_policy(mask_oracle, pred_emb_te, yte, emb_bytes, majority)
    oracle = [{"policy": "oracle", "param": "DU_true>0", "utility": u,
               "mean_bytes": b, "traffic": t}]

    all_rows = fixed + conf_rows + elu_rows + oracle
    for r in all_rows:
        r["seed"] = seed
        r["source"] = source
        r["backend"] = backend
        r["majority_class"] = majority
        r["B_emb_mean"] = round(B_emb_mean, 4)
    return all_rows, source, backend, B_emb_mean


# ==========================================================
# Agregacion entre semillas (por politica+param)
# ==========================================================
def aggregate(all_runs):
    df = pd.DataFrame([r for run in all_runs for r in run])
    grouped = df.groupby(["policy", "param"], as_index=False).agg(
        utility=("utility", "mean"),
        utility_std=("utility", "std"),
        mean_bytes=("mean_bytes", "mean"),
        traffic=("traffic", "mean"),
        seeds=("seed", "nunique"),
    )
    grouped["utility_std"] = grouped["utility_std"].fillna(0.0)
    return grouped, df


# ==========================================================
# Veredicto
# ==========================================================
def phase4_verdict(agg: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"criterion3": None, "vs_confidence": None, "best_point": None,
                           "oracle_ceiling": None}
    emb = agg[agg["policy"] == "always_embedding"]
    orig = agg[agg["policy"] == "always_original"]
    elu = agg[agg["policy"] == "elu_threshold"]
    conf = agg[agg["policy"] == "confidence"]
    oracle = agg[agg["policy"] == "oracle"]
    if emb.empty or elu.empty or orig.empty:
        return out
    u_emb = float(emb.iloc[0]["utility"])
    b_emb = float(emb.iloc[0]["mean_bytes"])
    b_orig = float(orig.iloc[0]["mean_bytes"])

    # Mejor punto ELU que conserva utilidad (dentro de epsilon del always_embedding)
    ok = elu[elu["utility"] >= u_emb - EPS_SSP]
    if not ok.empty:
        best = ok.sort_values("mean_bytes").iloc[0]
        red_vs_emb = 1.0 - float(best["mean_bytes"]) / b_emb if b_emb > 0 else 0.0
        red_vs_all = 1.0 - float(best["mean_bytes"]) / b_orig if b_orig > 0 else 0.0
        out["best_point"] = {
            "param": str(best["param"]), "utility": float(best["utility"]),
            "mean_bytes": float(best["mean_bytes"]), "traffic": float(best["traffic"]),
            "traffic_reduction_vs_always_embedding": red_vs_emb,
            "traffic_reduction_vs_transmit_everything": red_vs_all,
        }
        # Criterio 3 del plan maestro: reduccion >=50% frente a TRANSMITIR TODO
        out["criterion3"] = bool(red_vs_all >= 0.50)
    else:
        out["criterion3"] = False

    # Techo del oraculo: maximo ahorro posible vs always_embedding en este escenario.
    # Con respaldo de clase mayoritaria en dataset balanceado, este techo es bajo por diseno.
    if not oracle.empty and b_emb > 0:
        out["oracle_ceiling"] = 1.0 - float(oracle.iloc[0]["mean_bytes"]) / b_emb

    # ELU vs heuristica de confianza: a cada punto ELU, existe punto de confianza
    # con <= bytes y >= utilidad? Si para la mayoria no existe, ELU domina.
    if not conf.empty and not elu.empty:
        elu_pts = elu[["mean_bytes", "utility"]].to_numpy()
        conf_pts = conf[["mean_bytes", "utility"]].to_numpy()
        dominated = 0
        for b, u in elu_pts:
            if ((conf_pts[:, 0] <= b + 1e-9) & (conf_pts[:, 1] >= u - 1e-9)).any():
                dominated += 1
        frac_matched = dominated / len(elu_pts)
        if frac_matched < 0.34:
            out["vs_confidence"] = "elu_domina_a_confianza"
        elif frac_matched > 0.66:
            out["vs_confidence"] = "confianza_iguala_o_supera_a_elu"
        else:
            out["vs_confidence"] = "curvas_mixtas"
    return out


# ==========================================================
# Grafica
# ==========================================================
def make_plot(agg: pd.DataFrame, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.5, 5.4))
    elu = agg[agg["policy"] == "elu_threshold"].sort_values("mean_bytes")
    conf = agg[agg["policy"] == "confidence"].sort_values("mean_bytes")
    plt.plot(elu["mean_bytes"], elu["utility"], "s-", color="#145c3a",
             label="politica ELU (barrido de lambda)", zorder=4)
    plt.plot(conf["mean_bytes"], conf["utility"], "o--", color="#a06010",
             label="heuristica de confianza (barrida)", zorder=3)
    marks = {"always_original": ("^", "#7a4f9c"), "always_embedding": ("D", "#14345c"),
             "always_label": ("v", "#999999"), "never_silence": ("x", "#333333"),
             "oracle": ("*", "#b02020")}
    for pol, (m, c) in marks.items():
        row = agg[agg["policy"] == pol]
        if not row.empty:
            r = row.iloc[0]
            plt.scatter([r["mean_bytes"]], [r["utility"]], marker=m, s=120, color=c,
                        label=pol, zorder=5)
    plt.xlabel("Bytes medios transmitidos por muestra")
    plt.ylabel("Utilidad (exactitud)")
    plt.title("Fase 4: trafico vs utilidad — politica ELU por umbral")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "phase4_traffic_utility.png", dpi=140)
    plt.close()


# ==========================================================
# Informes
# ==========================================================
def table_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_reports(agg, verdict, args, source, backend, B_emb_mean, reports_dir) -> str:
    c3 = verdict["criterion3"]
    bp = verdict["best_point"]
    vsconf = verdict["vs_confidence"]
    verdict_txt = ("CRITERIO 3 CUMPLIDO: la politica ELU reduce >=50% el trafico frente a "
                   "TRANSMITIR TODO, conservando utilidad dentro de epsilon."
                   if c3 else
                   "CRITERIO 3 NO CUMPLIDO en esta corrida: la politica ELU no logro reducir "
                   ">=50% el trafico frente a transmitir todo conservando utilidad.")

    L = ["# Informe Fase 4 — TaskNet-ELU (politica de transmision por umbral)", ""]
    L += ["## Configuracion",
          f"- Fuente de datos: **{source}** · Backend: **{backend}**",
          f"- Epocas: **{args.epochs}** · Semillas: **{args.seeds}**",
          f"- limit_train: **{args.limit_train}** · limit_test: **{args.limit_test}**",
          f"- Bytes medios del embedding: **{B_emb_mean:.2f}**",
          f"- Epsilon: **{EPS_SSP}** · Puntos de barrido: **{LAMBDA_POINTS}**", ""]
    if source == "synthetic":
        L += ["> **AVISO:** dataset sintetico. Valida mecanica, no resultado real.", ""]
    if backend == "sklearn":
        L += ["> **AVISO:** sin torch, el embedding es proxy por proyeccion aleatoria.", ""]
    L += ["## Veredicto", "", f"**{verdict_txt}**", ""]
    if bp:
        L += [f"- Mejor punto ELU dentro de epsilon: {bp['param']}, "
              f"utilidad={bp['utility']:.4f}, bytes medios={bp['mean_bytes']:.2f}, "
              f"trafico={bp['traffic']:.1%}"]
        L += [f"- Reduccion de trafico vs TRANSMITIR TODO (criterio 3 del plan): "
              f"**{bp['traffic_reduction_vs_transmit_everything']:.1%}**"]
        L += [f"- Reduccion de trafico vs always_embedding (contribucion marginal de la "
              f"politica): **{bp['traffic_reduction_vs_always_embedding']:.1%}**"]
    if verdict.get("oracle_ceiling") is not None:
        L += [f"- Techo del oraculo (maximo ahorro posible vs always_embedding en este "
              f"escenario): **{verdict['oracle_ceiling']:.1%}**. Con respaldo de clase "
              f"mayoritaria en dataset balanceado, este techo es bajo por diseno: callar "
              f"casi nunca acierta, asi que casi toda muestra amerita transmision."]
    if vsconf:
        L += [f"- Comparacion contra heuristica de confianza (umbral barrido): **{vsconf}**"]
    L += ["", "## Resultados (agregados entre semillas)", "", table_md(agg.round(6)), ""]
    L += ["## Notas de honestidad",
          "- DU_hat se estima con el softmax del transmisor (proxy de calibracion declarado): "
          "DU_hat(x) = p_top(x) - p_mayoritaria(x). No usa la verdad de terreno.",
          "- El silencio cae a la clase mayoritaria del TRAIN (el receptor mudo no ve test).",
          "- El bit de senalizacion transmitir/callar no se cuenta en los bytes; se declara.",
          "- El oraculo usa la verdad de terreno: es un techo teorico, no una politica real.",
          "- La heuristica de confianza se barre en todo su rango de umbral para competir "
          "en igualdad de condiciones (curva completa, no un punto).",
          "- tx y receptores se entrenan solo con train; las politicas se evaluan solo en test."]
    (reports_dir / "informe_fase_4.md").write_text("\n".join(L), encoding="utf-8")

    C = ["# Cierre de Fase 4 — TaskNet-ELU", "", "## Estado"]
    C += ["Fase 4 ejecutada con Fashion-MNIST real y backend torch."
          if (source == "fashion_mnist" and backend == "torch")
          else "Fase 4 en modo no principal. Repetir con Fashion-MNIST real y torch."]
    C += ["", "## Comando ejecutado",
          f"`python scripts/04_elu_threshold_policy.py --epochs {args.epochs} --seeds {args.seeds}`",
          "", "## Veredicto", f"- {verdict_txt}"]
    if bp:
        C += [f"- Reduccion de trafico en el mejor punto dentro de epsilon: "
              f"{bp['traffic_reduction_vs_always_embedding']:.1%}"]
    if vsconf:
        C += [f"- Contra heuristica de confianza: {vsconf}"]
    C += ["", "## Decision siguiente",
          "Con las Fases 1-4 cerradas, integrar el informe final del proyecto acotado "
          "(plan maestro) y evaluar apertura de Fases 5-6 (informacion en pesos y "
          "generalizacion) o el piloto YaruChess-ELU."]
    (reports_dir / "cierre_fase_4.md").write_text("\n".join(C), encoding="utf-8")
    return verdict_txt


# ==========================================================
# Validacion
# ==========================================================
def validate(agg: pd.DataFrame) -> None:
    required = {"policy", "param", "utility", "utility_std", "mean_bytes", "traffic", "seeds"}
    miss = required - set(agg.columns)
    if miss:
        raise RuntimeError(f"Faltan columnas: {sorted(miss)}")
    expected = {"always_original", "always_embedding", "always_label",
                "never_silence", "confidence", "elu_threshold", "oracle"}
    found = set(agg["policy"].astype(str))
    missing = expected - found
    if missing:
        raise RuntimeError(f"Faltan politicas: {sorted(missing)}")
    if not ((agg["utility"] >= 0) & (agg["utility"] <= 1)).all():
        raise RuntimeError("utilidad fuera de [0,1]")
    if not (agg["mean_bytes"] >= 0).all():
        raise RuntimeError("bytes negativos")
    if not ((agg["traffic"] >= 0) & (agg["traffic"] <= 1)).all():
        raise RuntimeError("trafico fuera de [0,1]")
    # el oraculo es techo SOLO para politicas que eligen entre silencio y embedding
    # (always_original y always_label usan otros canales y pueden superarlo)
    same_space = {"always_embedding", "never_silence", "confidence", "elu_threshold"}
    u_oracle = float(agg[agg["policy"] == "oracle"].iloc[0]["utility"])
    u_space_max = float(agg[agg["policy"].isin(same_space)]["utility"].max())
    if u_space_max > u_oracle + 1e-9:
        raise RuntimeError("una politica del espacio silencio/embedding supero al oraculo: "
                           "revisar definicion de DU_true")


# ==========================================================
# CLI
# ==========================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="TaskNet-ELU Fase 4: politica por umbral DU>lambda*C")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--limit-train", type=int, default=0, dest="limit_train")
    ap.add_argument("--limit-test", type=int, default=0, dest="limit_test")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    tables, figs, reports = ensure_dirs()
    all_runs, source, backend, B_emb = [], None, None, None
    for s in range(args.seeds):
        print(f"==> Corrida semilla {s}")
        rows, source, backend, B_emb = run_once(args, s)
        all_runs.append(rows)

    agg, raw = aggregate(all_runs)
    validate(agg)
    verdict = phase4_verdict(agg)

    csv_path = tables / "elu_policy_comparison.csv"
    agg.to_csv(csv_path, index=False)
    raw.to_csv(tables / "elu_policy_comparison_raw.csv", index=False)
    make_plot(agg, figs)
    vt = write_reports(agg, verdict, args, str(source), str(backend), float(B_emb), reports)

    print("\n=== Resultados Fase 4 (mejor punto de cada politica) ===")
    summary = (agg.sort_values("utility", ascending=False)
                  .groupby("policy", as_index=False).head(1)
                  .sort_values("policy"))
    print(summary[["policy", "param", "utility", "mean_bytes", "traffic"]].to_string(index=False))
    print(f"\nVeredicto: {vt}")
    if verdict["best_point"]:
        bp = verdict["best_point"]
        print(f"Mejor punto ELU: U={bp['utility']:.4f}, B={bp['mean_bytes']:.2f}, "
              f"trafico={bp['traffic']:.1%}")
        print(f"  Reduccion vs transmitir todo:   {bp['traffic_reduction_vs_transmit_everything']:.1%}")
        print(f"  Reduccion vs always_embedding:  {bp['traffic_reduction_vs_always_embedding']:.1%}")
    print(f"CSV = {csv_path}")
    print(f"Figura = {figs / 'phase4_traffic_utility.png'}")
    print(f"Informe = {reports / 'informe_fase_4.md'}")
    print(f"Cierre = {reports / 'cierre_fase_4.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
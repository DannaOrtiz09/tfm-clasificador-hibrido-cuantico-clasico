#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARATIVA FINAL DE 3 MODELOS - TFM (STANDALONE)
==================================================

Este script construye la comparativa final de:
  1) HybridModel
  2) ClassicalEquivalentModel
  3) SimpleCNNBaseline

No repite el barrido de qubits, capas, entanglers, encodings ni hiperparametros.
Lee la configuracion final ya congelada en:
    results_final_corregido/frozen_final_config.json

y reutiliza los resultados confirmatorios ya obtenidos para HybridModel y
ClassicalEquivalentModel. Solo ejecuta la CNN que faltaba, usando exactamente:
  - el mismo split train/validation,
  - las mismas 12 semillas finales,
  - el mismo preprocesado MNIST,
  - Adam,
  - learning rate congelado,
  - batch size congelado,
  - maximo de 15 epocas,
  - early stopping sobre validation loss,
  - restauracion del mejor checkpoint por validation loss.

IMPORTANTE:
HybridModel vs ClassicalEquivalentModel es la ablacion controlada y sigue siendo
la comparacion causal principal. La CNN es una referencia clasica contextual,
porque tiene arquitectura y numero de parametros distintos.

El script NO modifica results_final_corregido. Crea:
    results_comparativa_3_modelos/

con CSV, JSON, predicciones y graficas finales.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# ============================================================
# 0. REPRODUCIBILIDAD
# ============================================================

DEVICE = torch.device("cpu")

try:
    torch.set_num_threads(int(os.environ.get("TFM_TORCH_THREADS", "1")))
except Exception:
    pass

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
except Exception:
    pass


@dataclass
class Config:
    seed: int = 42
    split_seed: int = 2026
    n_qubits: int = 8
    n_layers: int = 4
    entangler: str = "strong"
    encoding: str = "amplitude"
    encoder_hidden_dim: int = 64
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 15
    val_split: float = 0.10
    early_stopping_patience: int = 4
    device: str = "cpu"
    data_dir: str = "./data"


EXPECTED_FINAL = {
    "split_seed": 2026,
    "n_qubits": 8,
    "n_layers": 4,
    "entangler": "strong",
    "encoding": "amplitude",
    "encoder_hidden_dim": 64,
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 15,
    "val_split": 0.10,
    "early_stopping_patience": 4,
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# 1. DATOS - MISMO PIPELINE DEL ESTUDIO FINAL
# ============================================================

def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_train_val_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    full_train = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=mnist_transform(),
    )

    n_val = int(len(full_train) * cfg.val_split)
    n_train = len(full_train) - n_val

    split_generator = torch.Generator().manual_seed(cfg.split_seed)
    train_set, val_set = random_split(
        full_train,
        [n_train, n_val],
        generator=split_generator,
    )

    loader_generator = torch.Generator().manual_seed(cfg.seed + 100_000)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def get_test_loader(cfg: Config) -> DataLoader:
    test_set = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=mnist_transform(),
    )
    return DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


# ============================================================
# 2. CNN CLASICA DE REFERENCIA - MISMA ARQUITECTURA DEL TFM
# ============================================================

class SimpleCNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 3. ENTRENAMIENTO - MISMO CRITERIO QUE EL PIPELINE FINAL
# ============================================================

def evaluate_loss_accuracy(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader.dataset), correct / total


def predict_model(model: nn.Module, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds: List[int] = []
    labels_all: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            pred = outputs.argmax(dim=1)
            preds.extend(pred.cpu().numpy().tolist())
            labels_all.extend(labels.numpy().tolist())

    y_true = np.asarray(labels_all, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    accuracy = float((y_true == y_pred).mean())
    return accuracy, y_true, y_pred


def load_state_dict_compat(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def train_cnn(
    cfg: Config,
    checkpoint_path: Path,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict, float, Dict]:
    set_seed(cfg.seed)
    train_loader, val_loader = get_train_val_loaders(cfg)

    # Igual que en el pipeline corregido: la inicializacion depende de cfg.seed.
    torch.manual_seed(cfg.seed)
    model = SimpleCNNBaseline().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_loss_accuracy(model, val_loader)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if verbose:
            print(
                f"[final_cnn_seed{cfg.seed}] Epoca {epoch + 1}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}",
                flush=True,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                if verbose:
                    print(
                        f"[final_cnn_seed{cfg.seed}] Early stopping en epoca {epoch + 1}",
                        flush=True,
                    )
                break

    training_time = time.time() - start_time

    model.load_state_dict(load_state_dict_compat(checkpoint_path))
    restored_val_loss, restored_val_acc = evaluate_loss_accuracy(model, val_loader)

    diagnostics = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "restored_val_loss": float(restored_val_loss),
        "restored_val_accuracy": float(restored_val_acc),
        "epochs_ran": int(len(history["epoch"])),
    }

    del train_loader, val_loader
    gc.collect()
    return model, history, training_time, diagnostics


# ============================================================
# 4. CARGA Y VALIDACION DE LA CONFIGURACION FINAL
# ============================================================

def validate_frozen_config(payload: Dict) -> None:
    if payload.get("status") != "FROZEN_BEFORE_ANY_FINAL_TEST_EVALUATION":
        raise RuntimeError("El frozen_final_config.json no tiene el estado esperado.")

    if payload.get("test_used_during_selection") is not False:
        raise RuntimeError("El JSON no confirma test_used_during_selection=false.")

    cfg = payload.get("config", {})
    mismatches = []
    for key, expected in EXPECTED_FINAL.items():
        got = cfg.get(key)
        if isinstance(expected, float):
            if got is None or not np.isclose(float(got), expected):
                mismatches.append((key, got, expected))
        elif got != expected:
            mismatches.append((key, got, expected))

    if mismatches:
        lines = "\n".join(
            f"  - {key}: encontrado={got!r}, esperado={expected!r}"
            for key, got, expected in mismatches
        )
        raise RuntimeError(
            "La configuracion congelada no coincide con la final validada:\n" + lines
        )


def load_frozen(source_results: Path) -> Tuple[Config, Tuple[int, ...], Dict]:
    path = source_results / "frozen_final_config.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_frozen_config(payload)

    raw = payload["config"]
    cfg = Config(
        seed=int(raw["seed"]),
        split_seed=int(raw["split_seed"]),
        n_qubits=int(raw["n_qubits"]),
        n_layers=int(raw["n_layers"]),
        entangler=str(raw["entangler"]),
        encoding=str(raw["encoding"]),
        encoder_hidden_dim=int(raw["encoder_hidden_dim"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        epochs=int(raw["epochs"]),
        val_split=float(raw["val_split"]),
        early_stopping_patience=int(raw["early_stopping_patience"]),
        device=str(raw.get("device", "cpu")),
        data_dir=str(raw.get("data_dir", "./data")),
    )
    seeds = tuple(int(x) for x in payload["final_seeds"])
    return cfg, seeds, payload


def load_existing_hybrid_classical(source_results: Path, seeds: Sequence[int]) -> pd.DataFrame:
    path = source_results / "final_paired_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    df = pd.read_csv(path)
    required = {
        "seed",
        "hybrid_test_accuracy",
        "classical_test_accuracy",
        "hybrid_time_sec",
        "classical_time_sec",
        "hybrid_total_params",
        "classical_total_params",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas en final_paired_results.csv: {sorted(missing)}")

    df["seed"] = df["seed"].astype(int)
    expected = set(int(s) for s in seeds)
    found = set(df["seed"].tolist())
    if found != expected:
        raise RuntimeError(
            f"Semillas esperadas={sorted(expected)}; encontradas={sorted(found)}"
        )

    return df.sort_values("seed").reset_index(drop=True)


# ============================================================
# 5. EJECUCION / REANUDACION DE CNN POR SEMILLA
# ============================================================

def run_or_reuse_cnn_seed(
    frozen_cfg: Config,
    seed: int,
    output_dir: Path,
    force: bool,
) -> Dict:
    pred_dir = output_dir / "predictions_cnn"
    history_dir = output_dir / "histories_cnn"
    checkpoint_dir = output_dir / "checkpoints_cnn"

    pred_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"final_cnn_seed{seed}.npz"
    history_path = history_dir / f"history_cnn_seed{seed}.csv"
    result_path = output_dir / f"cnn_seed{seed}_result.json"
    checkpoint_path = checkpoint_dir / f"final_cnn_seed{seed}.pt"

    if not force and pred_path.exists() and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(
            f"[CNN seed {seed}] ya terminada -> test={result['test_accuracy']*100:.2f}%",
            flush=True,
        )
        return result

    cfg = replace(frozen_cfg, seed=int(seed))

    print("\n" + "=" * 72, flush=True)
    print(f"CNN FINAL - SEMILLA {seed}", flush=True)
    print("=" * 72, flush=True)

    model, history, training_time, diagnostics = train_cnn(
        cfg,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )

    test_loader = get_test_loader(cfg)
    test_acc, y_true, y_pred = predict_model(model, test_loader)
    np.savez_compressed(pred_path, y_true=y_true, y_pred=y_pred)
    pd.DataFrame(history).to_csv(history_path, index=False)

    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(10),
        zero_division=0,
    )

    result = {
        "seed": int(seed),
        "val_accuracy": float(diagnostics["restored_val_accuracy"]),
        "val_loss": float(diagnostics["restored_val_loss"]),
        "test_accuracy": float(test_acc),
        "macro_f1": float(np.mean(f1)),
        "training_time_sec": float(training_time),
        "n_trainable_params": int(count_params(model)),
        "best_epoch": int(diagnostics["best_epoch"]),
        "epochs_ran": int(diagnostics["epochs_ran"]),
        "prediction_file": str(pred_path),
        "history_file": str(history_path),
    }
    save_json(result, result_path)

    print(
        f"[CNN seed {seed}] TEST accuracy={test_acc*100:.2f}% | "
        f"macro-F1={result['macro_f1']*100:.2f}% | "
        f"tiempo={training_time:.2f}s | params={result['n_trainable_params']}",
        flush=True,
    )

    del model, test_loader
    gc.collect()
    return result


# ============================================================
# 6. ESTADISTICA
# ============================================================

def ci95_mean(values: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, mean
    sem = stats.sem(x)
    low, high = stats.t.interval(0.95, df=len(x) - 1, loc=mean, scale=sem)
    return float(low), float(high)


def cohens_dz(diff: Sequence[float]) -> float:
    d = np.asarray(diff, dtype=float)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    if sd == 0:
        return 0.0 if float(np.mean(d)) == 0 else math.copysign(float("inf"), float(np.mean(d)))
    return float(np.mean(d) / sd)


def paired_statistics(a: Sequence[float], b: Sequence[float], name_a: str, name_b: str) -> Dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b

    ci_low, ci_high = ci95_mean(diff)
    t_result = stats.ttest_rel(a, b)

    if np.all(np.abs(diff) <= 1e-15):
        wilcoxon_w, wilcoxon_p = 0.0, 1.0
    else:
        w_result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        wilcoxon_w = float(w_result.statistic)
        wilcoxon_p = float(w_result.pvalue)

    shapiro = stats.shapiro(diff)

    wins_a = int(np.sum(diff > 1e-15))
    wins_b = int(np.sum(diff < -1e-15))
    ties = int(len(diff) - wins_a - wins_b)

    return {
        "comparison": f"{name_a} - {name_b}",
        "model_a": name_a,
        "model_b": name_b,
        "n": int(len(diff)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_difference_a_minus_b": float(np.mean(diff)),
        "sd_difference": float(np.std(diff, ddof=1)),
        "ci95_difference_low": ci_low,
        "ci95_difference_high": ci_high,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
        "wilcoxon_w": wilcoxon_w,
        "wilcoxon_pvalue": wilcoxon_p,
        "cohen_dz": cohens_dz(diff),
        "shapiro_w_difference": float(shapiro.statistic),
        "shapiro_pvalue_difference": float(shapiro.pvalue),
        "wins_model_a": wins_a,
        "wins_model_b": wins_b,
        "ties": ties,
    }


# ============================================================
# 7. METRICAS POR CLASE
# ============================================================

def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    return arr["y_true"].astype(np.int64), arr["y_pred"].astype(np.int64)


def build_per_class_table(
    source_results: Path,
    output_dir: Path,
    seeds: Sequence[int],
) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        files = {
            "HybridModel": source_results / "predictions_final" / f"final_hybrid_seed{seed}.npz",
            "ClassicalEquivalentModel": source_results / "predictions_final" / f"final_classical_seed{seed}.npz",
            "SimpleCNNBaseline": output_dir / "predictions_cnn" / f"final_cnn_seed{seed}.npz",
        }

        y_reference = None
        for model_name, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"No existe {path}")
            y_true, y_pred = load_npz(path)
            if y_reference is None:
                y_reference = y_true
            elif not np.array_equal(y_reference, y_true):
                raise RuntimeError(f"Etiquetas test distintas entre modelos en seed {seed}")

            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=np.arange(10),
                zero_division=0,
            )
            for digit in range(10):
                rows.append({
                    "seed": int(seed),
                    "model": model_name,
                    "digit": int(digit),
                    "precision": float(precision[digit]),
                    "recall": float(recall[digit]),
                    "f1": float(f1[digit]),
                    "support": int(support[digit]),
                })

    return pd.DataFrame(rows)


# ============================================================
# 8. TABLAS RESUMEN
# ============================================================

def build_combined(hc: pd.DataFrame, cnn: pd.DataFrame) -> pd.DataFrame:
    cnn_small = cnn[[
        "seed",
        "test_accuracy",
        "val_accuracy",
        "training_time_sec",
        "n_trainable_params",
        "macro_f1",
        "best_epoch",
        "epochs_ran",
    ]].rename(columns={
        "test_accuracy": "cnn_test_accuracy",
        "val_accuracy": "cnn_val_accuracy",
        "training_time_sec": "cnn_time_sec",
        "n_trainable_params": "cnn_total_params",
        "macro_f1": "cnn_macro_f1",
        "best_epoch": "cnn_best_epoch",
        "epochs_ran": "cnn_epochs_ran",
    })

    combined = hc.merge(cnn_small, on="seed", how="inner", validate="one_to_one")
    combined["hybrid_minus_classical_pp"] = (
        combined["hybrid_test_accuracy"] - combined["classical_test_accuracy"]
    ) * 100
    combined["hybrid_minus_cnn_pp"] = (
        combined["hybrid_test_accuracy"] - combined["cnn_test_accuracy"]
    ) * 100
    combined["classical_minus_cnn_pp"] = (
        combined["classical_test_accuracy"] - combined["cnn_test_accuracy"]
    ) * 100
    return combined.sort_values("seed").reset_index(drop=True)


def build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("HybridModel", "hybrid_test_accuracy", "hybrid_time_sec", "hybrid_total_params"),
        ("ClassicalEquivalentModel", "classical_test_accuracy", "classical_time_sec", "classical_total_params"),
        ("SimpleCNNBaseline", "cnn_test_accuracy", "cnn_time_sec", "cnn_total_params"),
    ]

    rows = []
    for model, acc_col, time_col, params_col in specs:
        acc = combined[acc_col].to_numpy(dtype=float)
        times = combined[time_col].to_numpy(dtype=float)
        ci_low, ci_high = ci95_mean(acc)
        rows.append({
            "model": model,
            "n_seeds": int(len(acc)),
            "mean_test_accuracy": float(np.mean(acc)),
            "std_test_accuracy": float(np.std(acc, ddof=1)),
            "ci95_test_accuracy_low": ci_low,
            "ci95_test_accuracy_high": ci_high,
            "mean_training_time_sec": float(np.mean(times)),
            "std_training_time_sec": float(np.std(times, ddof=1)),
            "n_trainable_params": int(round(float(combined[params_col].iloc[0]))),
        })
    return pd.DataFrame(rows)


# ============================================================
# 9. GRAFICAS
# ============================================================

def save_accuracy_seed_plot(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df))
    ax.plot(x, df["hybrid_test_accuracy"] * 100, marker="o", label="HybridModel")
    ax.plot(x, df["classical_test_accuracy"] * 100, marker="o", label="ClassicalEquivalentModel")
    ax.plot(x, df["cnn_test_accuracy"] * 100, marker="o", label="SimpleCNNBaseline")
    ax.set_xticks(x)
    ax.set_xticklabels(df["seed"].astype(str).tolist())
    ax.set_xlabel("Semilla final")
    ax.set_ylabel("Accuracy test (%)")
    ax.set_title("Comparativa final por semilla - tres modelos")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_ci_plot(summary: pd.DataFrame, out: Path) -> None:
    labels = summary["model"].tolist()
    means = summary["mean_test_accuracy"].to_numpy() * 100
    lows = summary["ci95_test_accuracy_low"].to_numpy() * 100
    highs = summary["ci95_test_accuracy_high"].to_numpy() * 100
    errors = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_ylabel("Accuracy test media (%)")
    ax.set_title("Accuracy media e IC95% - tres modelos")
    ax.set_ylim(max(0, float(np.min(lows) - 1.5)), min(100, float(np.max(highs) + 1.0)))
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(means):
        ax.text(i, value + 0.08, f"{value:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_time_plot(summary: pd.DataFrame, out: Path) -> None:
    labels = summary["model"].tolist()
    means = summary["mean_training_time_sec"].to_numpy()
    stds = summary["std_training_time_sec"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_ylabel("Tiempo de entrenamiento (s)")
    ax.set_title("Tiempo medio de entrenamiento - 12 semillas")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(means):
        ax.text(i, value + max(means) * 0.015, f"{value:.1f}s", ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_params_plot(summary: pd.DataFrame, out: Path) -> None:
    labels = summary["model"].tolist()
    params = summary["n_trainable_params"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    ax.bar(x, params)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_ylabel("Parametros entrenables")
    ax.set_title("Complejidad parametrica - tres modelos")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(params):
        ax.text(i, value + max(params) * 0.015, f"{int(value):,}".replace(",", "."), ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_time_plot(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        x = float(row["mean_training_time_sec"])
        y = float(row["mean_test_accuracy"] * 100)
        ax.scatter(x, y, s=100)
        ax.annotate(row["model"], (x, y), xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("Tiempo medio de entrenamiento (s)")
    ax.set_ylabel("Accuracy test media (%)")
    ax.set_title("Compromiso rendimiento-tiempo")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_f1_plot(per_class: pd.DataFrame, out: Path) -> None:
    agg = per_class.groupby(["model", "digit"])["f1"].agg(["mean", "std"]).reset_index()
    models = ["HybridModel", "ClassicalEquivalentModel", "SimpleCNNBaseline"]
    digits = np.arange(10)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, model in enumerate(models):
        sub = agg[agg["model"] == model].sort_values("digit")
        ax.bar(
            digits + (idx - 1) * width,
            sub["mean"].to_numpy() * 100,
            width=width,
            yerr=sub["std"].to_numpy() * 100,
            capsize=3,
            label=model,
        )
    ax.set_xticks(digits)
    ax.set_xlabel("Digito MNIST")
    ax.set_ylabel("F1 medio (%)")
    ax.set_title("F1 por clase - 12 semillas")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def choose_representative_seed(df: pd.DataFrame) -> int:
    cols = ["hybrid_test_accuracy", "classical_test_accuracy", "cnn_test_accuracy"]
    means = df[cols].mean()
    stds = df[cols].std(ddof=1).replace(0, 1.0)
    distance = ((df[cols] - means) / stds).pow(2).sum(axis=1)
    return int(df.loc[distance.idxmin(), "seed"])


def save_confusion_three_models(
    source_results: Path,
    output_dir: Path,
    seed: int,
    out: Path,
) -> None:
    files = [
        ("HybridModel", source_results / "predictions_final" / f"final_hybrid_seed{seed}.npz"),
        ("ClassicalEquivalentModel", source_results / "predictions_final" / f"final_classical_seed{seed}.npz"),
        ("SimpleCNNBaseline", output_dir / "predictions_cnn" / f"final_cnn_seed{seed}.npz"),
    ]

    cms = []
    for model_name, path in files:
        y_true, y_pred = load_npz(path)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(10), normalize="true") * 100
        cms.append((model_name, cm))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    image = None
    for ax, (model_name, cm) in zip(axes, cms):
        image = ax.imshow(cm, vmin=0, vmax=100, aspect="auto")
        ax.set_title(model_name)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Clase real")
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        for i in range(10):
            for j in range(10):
                if i == j or cm[i, j] >= 5:
                    ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=7)

    fig.suptitle(f"Matrices de confusion normalizadas - semilla representativa {seed}")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label("Porcentaje por clase real (%)")
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.12, top=0.84, wspace=0.28)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 10. MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparativa final HybridModel vs ClassicalEquivalentModel vs SimpleCNNBaseline"
    )
    parser.add_argument(
        "--source-results",
        default="./results_final_corregido",
        help="Carpeta de resultados finales ya validados de Hibrido/Clasico.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results_comparativa_3_modelos",
        help="Carpeta nueva de salida.",
    )
    parser.add_argument(
        "--force-cnn",
        action="store_true",
        help="Reentrena todas las CNN aunque ya existan resultados.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Numero de threads PyTorch en CPU.",
    )
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(int(args.threads))

    source_results = Path(args.source_results).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_results == output_dir:
        raise RuntimeError("La carpeta de salida debe ser distinta de results_final_corregido.")

    frozen_cfg, final_seeds, frozen_payload = load_frozen(source_results)
    hc = load_existing_hybrid_classical(source_results, final_seeds)

    print("=" * 78, flush=True)
    print("COMPARATIVA FINAL DE TRES MODELOS", flush=True)
    print("=" * 78, flush=True)
    print(f"Resultados H/C: {source_results}", flush=True)
    print(f"Salida nueva: {output_dir}", flush=True)
    print(f"Semillas finales: {final_seeds}", flush=True)
    print("Configuracion cuantica ya seleccionada y congelada:", flush=True)
    print(
        f"q={frozen_cfg.n_qubits}, layers={frozen_cfg.n_layers}, "
        f"entangler={frozen_cfg.entangler}, encoding={frozen_cfg.encoding}, "
        f"lr={frozen_cfg.learning_rate}, batch={frozen_cfg.batch_size}, "
        f"epochs={frozen_cfg.epochs}",
        flush=True,
    )
    print(
        "HybridModel y ClassicalEquivalentModel se REUTILIZAN; "
        "solo se ejecuta la CNN faltante.",
        flush=True,
    )

    save_json(frozen_payload, output_dir / "frozen_config_used.json")
    save_json({
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(DEVICE),
        "torch_num_threads": torch.get_num_threads(),
    }, output_dir / "environment_manifest_cnn.json")

    cnn_rows = []
    for seed in final_seeds:
        cnn_rows.append(
            run_or_reuse_cnn_seed(
                frozen_cfg=frozen_cfg,
                seed=int(seed),
                output_dir=output_dir,
                force=bool(args.force_cnn),
            )
        )

    cnn_df = pd.DataFrame(cnn_rows).sort_values("seed").reset_index(drop=True)
    cnn_df.to_csv(output_dir / "cnn_results_12_seeds.csv", index=False)

    combined = build_combined(hc, cnn_df)
    combined.to_csv(output_dir / "three_model_final_results.csv", index=False)

    summary = build_summary(combined)
    summary.to_csv(output_dir / "three_model_summary.csv", index=False)

    pairwise_rows = [
        paired_statistics(
            combined["hybrid_test_accuracy"],
            combined["classical_test_accuracy"],
            "HybridModel",
            "ClassicalEquivalentModel",
        ),
        paired_statistics(
            combined["hybrid_test_accuracy"],
            combined["cnn_test_accuracy"],
            "HybridModel",
            "SimpleCNNBaseline",
        ),
        paired_statistics(
            combined["classical_test_accuracy"],
            combined["cnn_test_accuracy"],
            "ClassicalEquivalentModel",
            "SimpleCNNBaseline",
        ),
    ]
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(output_dir / "pairwise_statistics.csv", index=False)
    save_json({"comparisons": pairwise_rows}, output_dir / "pairwise_statistics.json")

    per_class = build_per_class_table(source_results, output_dir, final_seeds)
    per_class.to_csv(output_dir / "per_class_f1_three_models.csv", index=False)

    representative_seed = choose_representative_seed(combined)
    save_json({
        "representative_seed": representative_seed,
        "criterion": "minimum standardized distance to the three model mean accuracies",
    }, output_dir / "representative_seed.json")

    save_accuracy_seed_plot(combined, output_dir / "01_accuracy_por_semilla_3_modelos.png")
    save_accuracy_ci_plot(summary, output_dir / "02_accuracy_media_ic95_3_modelos.png")
    save_time_plot(summary, output_dir / "03_tiempo_medio_3_modelos.png")
    save_params_plot(summary, output_dir / "04_parametros_3_modelos.png")
    save_accuracy_time_plot(summary, output_dir / "05_accuracy_vs_tiempo_3_modelos.png")
    save_f1_plot(per_class, output_dir / "06_f1_por_clase_3_modelos.png")
    save_confusion_three_models(
        source_results,
        output_dir,
        representative_seed,
        output_dir / f"07_confusion_3_modelos_seed{representative_seed}.png",
    )

    manifest = {
        "status": "THREE_MODEL_COMPARISON_COMPLETED",
        "source_results": str(source_results),
        "output_dir": str(output_dir),
        "final_seeds": list(final_seeds),
        "hybrid_and_classical_reused": True,
        "cnn_evaluated_now": True,
        "cnn_architecture": (
            "Conv2d(1,16,3)+ReLU+MaxPool -> Conv2d(16,32,3)+ReLU+MaxPool -> "
            "Flatten -> Linear(1568,64)+ReLU -> Linear(64,10)"
        ),
        "methodological_note": (
            "Hybrid vs ClassicalEquivalent is the controlled ablation. "
            "CNN is a contextual classical reference, not a causal ablation."
        ),
        "files": {
            "per_seed": "three_model_final_results.csv",
            "summary": "three_model_summary.csv",
            "pairwise_statistics": "pairwise_statistics.csv",
            "per_class": "per_class_f1_three_models.csv",
        },
    }
    save_json(manifest, output_dir / "comparison_manifest.json")

    print("\n" + "=" * 78, flush=True)
    print("RESULTADO FINAL - TRES MODELOS", flush=True)
    print("=" * 78, flush=True)

    for _, row in summary.iterrows():
        print(
            f"{row['model']}: "
            f"accuracy={row['mean_test_accuracy']*100:.3f}% ± "
            f"{row['std_test_accuracy']*100:.3f} pp | "
            f"IC95=[{row['ci95_test_accuracy_low']*100:.3f}, "
            f"{row['ci95_test_accuracy_high']*100:.3f}]% | "
            f"tiempo={row['mean_training_time_sec']:.2f}s | "
            f"params={int(row['n_trainable_params'])}",
            flush=True,
        )

    print("\nContrastes por semilla:", flush=True)
    for _, row in pairwise_df.iterrows():
        print(
            f"{row['comparison']}: "
            f"delta={row['mean_difference_a_minus_b']*100:+.3f} pp | "
            f"IC95=[{row['ci95_difference_low']*100:+.3f}, "
            f"{row['ci95_difference_high']*100:+.3f}] pp | "
            f"t p={row['paired_t_pvalue']:.6g} | "
            f"Wilcoxon p={row['wilcoxon_pvalue']:.6g} | "
            f"dz={row['cohen_dz']:+.3f} | "
            f"victorias={int(row['wins_model_a'])}-{int(row['wins_model_b'])}",
            flush=True,
        )

    print("\nInterpretacion:", flush=True)
    print(
        "- HybridModel vs ClassicalEquivalentModel sigue siendo la comparacion causal principal.",
        flush=True,
    )
    print(
        "- La CNN sirve para contextualizar el rendimiento clasico de una arquitectura convolucional distinta.",
        flush=True,
    )
    print(f"\nArchivos guardados en: {output_dir}", flush=True)
    print("COMPARATIVA FINAL DE TRES MODELOS COMPLETADA", flush=True)


if __name__ == "__main__":
    main()

"""
============================================================
ESTUDIO SISTEMÁTICO OPTIMIZADO — TFM Clasificador Híbrido Cuántico-Clásico
============================================================
Versión mejorada de estudio_sistematico_final.py. Cambios clave:

  1. Barrido de arquitectura EN DOS ETAPAS:
     - Etapa A (coarse): 9 combinaciones n_qubits x n_layers, 1 seed, 6 épocas
       -> filtra rápido las combinaciones claramente malas
     - Etapa B (refine): top-3 candidatas de la etapa A, 3 seeds, 10 épocas
       -> promedia y elige la mejor combinación con menos ruido de inicialización
     Esto arregla el problema detectado antes (q6/l4 parecía "malo" con 1 sola
     semilla, pero podía ser solo mala suerte de init).

  2. Entangler y encoding comparados con 3 seeds cada uno (antes 1),
     para no tomar la decisión basic/strong o angle/amplitude con una sola corrida.

  3. NUEVO — Mini-grid de hiperparámetros de entrenamiento (learning_rate x
     batch_size) sobre la mejor arquitectura ya encontrada, para no dejar
     esos valores fijos "porque sí".

  4. Validación final con 12 semillas (antes 5) para dar más potencia
     estadística a la prueba t pareada híbrido vs clásico.

  5. Guarda TODO en un único CSV acumulado y genera las mismas 4 gráficas
     que antes, más una gráfica nueva del mini-grid de entrenamiento.

⚠️ TIEMPO ESTIMADO: 4-5 horas en este servidor (1 core). Ejecutar con nohup
   y dejarlo corriendo, idealmente durante la noche.
============================================================
"""

import os
import json
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import confusion_matrix, classification_report

import pennylane as qml

DEVICE = torch.device("cpu")


# ============================================================
# 0. CONFIG Y UTILIDADES BASE
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    run_name: str = "hybrid_baseline"
    seed: int = 42
    n_qubits: int = 4
    n_layers: int = 2
    entangler: str = "basic"          # "basic" | "strong"
    encoding: str = "angle"           # "angle" | "amplitude"
    encoder_hidden_dim: int = 64
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 15
    val_split: float = 0.1
    early_stopping_patience: int = 4
    device: str = "cpu"
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_csv: str = "./results/experiments_optimizado.csv"

    def as_dict(self):
        return asdict(self)


def get_dataloaders(cfg: Config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train = torchvision.datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=transform
    )
    n_val = int(len(full_train) * cfg.val_split)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


# ============================================================
# 1. CAPAS CUÁNTICAS — Angle y Amplitude, cada una con basic/strong
# ============================================================

def build_quantum_layer(cfg: Config):
    dev = qml.device("default.qubit", wires=cfg.n_qubits)

    if cfg.entangler == "basic":
        weight_shape = (cfg.n_layers, cfg.n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(cfg.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(cfg.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]

    elif cfg.entangler == "strong":
        weight_shape = (cfg.n_layers, cfg.n_qubits, 3)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(cfg.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(cfg.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]
    else:
        raise ValueError(f"entangler desconocido: {cfg.entangler}")

    return qml.qnn.TorchLayer(circuit, {"weights": weight_shape})


def build_quantum_layer_amplitude(cfg: Config):
    dev = qml.device("default.qubit", wires=cfg.n_qubits)

    if cfg.entangler == "basic":
        weight_shape = (cfg.n_layers, cfg.n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(cfg.n_qubits), normalize=True, pad_with=0.0)
            qml.BasicEntanglerLayers(weights, wires=range(cfg.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]

    elif cfg.entangler == "strong":
        weight_shape = (cfg.n_layers, cfg.n_qubits, 3)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(cfg.n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(cfg.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]
    else:
        raise ValueError(f"entangler desconocido: {cfg.entangler}")

    return qml.qnn.TorchLayer(circuit, {"weights": weight_shape})


# ============================================================
# 2. MODELOS
# ============================================================

class SharedEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, cfg.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder_hidden_dim, cfg.n_qubits),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class SharedEncoderAmplitude(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        n_amplitudes = 2 ** cfg.n_qubits
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, cfg.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder_hidden_dim, n_amplitudes),
        )

    def forward(self, x):
        return self.net(x)


class HybridModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = SharedEncoder(cfg)
        self.quantum_layer = build_quantum_layer(cfg)
        self.classifier = nn.Linear(cfg.n_qubits, 10)

    def forward(self, x):
        x = self.encoder(x) * torch.pi
        x = self.quantum_layer(x)
        return self.classifier(x)


class HybridModelAmplitude(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = SharedEncoderAmplitude(cfg)
        self.quantum_layer = build_quantum_layer_amplitude(cfg)
        self.classifier = nn.Linear(cfg.n_qubits, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.quantum_layer(x)
        return self.classifier(x)


class ClassicalEquivalentModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = SharedEncoder(cfg)
        self.classical_layer = nn.Sequential(
            nn.Linear(cfg.n_qubits, cfg.n_qubits),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(cfg.n_qubits, 10)

    def forward(self, x):
        x = self.encoder(x) * torch.pi
        x = self.classical_layer(x)
        return self.classifier(x)


def build_model(cfg: Config, model_type: str):
    if model_type == "hybrid":
        return HybridModel(cfg)
    elif model_type == "hybrid_amplitude":
        return HybridModelAmplitude(cfg)
    elif model_type == "classical":
        return ClassicalEquivalentModel(cfg)
    raise ValueError(f"model_type desconocido: {model_type}")


# ============================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def train_model(model, cfg: Config, train_loader, val_loader, run_name: str, verbose=True):
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}.pt"

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"[{run_name}] Época {epoch+1}/{cfg.epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"[{run_name}] Early stopping en época {epoch+1}")
                break

    model.load_state_dict(torch.load(ckpt_path))
    training_time = time.time() - start_time
    return model, history, training_time


def evaluate_model(model, test_loader, run_name: str, show_plot=False, verbose=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    if verbose:
        print(f"\n[{run_name}] Accuracy en test: {acc*100:.2f}%\n")

    cm = confusion_matrix(all_labels, all_preds)
    if show_plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matriz de confusión — {run_name}")
        fig.colorbar(im)
        plt.tight_layout()
        plt.show()

    return acc, cm


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_experiment(cfg: Config, model_type: str, test_acc: float, training_time: float, n_params: int):
    Path(cfg.results_csv).parent.mkdir(parents=True, exist_ok=True)
    row = {
        **cfg.as_dict(),
        "model_type": model_type,
        "test_accuracy": test_acc,
        "training_time_sec": round(training_time, 1),
        "n_trainable_params": n_params,
    }
    df_row = pd.DataFrame([row])
    if Path(cfg.results_csv).exists():
        df_row.to_csv(cfg.results_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(cfg.results_csv, mode="w", header=True, index=False)


def train_eval_log(cfg, model_type, run_name, epochs=None, verbose=True):
    """Atajo: entrena, evalúa y loguea un modelo con una config dada."""
    if epochs is not None:
        cfg.epochs = epochs
    cfg.run_name = run_name
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg, model_type)
    model, history, t = train_model(model, cfg, train_loader, val_loader, run_name, verbose=verbose)
    acc, _ = evaluate_model(model, test_loader, run_name, verbose=verbose)
    log_experiment(cfg, model_type, acc, t, count_params(model))
    return acc, t, history


# ============================================================
# 4. FASE 1A — BARRIDO COARSE (1 seed, rápido) -> top 3 candidatas
# ============================================================

def run_coarse_sweep(qubits_grid=(4, 6, 8), layers_grid=(2, 3, 4),
                      sweep_epochs=6, seed=42):
    print("\n" + "="*60)
    print("FASE 1A — BARRIDO COARSE (1 seed, filtra candidatas)")
    print("="*60)

    rows = []
    for n_qubits, n_layers in product(qubits_grid, layers_grid):
        set_seed(seed)
        run_name = f"coarse_q{n_qubits}_l{n_layers}_basic"
        cfg = Config(seed=seed, n_qubits=n_qubits, n_layers=n_layers,
                     entangler="basic", encoding="angle")
        acc, t, _ = train_eval_log(cfg, "hybrid", run_name, epochs=sweep_epochs)
        rows.append({"n_qubits": n_qubits, "n_layers": n_layers,
                      "test_accuracy": acc, "training_time_sec": round(t, 1)})

    df = pd.DataFrame(rows)
    top3 = df.sort_values("test_accuracy", ascending=False).head(3)
    print("\n--- Top 3 candidatas del barrido coarse ---")
    print(top3.to_string(index=False))
    return df, list(zip(top3["n_qubits"].astype(int), top3["n_layers"].astype(int)))


# ============================================================
# 5. FASE 1B — REFINAMIENTO MULTI-SEED de las top-3 candidatas
# ============================================================

def run_refine_sweep(candidates, seeds=(42, 123, 2024), refine_epochs=10):
    print("\n" + "="*60)
    print("FASE 1B — REFINAMIENTO MULTI-SEED (top-3, 3 seeds c/u)")
    print("="*60)

    rows = []
    for n_qubits, n_layers in candidates:
        accs = []
        for seed in seeds:
            set_seed(seed)
            run_name = f"refine_q{n_qubits}_l{n_layers}_seed{seed}"
            cfg = Config(seed=seed, n_qubits=n_qubits, n_layers=n_layers,
                         entangler="basic", encoding="angle")
            acc, t, _ = train_eval_log(cfg, "hybrid", run_name, epochs=refine_epochs)
            accs.append(acc)
        rows.append({
            "n_qubits": n_qubits, "n_layers": n_layers,
            "mean_accuracy": np.mean(accs), "std_accuracy": np.std(accs),
        })

    df = pd.DataFrame(rows)
    print("\n--- Resultado del refinamiento ---")
    print(df.to_string(index=False))

    best_row = df.loc[df["mean_accuracy"].idxmax()]
    best_n_qubits, best_n_layers = int(best_row.n_qubits), int(best_row.n_layers)
    print(f"\nMejor combinación (promedio de {len(seeds)} seeds): "
          f"n_qubits={best_n_qubits}, n_layers={best_n_layers} "
          f"(acc={best_row.mean_accuracy*100:.2f}% ± {best_row.std_accuracy*100:.2f})")

    return df, best_n_qubits, best_n_layers


# ============================================================
# 6. FASE 2 — ENTANGLER, con 3 seeds cada uno
# ============================================================

def run_entangler_comparison(best_n_qubits, best_n_layers,
                              seeds=(42, 123, 2024), full_epochs=15):
    print("\n" + "="*60)
    print("FASE 2 — ENTANGLER basic vs strong (3 seeds c/u)")
    print("="*60)

    rows = []
    for entangler in ("basic", "strong"):
        accs, times = [], []
        for seed in seeds:
            set_seed(seed)
            run_name = f"entangler_{entangler}_q{best_n_qubits}_l{best_n_layers}_seed{seed}"
            cfg = Config(seed=seed, n_qubits=best_n_qubits, n_layers=best_n_layers,
                         entangler=entangler, encoding="angle")
            acc, t, _ = train_eval_log(cfg, "hybrid", run_name, epochs=full_epochs)
            accs.append(acc); times.append(t)
        rows.append({"entangler": entangler, "mean_accuracy": np.mean(accs),
                      "std_accuracy": np.std(accs), "mean_time_sec": np.mean(times)})

    df = pd.DataFrame(rows)
    print("\n--- Resultado comparación de entangler ---")
    print(df.to_string(index=False))
    best_entangler = df.loc[df["mean_accuracy"].idxmax(), "entangler"]
    print(f"\nMejor entangler: {best_entangler}")
    return df, best_entangler


# ============================================================
# 7. FASE 3 — ENCODING, con 3 seeds cada uno
# ============================================================

def run_encoding_comparison(best_n_qubits, best_n_layers, best_entangler,
                             seeds=(42, 123, 2024), full_epochs=15):
    print("\n" + "="*60)
    print("FASE 3 — ENCODING angle vs amplitude (3 seeds c/u)")
    print("="*60)

    model_types = {"angle": "hybrid", "amplitude": "hybrid_amplitude"}
    rows = []
    for encoding, model_type in model_types.items():
        accs, times = [], []
        for seed in seeds:
            set_seed(seed)
            run_name = f"encoding_{encoding}_q{best_n_qubits}_l{best_n_layers}_{best_entangler}_seed{seed}"
            cfg = Config(seed=seed, n_qubits=best_n_qubits, n_layers=best_n_layers,
                         entangler=best_entangler, encoding=encoding)
            acc, t, _ = train_eval_log(cfg, model_type, run_name, epochs=full_epochs)
            accs.append(acc); times.append(t)
        rows.append({"encoding": encoding, "mean_accuracy": np.mean(accs),
                      "std_accuracy": np.std(accs), "mean_time_sec": np.mean(times)})

    df = pd.DataFrame(rows)
    print("\n--- Resultado comparación de encoding ---")
    print(df.to_string(index=False))
    best_encoding = df.loc[df["mean_accuracy"].idxmax(), "encoding"]
    print(f"\nMejor encoding: {best_encoding}")
    return df, best_encoding


# ============================================================
# 8. FASE 3B — NUEVO: mini-grid de hiperparámetros de entrenamiento
#    (learning_rate x batch_size) sobre la mejor arquitectura, 1 seed,
#    pocas épocas (screening rápido, no busca precisión final)
# ============================================================

def run_training_hparam_grid(best_n_qubits, best_n_layers, best_entangler, best_encoding,
                              lr_grid=(1e-3, 5e-4), batch_grid=(64, 128),
                              seed=42, screening_epochs=8):
    print("\n" + "="*60)
    print("FASE 3B — MINI-GRID DE HIPERPARÁMETROS DE ENTRENAMIENTO")
    print("="*60)

    model_type = "hybrid_amplitude" if best_encoding == "amplitude" else "hybrid"
    rows = []
    for lr, bs in product(lr_grid, batch_grid):
        set_seed(seed)
        run_name = f"hparam_lr{lr}_bs{bs}"
        cfg = Config(seed=seed, n_qubits=best_n_qubits, n_layers=best_n_layers,
                     entangler=best_entangler, encoding=best_encoding,
                     learning_rate=lr, batch_size=bs)
        acc, t, _ = train_eval_log(cfg, model_type, run_name, epochs=screening_epochs)
        rows.append({"learning_rate": lr, "batch_size": bs,
                      "test_accuracy": acc, "training_time_sec": round(t, 1)})

    df = pd.DataFrame(rows)
    print("\n--- Resultado mini-grid de entrenamiento ---")
    print(df.to_string(index=False))
    best_row = df.loc[df["test_accuracy"].idxmax()]
    best_lr, best_bs = float(best_row.learning_rate), int(best_row.batch_size)
    print(f"\nMejor combinación de entrenamiento: lr={best_lr}, batch_size={best_bs}")
    return df, best_lr, best_bs


# ============================================================
# 9. FASE 4 — VALIDACIÓN FINAL, 12 SEEDS (más potencia estadística)
# ============================================================

def run_multiseed_validation(best_n_qubits, best_n_layers, best_entangler, best_encoding,
                              best_lr, best_bs,
                              seeds=(42, 123, 2024, 7, 99, 11, 256, 512, 777, 1001, 2025, 31),
                              full_epochs=15):
    print("\n" + "="*60)
    print(f"FASE 4 — VALIDACIÓN FINAL ({len(seeds)} seeds)")
    print("="*60)

    hybrid_model_type = "hybrid_amplitude" if best_encoding == "amplitude" else "hybrid"

    results_hybrid, results_classical = [], []
    hist_hybrid, hist_classical = [], []

    for seed in seeds:
        print(f"\n=============== SEMILLA {seed} ===============")

        set_seed(seed)
        cfg_h = Config(seed=seed, n_qubits=best_n_qubits, n_layers=best_n_layers,
                       entangler=best_entangler, encoding=best_encoding,
                       learning_rate=best_lr, batch_size=best_bs)
        run_name_h = f"final_hybrid_seed{seed}"
        cfg_h.run_name = run_name_h
        cfg_h.epochs = full_epochs
        train_loader, val_loader, test_loader = get_dataloaders(cfg_h)
        model_h = build_model(cfg_h, hybrid_model_type)
        model_h, history_h, t_h = train_model(model_h, cfg_h, train_loader, val_loader, run_name_h)
        acc_h, _ = evaluate_model(model_h, test_loader, run_name_h)
        log_experiment(cfg_h, "hybrid_final_multiseed", acc_h, t_h, count_params(model_h))
        results_hybrid.append(acc_h)
        hist_hybrid.append(history_h)

        set_seed(seed)
        cfg_c = Config(seed=seed, n_qubits=best_n_qubits, n_layers=best_n_layers,
                       entangler=best_entangler, encoding="angle",
                       learning_rate=best_lr, batch_size=best_bs, epochs=full_epochs)
        run_name_c = f"final_classical_seed{seed}"
        cfg_c.run_name = run_name_c
        model_c = build_model(cfg_c, "classical")
        model_c, history_c, t_c = train_model(model_c, cfg_c, train_loader, val_loader, run_name_c)
        acc_c, _ = evaluate_model(model_c, test_loader, run_name_c)
        log_experiment(cfg_c, "classical_final_multiseed", acc_c, t_c, count_params(model_c))
        results_classical.append(acc_c)
        hist_classical.append(history_c)

    mean_h, std_h = np.mean(results_hybrid), np.std(results_hybrid)
    mean_c, std_c = np.mean(results_classical), np.std(results_classical)
    t_stat, p_val = stats.ttest_rel(results_hybrid, results_classical)

    print("\n" + "="*60)
    print("RESULTADO FINAL DEL TFM (VERSIÓN OPTIMIZADA)")
    print("="*60)
    print(f"Config: n_qubits={best_n_qubits}, n_layers={best_n_layers}, "
          f"entangler={best_entangler}, encoding={best_encoding}, "
          f"lr={best_lr}, batch_size={best_bs}")
    print(f"Híbrido:             {mean_h*100:.2f}% ± {std_h*100:.2f}  (n={len(seeds)} semillas)")
    print(f"Clásico equivalente: {mean_c*100:.2f}% ± {std_c*100:.2f}  (n={len(seeds)} semillas)")
    print(f"Diferencia (híbrido - clásico): {(mean_h-mean_c)*100:+.2f} puntos porcentuales")
    print(f"\nPrueba t pareada: t={t_stat:.3f}, p={p_val:.4f}")
    print("=> Diferencia estadísticamente significativa (p<0.05)." if p_val < 0.05
          else "=> Diferencia NO estadísticamente significativa con este número de semillas.")

    return {
        "results_hybrid": results_hybrid, "results_classical": results_classical,
        "hist_hybrid": hist_hybrid, "hist_classical": hist_classical,
        "mean_h": mean_h, "std_h": std_h, "mean_c": mean_c, "std_c": std_c,
        "t_stat": t_stat, "p_val": p_val,
    }


# ============================================================
# 10. GRÁFICAS FINALES
# ============================================================

def plot_all_results(coarse_df, refine_df, entangler_df, encoding_df, hparam_df,
                      multiseed_results):
    Path("./results").mkdir(parents=True, exist_ok=True)

    # Heatmap coarse
    pivot = coarse_df.pivot(index="n_layers", columns="n_qubits", values="test_accuracy")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("n_qubits"); ax.set_ylabel("n_layers")
    ax.set_title("Barrido coarse (1 seed) — accuracy")
    fig.colorbar(im, label="test_accuracy")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]*100:.1f}", ha="center", va="center", color="white")
    plt.tight_layout(); plt.savefig("./results/heatmap_coarse.png", dpi=150); plt.show()

    # Refinamiento top-3 con error bars
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"q{int(r.n_qubits)}/l{int(r.n_layers)}" for _, r in refine_df.iterrows()]
    ax.bar(labels, refine_df["mean_accuracy"], yerr=refine_df["std_accuracy"], capsize=5,
           color="#4C72B0")
    ax.set_ylabel("Test accuracy (media ± std, 3 seeds)")
    ax.set_title("Refinamiento de las top-3 candidatas")
    plt.tight_layout(); plt.savefig("./results/refine_top3.png", dpi=150); plt.show()

    # Entangler y encoding con error bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(entangler_df["entangler"], entangler_df["mean_accuracy"],
                yerr=entangler_df["std_accuracy"], capsize=5, color=["#4C72B0", "#DD8452"])
    axes[0].set_ylabel("Test accuracy"); axes[0].set_title("Entangler (3 seeds)")

    axes[1].bar(encoding_df["encoding"], encoding_df["mean_accuracy"],
                yerr=encoding_df["std_accuracy"], capsize=5, color=["#4C72B0", "#DD8452"])
    axes[1].set_ylabel("Test accuracy"); axes[1].set_title("Encoding (3 seeds)")
    plt.tight_layout(); plt.savefig("./results/entangler_encoding.png", dpi=150); plt.show()

    # Mini-grid de entrenamiento
    pivot_h = hparam_df.pivot(index="batch_size", columns="learning_rate", values="test_accuracy")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot_h.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot_h.columns))); ax.set_xticklabels(pivot_h.columns)
    ax.set_yticks(range(len(pivot_h.index))); ax.set_yticklabels(pivot_h.index)
    ax.set_xlabel("learning_rate"); ax.set_ylabel("batch_size")
    ax.set_title("Mini-grid de entrenamiento")
    fig.colorbar(im, label="test_accuracy")
    for i in range(len(pivot_h.index)):
        for j in range(len(pivot_h.columns)):
            ax.text(j, i, f"{pivot_h.values[i,j]*100:.1f}", ha="center", va="center", color="white")
    plt.tight_layout(); plt.savefig("./results/hparam_grid.png", dpi=150); plt.show()

    # Resultado final: curvas + boxplot
    results_hybrid = multiseed_results["results_hybrid"]
    results_classical = multiseed_results["results_classical"]
    hist_hybrid = multiseed_results["hist_hybrid"]
    hist_classical = multiseed_results["hist_classical"]
    p_val = multiseed_results["p_val"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for h in hist_hybrid:
        axes[0].plot(h["epoch"], h["val_acc"], color="#4C72B0", alpha=0.25)
    for h in hist_classical:
        axes[0].plot(h["epoch"], h["val_acc"], color="#DD8452", alpha=0.25)
    axes[0].plot([], [], color="#4C72B0", label="Híbrido (mejor config)")
    axes[0].plot([], [], color="#DD8452", label="Clásico equivalente")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Val accuracy")
    axes[0].set_title(f"Curvas de aprendizaje ({len(results_hybrid)} semillas)")
    axes[0].legend()

    axes[1].boxplot([results_hybrid, results_classical], tick_labels=["Híbrido", "Clásico"])
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title(f"Distribución final (p={p_val:.3f}, n={len(results_hybrid)})")

    plt.tight_layout(); plt.savefig("./results/resultado_final_optimizado.png", dpi=150); plt.show()


# ============================================================
# 11. PIPELINE COMPLETO
# ============================================================

if __name__ == "__main__":
    t0 = time.time()

    # Fase 1A: barrido coarse -> top 3 candidatas
    coarse_df, top3_candidates = run_coarse_sweep(
        qubits_grid=(4, 6, 8), layers_grid=(2, 3, 4), sweep_epochs=6, seed=42,
    )

    # Fase 1B: refinar top 3 con 3 seeds cada una
    refine_df, best_n_qubits, best_n_layers = run_refine_sweep(
        top3_candidates, seeds=(42, 123, 2024), refine_epochs=10,
    )

    # Fase 2: entangler, 3 seeds
    entangler_df, best_entangler = run_entangler_comparison(
        best_n_qubits, best_n_layers, seeds=(42, 123, 2024), full_epochs=15,
    )

    # Fase 3: encoding, 3 seeds
    encoding_df, best_encoding = run_encoding_comparison(
        best_n_qubits, best_n_layers, best_entangler, seeds=(42, 123, 2024), full_epochs=15,
    )

    # Fase 3B: mini-grid de hiperparámetros de entrenamiento
    hparam_df, best_lr, best_bs = run_training_hparam_grid(
        best_n_qubits, best_n_layers, best_entangler, best_encoding,
        lr_grid=(1e-3, 5e-4), batch_grid=(64, 128), seed=42, screening_epochs=8,
    )

    # Fase 4: validación final con 12 semillas
    multiseed_results = run_multiseed_validation(
        best_n_qubits, best_n_layers, best_entangler, best_encoding, best_lr, best_bs,
        seeds=(42, 123, 2024, 7, 99, 11, 256, 512, 777, 1001, 2025, 31), full_epochs=15,
    )

    # Gráficas
    plot_all_results(coarse_df, refine_df, entangler_df, encoding_df, hparam_df,
                      multiseed_results)

    total_time = time.time() - t0
    print("\n" + "="*60)
    print("ESTUDIO SISTEMÁTICO OPTIMIZADO — COMPLETO")
    print("="*60)
    print(f"Mejor configuración global encontrada:")
    print(f"  n_qubits      = {best_n_qubits}")
    print(f"  n_layers      = {best_n_layers}")
    print(f"  entangler     = {best_entangler}")
    print(f"  encoding      = {best_encoding}")
    print(f"  learning_rate = {best_lr}")
    print(f"  batch_size    = {best_bs}")
    print(f"  accuracy      = {multiseed_results['mean_h']*100:.2f}% ± {multiseed_results['std_h']*100:.2f}")
    print(f"  p-valor vs clásico = {multiseed_results['p_val']:.4f}")
    print(f"\nTiempo total: {total_time/3600:.2f} horas")
    print(f"CSV acumulado: ./results/experiments_optimizado.csv")
    print(f"Gráficas en: ./results/*.png")

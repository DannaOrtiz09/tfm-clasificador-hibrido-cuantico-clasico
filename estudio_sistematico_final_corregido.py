"""
ESTUDIO SISTEMATICO FINAL CORREGIDO - TFM
Clasificador Hibrido Cuantico-Clasico sobre MNIST

Objetivo metodologico de esta version
--------------------------------------
1. El conjunto TEST no participa en ninguna decision de seleccion.
   - Arquitectura, entangler, encoding, learning rate y batch size se eligen SOLO con VALIDACION.
   - El TEST oficial de MNIST se abre unicamente al final, con la configuracion congelada.

2. El encoder clasico es IDENTICO para AngleEmbedding y AmplitudeEmbedding.
   - Siempre: 784 -> encoder_hidden_dim -> n_qubits.
   - AmplitudeEmbedding recibe esos mismos n_qubits valores y usa zero-padding hasta 2**n_qubits.
   - Asi, al comparar encodings no se cambia simultaneamente el tamano del encoder.

3. El baseline clasico comparte exactamente:
   - el mismo tipo de encoder,
   - el mismo clasificador,
   - la misma inicializacion del encoder y clasificador para cada semilla,
   - el mismo split train/validation,
   - el mismo orden de minibatches,
   - los mismos hiperparametros de entrenamiento.
   Solo cambia el bloque intermedio: circuito cuantico vs Linear + Tanh.

4. Se separan las fuentes de aleatoriedad:
   - split_seed: fija SIEMPRE el mismo train/validation split.
   - seed: varia la inicializacion del modelo y el shuffle del DataLoader.

5. Las semillas de seleccion y las semillas finales son DISJUNTAS.
   Esto evita reutilizar en la prueba final las inicializaciones que ayudaron a escoger el modelo.

6. AmplitudeEmbedding incluye un control explicito del flujo de gradiente.
   Si la version instalada de PennyLane no permite que el gradiente llegue al encoder,
   el programa se detiene en lugar de producir un resultado metodologicamente invalido.

7. La estadistica final incluye:
   - media y desviacion estandar muestral,
   - t de Student pareada,
   - IC 95% de la diferencia pareada,
   - Cohen dz,
   - Wilcoxon como contraste no parametrico de apoyo,
   - numero de victorias/empates.

IMPORTANTE
----------
Conserva intactos los notebooks antiguos. Ejecuta esta version como experimento final nuevo.
No mezcles su CSV con experiments_optimizado.csv.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

from sklearn.metrics import classification_report, confusion_matrix
import sklearn

import pennylane as qml


# ============================================================
# 0. CONFIGURACION GLOBAL
# ============================================================

DEVICE = torch.device("cpu")

# Para maximizar reproducibilidad en CPU. Puedes cambiar TFM_TORCH_THREADS
# desde el entorno si el administrador del HPC te asigna mas cores.
try:
    torch.set_num_threads(int(os.environ.get("TFM_TORCH_THREADS", "1")))
except Exception:
    pass

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    # Compatibilidad con versiones antiguas de PyTorch.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
except Exception:
    pass


@dataclass
class Config:
    run_name: str = "tfm_final"

    # seed = inicializacion del modelo + shuffle del DataLoader.
    seed: int = 42

    # El split train/validation queda FIJO en todo el estudio.
    split_seed: int = 2026

    n_qubits: int = 4
    n_layers: int = 2
    entangler: str = "basic"          # "basic" | "strong"
    encoding: str = "angle"           # "angle" | "amplitude"

    encoder_hidden_dim: int = 64
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 15
    val_split: float = 0.10
    early_stopping_patience: int = 4

    device: str = "cpu"
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints_final_corregido"
    results_dir: str = "./results_final_corregido"
    results_csv: str = "./results_final_corregido/experiments_final_corregido.csv"

    # Control de calidad: obliga a verificar gradiente del encoder
    # en la primera iteracion de cada modelo hibrido.
    enforce_encoder_gradient: bool = True

    def as_dict(self) -> Dict:
        return asdict(self)


# Semillas usadas SOLO para seleccionar configuraciones con validation.
SELECTION_SEEDS: Tuple[int, ...] = (42, 123, 2024)

# Semillas usadas SOLO para la evaluacion final sobre test.
# Son deliberadamente distintas de SELECTION_SEEDS.
FINAL_SEEDS: Tuple[int, ...] = (
    7, 11, 17, 31, 73, 99, 256, 314, 512, 777, 1001, 2025
)


# ============================================================
# 1. REPRODUCIBILIDAD Y TRAZABILIDAD
# ============================================================

def set_seed(seed: int) -> None:
    """Fija las fuentes de aleatoriedad principales."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_environment_manifest(cfg: Config) -> Path:
    """Guarda versiones y plataforma para poder reproducir el experimento."""
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pennylane": qml.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
        "scikit_learn": sklearn.__version__,
        "device": str(DEVICE),
        "torch_num_threads": torch.get_num_threads(),
        "selection_seeds": list(SELECTION_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "base_config": cfg.as_dict(),
    }

    path = out_dir / "environment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# 2. DATOS
#    TEST SE MANTIENE CERRADO DURANTE TODA LA SELECCION
# ============================================================

def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_train_val_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Devuelve SOLO train y validation.

    El split depende de cfg.split_seed, que permanece constante en todo el estudio.
    El shuffle de train depende de cfg.seed, que cambia entre replicas.
    """
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

    # Generador propio para el orden de minibatches.
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
    """
    Abre el conjunto TEST oficial de MNIST.
    Esta funcion SOLO se llama en la fase final, despues de congelar la configuracion.
    """
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


def describe_data_split(cfg: Config) -> Dict:
    """
    Documenta train/validation SIN cargar el conjunto test.
    El test permanece literalmente sin abrir hasta la fase final.
    """
    full_train = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=mnist_transform(),
    )
    n_val = int(len(full_train) * cfg.val_split)
    n_train = len(full_train) - n_val
    return {
        "train": n_train,
        "validation": n_val,
        "test": "NOT_LOADED_UNTIL_FINAL_PHASE",
        "split_seed": cfg.split_seed,
    }


# ============================================================
# 3. CAPAS CUANTICAS
# ============================================================

def build_quantum_layer(cfg: Config) -> qml.qnn.TorchLayer:
    """
    Construye el bloque cuantico para AngleEmbedding o AmplitudeEmbedding.

    CONTROL DE EQUIDAD ENTRE ENCODINGS:
    Ambos reciben exactamente cfg.n_qubits caracteristicas procedentes del MISMO encoder.

    - AngleEmbedding: usa esas n_qubits caracteristicas como angulos RX.
    - AmplitudeEmbedding: usa esas mismas n_qubits caracteristicas y las rellena con ceros
      hasta 2**n_qubits amplitudes mediante pad_with=0.0.

    De esta forma NO se sustituye el encoder 64->n_qubits por un encoder 64->2**n_qubits.
    """
    dev = qml.device("default.qubit", wires=cfg.n_qubits)

    if cfg.entangler == "basic":
        weight_shape = (cfg.n_layers, cfg.n_qubits)
    elif cfg.entangler == "strong":
        weight_shape = (cfg.n_layers, cfg.n_qubits, 3)
    else:
        raise ValueError(f"Entangler desconocido: {cfg.entangler}")

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        if cfg.encoding == "angle":
            # El notebook original no indicaba rotation; aqui se fija explicitamente X.
            qml.AngleEmbedding(
                inputs * torch.pi,
                wires=range(cfg.n_qubits),
                rotation="X",
            )
        elif cfg.encoding == "amplitude":
            qml.AmplitudeEmbedding(
                inputs,
                wires=range(cfg.n_qubits),
                pad_with=0.0,
                normalize=True,
            )
        else:
            raise ValueError(f"Encoding desconocido: {cfg.encoding}")

        if cfg.entangler == "basic":
            qml.BasicEntanglerLayers(weights, wires=range(cfg.n_qubits))
        else:
            qml.StronglyEntanglingLayers(weights, wires=range(cfg.n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": weight_shape})


# ============================================================
# 4. MODELOS
# ============================================================

class SharedEncoder(nn.Module):
    """Encoder comun a TODAS las ramas: 784 -> 64 -> n_qubits."""

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


class HybridModel(nn.Module):
    """Encoder comun -> bloque cuantico -> clasificador comun."""

    def __init__(self, cfg: Config):
        super().__init__()

        # Semillas de componente: encoder y clasificador son reproducibles
        # e identicos al baseline clasico de la misma replica.
        torch.manual_seed(cfg.seed)
        self.encoder = SharedEncoder(cfg)

        torch.manual_seed(cfg.seed + 10_000)
        self.quantum_layer = build_quantum_layer(cfg)

        torch.manual_seed(cfg.seed + 20_000)
        self.classifier = nn.Linear(cfg.n_qubits, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.quantum_layer(x)
        return self.classifier(x)


class ClassicalEquivalentModel(nn.Module):
    """
    Ablation baseline:
    Encoder comun -> Linear(n_qubits,n_qubits)+Tanh -> clasificador comun.

    El punto de sustitucion coincide exactamente con el bloque cuantico.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        torch.manual_seed(cfg.seed)
        self.encoder = SharedEncoder(cfg)

        torch.manual_seed(cfg.seed + 30_000)
        self.classical_layer = nn.Sequential(
            nn.Linear(cfg.n_qubits, cfg.n_qubits),
            nn.Tanh(),
        )

        torch.manual_seed(cfg.seed + 20_000)
        self.classifier = nn.Linear(cfg.n_qubits, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classical_layer(x)
        return self.classifier(x)


class SimpleCNNBaseline(nn.Module):
    """Referencia clasica convencional opcional; no forma parte de la ablacion causal."""

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


def build_model(cfg: Config, model_type: str) -> nn.Module:
    if model_type == "hybrid":
        return HybridModel(cfg)
    if model_type == "classical":
        return ClassicalEquivalentModel(cfg)
    if model_type == "cnn":
        torch.manual_seed(cfg.seed)
        return SimpleCNNBaseline()
    raise ValueError(f"model_type desconocido: {model_type}")


def assert_shared_initialization_equal(hybrid: HybridModel, classical: ClassicalEquivalentModel) -> None:
    """Comprueba que encoder y clasificador parten EXACTAMENTE de los mismos pesos."""
    for (name_h, p_h), (name_c, p_c) in zip(
        hybrid.encoder.named_parameters(), classical.encoder.named_parameters()
    ):
        if name_h != name_c or not torch.equal(p_h.detach().cpu(), p_c.detach().cpu()):
            raise RuntimeError("El encoder hibrido y el clasico no tienen la misma inicializacion.")

    for (name_h, p_h), (name_c, p_c) in zip(
        hybrid.classifier.named_parameters(), classical.classifier.named_parameters()
    ):
        if name_h != name_c or not torch.equal(p_h.detach().cpu(), p_c.detach().cpu()):
            raise RuntimeError("El clasificador hibrido y el clasico no tienen la misma inicializacion.")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params_by_component(model: nn.Module) -> Dict[str, int]:
    out = {
        "encoder": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        if hasattr(model, "encoder") else 0,
        "middle": 0,
        "classifier": sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        if hasattr(model, "classifier") else 0,
    }
    if hasattr(model, "quantum_layer"):
        out["middle"] = sum(p.numel() for p in model.quantum_layer.parameters() if p.requires_grad)
    elif hasattr(model, "classical_layer"):
        out["middle"] = sum(p.numel() for p in model.classical_layer.parameters() if p.requires_grad)
    out["total"] = sum(out.values())
    return out


def parameter_comparison(cfg: Config) -> pd.DataFrame:
    set_seed(cfg.seed)
    h = HybridModel(cfg)
    c = ClassicalEquivalentModel(cfg)
    assert_shared_initialization_equal(h, c)
    h_count = count_params_by_component(h)
    c_count = count_params_by_component(c)
    df = pd.DataFrame([
        {"model": "hybrid", **h_count},
        {"model": "classical_equivalent", **c_count},
    ])
    del h, c
    gc.collect()
    return df


# ============================================================
# 5. ENTRENAMIENTO Y EVALUACION
# ============================================================

def _load_state_dict_compat(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def encoder_gradient_norm(model: nn.Module) -> Optional[float]:
    if not hasattr(model, "encoder"):
        return None

    sq_sum = 0.0
    found = False
    for p in model.encoder.parameters():
        if p.grad is not None:
            found = True
            g = p.grad.detach()
            sq_sum += float(torch.sum(g * g).cpu())

    if not found:
        return None
    return math.sqrt(sq_sum)


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


def train_model(
    model: nn.Module,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_name: str,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict, float, Dict]:
    """
    Entrena con early stopping basado EXCLUSIVAMENTE en validation loss.
    Devuelve el checkpoint de menor validation loss.
    """
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{run_name}.pt"

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    first_encoder_grad_norm = None
    gradient_checked = False

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # CONTROL CRITICO PARA EL PIPELINE HIBRIDO.
            if (
                not gradient_checked
                and hasattr(model, "quantum_layer")
                and batch_idx == 0
                and epoch == 0
            ):
                first_encoder_grad_norm = encoder_gradient_norm(model)
                gradient_checked = True

                if verbose:
                    print(
                        f"[{run_name}] Norma gradiente encoder (primer batch): "
                        f"{first_encoder_grad_norm}"
                    )

                if cfg.enforce_encoder_gradient:
                    if first_encoder_grad_norm is None:
                        raise RuntimeError(
                            "No existe gradiente en el encoder. El entrenamiento end-to-end "
                            "no es valido con esta configuracion/version de PennyLane. "
                            "No continues con el experimento final."
                        )
                    if not np.isfinite(first_encoder_grad_norm) or first_encoder_grad_norm <= 1e-12:
                        raise RuntimeError(
                            "El gradiente del encoder es nulo/no finito. El entrenamiento "
                            "end-to-end no es valido. No continues con el experimento final."
                        )

            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_loss_accuracy(model, val_loader)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"[{run_name}] Epoca {epoch + 1}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                if verbose:
                    print(f"[{run_name}] Early stopping en epoca {epoch + 1}")
                break

    training_time = time.time() - start_time

    model.load_state_dict(_load_state_dict_compat(ckpt_path))
    restored_val_loss, restored_val_acc = evaluate_loss_accuracy(model, val_loader)

    diagnostics = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "restored_val_loss": float(restored_val_loss),
        "restored_val_accuracy": float(restored_val_acc),
        "first_encoder_grad_norm": (
            None if first_encoder_grad_norm is None else float(first_encoder_grad_norm)
        ),
        "epochs_ran": len(history["epoch"]),
    }

    return model, history, training_time, diagnostics


def predict_model(model: nn.Module, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred


# ============================================================
# 6. REGISTRO DE EXPERIMENTOS SIN DUPLICADOS
# ============================================================

def upsert_experiment(record: Dict, csv_path: str) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([record])

    if path.exists():
        old = pd.read_csv(path)
        if "run_name" in old.columns and "run_name" in new_df.columns:
            old = old[old["run_name"] != record["run_name"]]
        combined = pd.concat([old, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    combined.to_csv(path, index=False)


def make_record(
    cfg: Config,
    phase: str,
    model_type: str,
    training_time: float,
    diagnostics: Dict,
    model: nn.Module,
    test_accuracy: Optional[float] = None,
) -> Dict:
    counts = count_params_by_component(model)
    return {
        **cfg.as_dict(),
        "phase": phase,
        "model_type": model_type,
        "selection_metric_source": "validation" if phase != "final_test" else "frozen_before_test",
        "val_accuracy": diagnostics.get("restored_val_accuracy"),
        "val_loss": diagnostics.get("restored_val_loss"),
        "test_accuracy": test_accuracy,
        "best_epoch": diagnostics.get("best_epoch"),
        "epochs_ran": diagnostics.get("epochs_ran"),
        "first_encoder_grad_norm": diagnostics.get("first_encoder_grad_norm"),
        "training_time_sec": round(training_time, 2),
        "encoder_params": counts.get("encoder"),
        "middle_params": counts.get("middle"),
        "classifier_params": counts.get("classifier"),
        "n_trainable_params": counts.get("total", count_params(model)),
    }


# ============================================================
# 7. EJECUCION DE UNA CORRIDA DE SELECCION
#    NUNCA TOCA TEST
# ============================================================

def run_selection_experiment(
    cfg: Config,
    phase: str,
    run_name: str,
    model_type: str = "hybrid",
    epochs: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    if epochs is not None:
        cfg = replace(cfg, epochs=epochs)
    cfg = replace(cfg, run_name=run_name)

    set_seed(cfg.seed)
    train_loader, val_loader = get_train_val_loaders(cfg)
    model = build_model(cfg, model_type)

    model, history, training_time, diagnostics = train_model(
        model, cfg, train_loader, val_loader, run_name, verbose=verbose
    )

    record = make_record(
        cfg=cfg,
        phase=phase,
        model_type=model_type,
        training_time=training_time,
        diagnostics=diagnostics,
        model=model,
        test_accuracy=None,  # CRITICO: test cerrado.
    )
    upsert_experiment(record, cfg.results_csv)

    result = {
        "val_accuracy": diagnostics["restored_val_accuracy"],
        "val_loss": diagnostics["restored_val_loss"],
        "training_time_sec": training_time,
        "history": history,
        "diagnostics": diagnostics,
        "record": record,
    }

    del model, train_loader, val_loader
    gc.collect()
    return result


# ============================================================
# 8. FASE 1A - BARRIDO COARSE
# ============================================================

def run_coarse_sweep(
    base_cfg: Config,
    qubits_grid: Sequence[int] = (4, 6, 8),
    layers_grid: Sequence[int] = (2, 3, 4),
    sweep_epochs: int = 6,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    print("\n" + "=" * 70)
    print("FASE 1A - BARRIDO COARSE: SOLO VALIDATION")
    print("=" * 70)

    rows = []
    for n_qubits, n_layers in product(qubits_grid, layers_grid):
        cfg = replace(
            base_cfg,
            seed=seed,
            n_qubits=n_qubits,
            n_layers=n_layers,
            entangler="basic",
            encoding="angle",
            epochs=sweep_epochs,
        )
        run_name = f"coarse_q{n_qubits}_l{n_layers}_basic_angle_seed{seed}"
        out = run_selection_experiment(
            cfg, "coarse", run_name, "hybrid", verbose=True
        )
        rows.append({
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "val_accuracy": out["val_accuracy"],
            "val_loss": out["val_loss"],
            "training_time_sec": out["training_time_sec"],
        })

    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    top3 = df.head(3)

    out_path = Path(base_cfg.results_dir) / "selection_coarse.csv"
    df.to_csv(out_path, index=False)

    print("\nTop 3 segun VALIDATION accuracy:")
    print(top3.to_string(index=False))

    candidates = list(zip(top3["n_qubits"].astype(int), top3["n_layers"].astype(int)))
    return df, candidates


# ============================================================
# 9. FASE 1B - REFINAMIENTO MULTI-SEED
# ============================================================

def run_refine_sweep(
    base_cfg: Config,
    candidates: Sequence[Tuple[int, int]],
    seeds: Sequence[int] = SELECTION_SEEDS,
    refine_epochs: int = 10,
) -> Tuple[pd.DataFrame, int, int]:
    print("\n" + "=" * 70)
    print("FASE 1B - REFINAMIENTO MULTI-SEED: SOLO VALIDATION")
    print("=" * 70)

    rows = []
    for n_qubits, n_layers in candidates:
        vals = []
        times = []
        for seed in seeds:
            cfg = replace(
                base_cfg,
                seed=seed,
                n_qubits=n_qubits,
                n_layers=n_layers,
                entangler="basic",
                encoding="angle",
                epochs=refine_epochs,
            )
            run_name = f"refine_q{n_qubits}_l{n_layers}_basic_angle_seed{seed}"
            out = run_selection_experiment(cfg, "refine", run_name, "hybrid")
            vals.append(out["val_accuracy"])
            times.append(out["training_time_sec"])

        rows.append({
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "mean_val_accuracy": float(np.mean(vals)),
            "std_val_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "mean_time_sec": float(np.mean(times)),
            "n_seeds": len(vals),
        })

    df = pd.DataFrame(rows).sort_values("mean_val_accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(Path(base_cfg.results_dir) / "selection_refine.csv", index=False)

    best = df.iloc[0]
    best_n_qubits = int(best["n_qubits"])
    best_n_layers = int(best["n_layers"])

    print("\nResultado refinamiento:")
    print(df.to_string(index=False))
    print(
        f"\nMejor arquitectura por VALIDATION: q={best_n_qubits}, l={best_n_layers}, "
        f"acc={best['mean_val_accuracy']*100:.2f}%"
    )

    return df, best_n_qubits, best_n_layers


# ============================================================
# 10. FASE 2 - ENTANGLER
# ============================================================

def run_entangler_comparison(
    base_cfg: Config,
    best_n_qubits: int,
    best_n_layers: int,
    seeds: Sequence[int] = SELECTION_SEEDS,
    full_epochs: int = 15,
) -> Tuple[pd.DataFrame, str]:
    print("\n" + "=" * 70)
    print("FASE 2 - BASIC vs STRONG: SOLO VALIDATION")
    print("=" * 70)

    rows = []
    for entangler in ("basic", "strong"):
        vals = []
        times = []
        for seed in seeds:
            cfg = replace(
                base_cfg,
                seed=seed,
                n_qubits=best_n_qubits,
                n_layers=best_n_layers,
                entangler=entangler,
                encoding="angle",
                epochs=full_epochs,
            )
            run_name = (
                f"entangler_{entangler}_q{best_n_qubits}_l{best_n_layers}_"
                f"angle_seed{seed}"
            )
            out = run_selection_experiment(cfg, "entangler", run_name, "hybrid")
            vals.append(out["val_accuracy"])
            times.append(out["training_time_sec"])

        rows.append({
            "entangler": entangler,
            "mean_val_accuracy": float(np.mean(vals)),
            "std_val_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "mean_time_sec": float(np.mean(times)),
            "n_seeds": len(vals),
        })

    df = pd.DataFrame(rows).sort_values("mean_val_accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(Path(base_cfg.results_dir) / "selection_entangler.csv", index=False)
    best_entangler = str(df.iloc[0]["entangler"])

    print("\nResultado entanglers:")
    print(df.to_string(index=False))
    print(f"\nMejor entangler por VALIDATION: {best_entangler}")
    return df, best_entangler


# ============================================================
# 11. FASE 3 - ENCODING JUSTO
# ============================================================

def run_encoding_comparison(
    base_cfg: Config,
    best_n_qubits: int,
    best_n_layers: int,
    best_entangler: str,
    seeds: Sequence[int] = SELECTION_SEEDS,
    full_epochs: int = 15,
) -> Tuple[pd.DataFrame, str]:
    print("\n" + "=" * 70)
    print("FASE 3 - ANGLE vs AMPLITUDE CON EL MISMO ENCODER: SOLO VALIDATION")
    print("=" * 70)

    rows = []
    for encoding in ("angle", "amplitude"):
        vals = []
        times = []
        grad_norms = []

        for seed in seeds:
            cfg = replace(
                base_cfg,
                seed=seed,
                n_qubits=best_n_qubits,
                n_layers=best_n_layers,
                entangler=best_entangler,
                encoding=encoding,
                epochs=full_epochs,
            )
            run_name = (
                f"encoding_{encoding}_q{best_n_qubits}_l{best_n_layers}_"
                f"{best_entangler}_seed{seed}"
            )
            out = run_selection_experiment(cfg, "encoding", run_name, "hybrid")
            vals.append(out["val_accuracy"])
            times.append(out["training_time_sec"])
            grad_norms.append(out["diagnostics"]["first_encoder_grad_norm"])

        rows.append({
            "encoding": encoding,
            "mean_val_accuracy": float(np.mean(vals)),
            "std_val_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "mean_time_sec": float(np.mean(times)),
            "mean_first_encoder_grad_norm": float(np.mean(grad_norms)),
            "n_seeds": len(vals),
        })

    df = pd.DataFrame(rows).sort_values("mean_val_accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(Path(base_cfg.results_dir) / "selection_encoding.csv", index=False)
    best_encoding = str(df.iloc[0]["encoding"])

    print("\nResultado encodings:")
    print(df.to_string(index=False))
    print(f"\nMejor encoding por VALIDATION: {best_encoding}")
    return df, best_encoding


# ============================================================
# 12. FASE 3B - MINI-GRID DE ENTRENAMIENTO
#     TAMBIEN MULTI-SEED Y SOLO VALIDATION
# ============================================================

def run_training_hparam_grid(
    base_cfg: Config,
    best_n_qubits: int,
    best_n_layers: int,
    best_entangler: str,
    best_encoding: str,
    lr_grid: Sequence[float] = (1e-3, 5e-4),
    batch_grid: Sequence[int] = (64, 128),
    seeds: Sequence[int] = SELECTION_SEEDS,
    screening_epochs: int = 8,
) -> Tuple[pd.DataFrame, float, int]:
    print("\n" + "=" * 70)
    print("FASE 3B - LR x BATCH SIZE: SOLO VALIDATION")
    print("=" * 70)

    rows = []
    for lr, bs in product(lr_grid, batch_grid):
        vals = []
        times = []

        for seed in seeds:
            cfg = replace(
                base_cfg,
                seed=seed,
                n_qubits=best_n_qubits,
                n_layers=best_n_layers,
                entangler=best_entangler,
                encoding=best_encoding,
                learning_rate=float(lr),
                batch_size=int(bs),
                epochs=screening_epochs,
            )
            run_name = f"hparam_lr{lr}_bs{bs}_seed{seed}"
            out = run_selection_experiment(cfg, "training_hparams", run_name, "hybrid")
            vals.append(out["val_accuracy"])
            times.append(out["training_time_sec"])

        rows.append({
            "learning_rate": float(lr),
            "batch_size": int(bs),
            "mean_val_accuracy": float(np.mean(vals)),
            "std_val_accuracy": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "mean_time_sec": float(np.mean(times)),
            "n_seeds": len(vals),
        })

    df = pd.DataFrame(rows).sort_values("mean_val_accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(Path(base_cfg.results_dir) / "selection_training_hparams.csv", index=False)

    best = df.iloc[0]
    best_lr = float(best["learning_rate"])
    best_bs = int(best["batch_size"])

    print("\nResultado hiperparametros de entrenamiento:")
    print(df.to_string(index=False))
    print(f"\nMejor combinacion por VALIDATION: lr={best_lr}, batch_size={best_bs}")
    return df, best_lr, best_bs


# ============================================================
# 13. CONGELAR CONFIGURACION ANTES DE TEST
# ============================================================

def freeze_final_config(
    base_cfg: Config,
    n_qubits: int,
    n_layers: int,
    entangler: str,
    encoding: str,
    learning_rate: float,
    batch_size: int,
    full_epochs: int = 15,
) -> Config:
    frozen = replace(
        base_cfg,
        n_qubits=n_qubits,
        n_layers=n_layers,
        entangler=entangler,
        encoding=encoding,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=full_epochs,
        run_name="FROZEN_BEFORE_TEST",
    )

    payload = {
        "status": "FROZEN_BEFORE_ANY_FINAL_TEST_EVALUATION",
        "timestamp_unix": time.time(),
        "config": frozen.as_dict(),
        "selection_seeds": list(SELECTION_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "selection_metric": "validation_accuracy",
        "test_used_during_selection": False,
        "encoder_definition": "784 -> encoder_hidden_dim -> n_qubits for every branch and encoding",
    }
    save_json(payload, Path(base_cfg.results_dir) / "frozen_final_config.json")
    return frozen


# ============================================================
# 14. FASE FINAL - TEST UNA VEZ CON CONFIGURACION CONGELADA
# ============================================================

def run_multiseed_final_test(
    frozen_cfg: Config,
    seeds: Sequence[int] = FINAL_SEEDS,
) -> Dict:
    print("\n" + "=" * 70)
    print(f"FASE FINAL - TEST CON CONFIGURACION CONGELADA ({len(seeds)} semillas)")
    print("=" * 70)

    results_dir = Path(frozen_cfg.results_dir)
    pred_dir = results_dir / "predictions_final"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    histories_h = []
    histories_c = []

    for seed in seeds:
        print(f"\n{'=' * 25} SEMILLA FINAL {seed} {'=' * 25}")

        cfg = replace(frozen_cfg, seed=int(seed))
        set_seed(cfg.seed)

        # Construccion pareada: encoder y clasificador deben ser exactamente iguales al inicio.
        hybrid = HybridModel(cfg)
        classical = ClassicalEquivalentModel(cfg)
        assert_shared_initialization_equal(hybrid, classical)

        # Comprobacion de parametros antes de entrenar.
        h_counts = count_params_by_component(hybrid)
        c_counts = count_params_by_component(classical)
        print("Parametros hibrido:", h_counts)
        print("Parametros clasico:", c_counts)

        # Loaders independientes pero reproducibles, para que ambos modelos vean
        # el MISMO split y el MISMO orden de minibatches para cada epoca.
        train_h, val_h = get_train_val_loaders(cfg)
        train_c, val_c = get_train_val_loaders(cfg)

        # ----------------------------------------------------
        # HIBRIDO
        # ----------------------------------------------------
        run_h = f"final_hybrid_seed{seed}"
        cfg_h = replace(cfg, run_name=run_h)
        hybrid, hist_h, time_h, diag_h = train_model(
            hybrid, cfg_h, train_h, val_h, run_h, verbose=True
        )

        # TEST se abre aqui, por primera vez para esta replica, DESPUES de congelar todo.
        test_loader_h = get_test_loader(cfg_h)
        acc_h, y_true_h, y_pred_h = predict_model(hybrid, test_loader_h)
        print(f"[{run_h}] TEST accuracy = {acc_h * 100:.2f}%")

        np.savez_compressed(
            pred_dir / f"{run_h}.npz",
            y_true=y_true_h,
            y_pred=y_pred_h,
        )

        rec_h = make_record(
            cfg_h,
            "final_test",
            "hybrid",
            time_h,
            diag_h,
            hybrid,
            test_accuracy=acc_h,
        )
        upsert_experiment(rec_h, cfg_h.results_csv)

        # ----------------------------------------------------
        # CLASICO EQUIVALENTE
        # ----------------------------------------------------
        run_c = f"final_classical_seed{seed}"
        cfg_c = replace(cfg, run_name=run_c)
        classical, hist_c, time_c, diag_c = train_model(
            classical, cfg_c, train_c, val_c, run_c, verbose=True
        )

        test_loader_c = get_test_loader(cfg_c)
        acc_c, y_true_c, y_pred_c = predict_model(classical, test_loader_c)
        print(f"[{run_c}] TEST accuracy = {acc_c * 100:.2f}%")

        if not np.array_equal(y_true_h, y_true_c):
            raise RuntimeError("Los dos modelos no fueron evaluados sobre el mismo TEST.")

        np.savez_compressed(
            pred_dir / f"{run_c}.npz",
            y_true=y_true_c,
            y_pred=y_pred_c,
        )

        rec_c = make_record(
            cfg_c,
            "final_test",
            "classical_equivalent",
            time_c,
            diag_c,
            classical,
            test_accuracy=acc_c,
        )
        upsert_experiment(rec_c, cfg_c.results_csv)

        pair_rows.append({
            "seed": int(seed),
            "hybrid_test_accuracy": acc_h,
            "classical_test_accuracy": acc_c,
            "difference_h_minus_c": acc_h - acc_c,
            "hybrid_val_accuracy": diag_h["restored_val_accuracy"],
            "classical_val_accuracy": diag_c["restored_val_accuracy"],
            "hybrid_time_sec": time_h,
            "classical_time_sec": time_c,
            "hybrid_total_params": h_counts["total"],
            "classical_total_params": c_counts["total"],
            "hybrid_middle_params": h_counts["middle"],
            "classical_middle_params": c_counts["middle"],
            "hybrid_encoder_grad_norm_first_batch": diag_h["first_encoder_grad_norm"],
        })

        histories_h.append({"seed": int(seed), "history": hist_h})
        histories_c.append({"seed": int(seed), "history": hist_c})

        del hybrid, classical
        del train_h, val_h, train_c, val_c, test_loader_h, test_loader_c
        gc.collect()

    pair_df = pd.DataFrame(pair_rows).sort_values("seed").reset_index(drop=True)
    pair_path = results_dir / "final_paired_results.csv"
    pair_df.to_csv(pair_path, index=False)

    stats_result = compute_final_statistics(pair_df)
    save_json(stats_result, results_dir / "final_statistics.json")

    print_final_statistics(stats_result)

    return {
        "pairs": pair_df,
        "statistics": stats_result,
        "histories_hybrid": histories_h,
        "histories_classical": histories_c,
    }


# ============================================================
# 15. ESTADISTICA FINAL
# ============================================================

def compute_final_statistics(pair_df: pd.DataFrame) -> Dict:
    h = pair_df["hybrid_test_accuracy"].to_numpy(dtype=float)
    c = pair_df["classical_test_accuracy"].to_numpy(dtype=float)
    d = h - c
    n = len(d)

    if n < 2:
        raise ValueError("Se necesitan al menos 2 semillas para inferencia pareada.")

    mean_h = float(np.mean(h))
    mean_c = float(np.mean(c))
    sd_h = float(np.std(h, ddof=1))
    sd_c = float(np.std(c, ddof=1))

    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se_d = sd_d / math.sqrt(n)

    t_stat, p_val = stats.ttest_rel(h, c)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d

    cohen_dz = mean_d / sd_d if sd_d > 0 else float("inf")

    # Contraste no parametrico de apoyo.
    if np.allclose(d, 0):
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    else:
        try:
            w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
            wilcoxon_stat = float(w.statistic)
            wilcoxon_p = float(w.pvalue)
        except ValueError:
            wilcoxon_stat = float("nan")
            wilcoxon_p = float("nan")

    # Diagnostico de normalidad de las diferencias (interpretar con cautela con n pequeno).
    if 3 <= n <= 5000:
        shapiro = stats.shapiro(d)
        shapiro_stat = float(shapiro.statistic)
        shapiro_p = float(shapiro.pvalue)
    else:
        shapiro_stat = float("nan")
        shapiro_p = float("nan")

    wins_h = int(np.sum(d > 0))
    wins_c = int(np.sum(d < 0))
    ties = int(np.sum(d == 0))

    return {
        "n_seeds": n,
        "hybrid_mean_accuracy": mean_h,
        "hybrid_sd_accuracy_sample": sd_h,
        "classical_mean_accuracy": mean_c,
        "classical_sd_accuracy_sample": sd_c,
        "mean_difference_h_minus_c": mean_d,
        "sd_difference_sample": sd_d,
        "ci95_difference_low": float(ci_low),
        "ci95_difference_high": float(ci_high),
        "paired_t_statistic": float(t_stat),
        "paired_t_df": n - 1,
        "paired_t_pvalue": float(p_val),
        "cohen_dz": float(cohen_dz),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "shapiro_difference_statistic": shapiro_stat,
        "shapiro_difference_pvalue": shapiro_p,
        "hybrid_wins": wins_h,
        "classical_wins": wins_c,
        "ties": ties,
        "difference_percentage_points": mean_d * 100.0,
        "ci95_percentage_points_low": ci_low * 100.0,
        "ci95_percentage_points_high": ci_high * 100.0,
    }


def print_final_statistics(s: Dict) -> None:
    print("\n" + "=" * 70)
    print("RESULTADO FINAL - COMPARACION PAREADA SOBRE TEST")
    print("=" * 70)
    print(
        f"Hibrido:             {s['hybrid_mean_accuracy']*100:.3f}% "
        f"+/- {s['hybrid_sd_accuracy_sample']*100:.3f}%"
    )
    print(
        f"Clasico equivalente: {s['classical_mean_accuracy']*100:.3f}% "
        f"+/- {s['classical_sd_accuracy_sample']*100:.3f}%"
    )
    print(
        f"Diferencia media: {s['difference_percentage_points']:+.3f} puntos porcentuales"
    )
    print(
        "IC 95% diferencia: "
        f"[{s['ci95_percentage_points_low']:+.3f}, "
        f"{s['ci95_percentage_points_high']:+.3f}] puntos porcentuales"
    )
    print(
        f"t({s['paired_t_df']})={s['paired_t_statistic']:.4f}, "
        f"p={s['paired_t_pvalue']:.6f}"
    )
    print(f"Cohen dz={s['cohen_dz']:.4f}")
    print(
        f"Wilcoxon: W={s['wilcoxon_statistic']:.4f}, "
        f"p={s['wilcoxon_pvalue']:.6f}"
    )
    print(
        f"Victorias: hibrido={s['hybrid_wins']}, "
        f"clasico={s['classical_wins']}, empates={s['ties']}"
    )

    if s["paired_t_pvalue"] < 0.05:
        print("Conclusion estadistica: diferencia significativa al nivel alpha=0.05.")
    else:
        print("Conclusion estadistica: no se detecta diferencia significativa al nivel alpha=0.05.")


# ============================================================
# 16. GRAFICAS
# ============================================================

def plot_selection_results(
    base_cfg: Config,
    coarse_df: pd.DataFrame,
    refine_df: pd.DataFrame,
    entangler_df: pd.DataFrame,
    encoding_df: pd.DataFrame,
    hparam_df: pd.DataFrame,
) -> None:
    out_dir = Path(base_cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Heatmap coarse - VALIDATION
    pivot = coarse_df.pivot(index="n_layers", columns="n_qubits", values="val_accuracy")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("n_qubits")
    ax.set_ylabel("n_layers")
    ax.set_title("Barrido coarse - validation accuracy")
    fig.colorbar(im, label="validation accuracy")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]*100:.1f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "01_coarse_validation.png", dpi=160)
    plt.close(fig)

    # 2) Refine
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"q{int(r.n_qubits)}/l{int(r.n_layers)}" for _, r in refine_df.iterrows()]
    ax.bar(labels, refine_df["mean_val_accuracy"], yerr=refine_df["std_val_accuracy"], capsize=5)
    ax.set_ylabel("Validation accuracy (media +/- SD)")
    ax.set_title("Refinamiento top-3")
    fig.tight_layout()
    fig.savefig(out_dir / "02_refine_validation.png", dpi=160)
    plt.close(fig)

    # 3) Entangler
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        entangler_df["entangler"],
        entangler_df["mean_val_accuracy"],
        yerr=entangler_df["std_val_accuracy"],
        capsize=5,
    )
    ax.set_ylabel("Validation accuracy (media +/- SD)")
    ax.set_title("Comparacion de entanglers")
    fig.tight_layout()
    fig.savefig(out_dir / "03_entangler_validation.png", dpi=160)
    plt.close(fig)

    # 4) Encoding
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        encoding_df["encoding"],
        encoding_df["mean_val_accuracy"],
        yerr=encoding_df["std_val_accuracy"],
        capsize=5,
    )
    ax.set_ylabel("Validation accuracy (media +/- SD)")
    ax.set_title("Comparacion de encoding con encoder identico")
    fig.tight_layout()
    fig.savefig(out_dir / "04_encoding_validation.png", dpi=160)
    plt.close(fig)

    # 5) LR x batch - VALIDATION
    pivot_h = hparam_df.pivot(
        index="batch_size",
        columns="learning_rate",
        values="mean_val_accuracy",
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot_h.values, aspect="auto")
    ax.set_xticks(range(len(pivot_h.columns)))
    ax.set_xticklabels(pivot_h.columns)
    ax.set_yticks(range(len(pivot_h.index)))
    ax.set_yticklabels(pivot_h.index)
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("batch_size")
    ax.set_title("Mini-grid - validation accuracy")
    fig.colorbar(im, label="validation accuracy")
    for i in range(len(pivot_h.index)):
        for j in range(len(pivot_h.columns)):
            ax.text(j, i, f"{pivot_h.values[i, j]*100:.2f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "05_hparams_validation.png", dpi=160)
    plt.close(fig)


def plot_final_results(frozen_cfg: Config, final_results: Dict) -> None:
    out_dir = Path(frozen_cfg.results_dir)
    pair_df = final_results["pairs"]
    stats_result = final_results["statistics"]

    # Curvas de validation de todas las semillas finales.
    fig, ax = plt.subplots(figsize=(8, 5))
    for item in final_results["histories_hybrid"]:
        h = item["history"]
        ax.plot(h["epoch"], h["val_acc"], alpha=0.25)
    for item in final_results["histories_classical"]:
        h = item["history"]
        ax.plot(h["epoch"], h["val_acc"], alpha=0.25, linestyle="--")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Curvas de validation - 12 semillas finales")
    fig.tight_layout()
    fig.savefig(out_dir / "06_final_learning_curves.png", dpi=160)
    plt.close(fig)

    # Distribucion test.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [
            pair_df["hybrid_test_accuracy"].to_numpy(),
            pair_df["classical_test_accuracy"].to_numpy(),
        ],
        tick_labels=["Hibrido", "Clasico"],
    )
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        f"Resultado final (n={len(pair_df)}, p={stats_result['paired_t_pvalue']:.4g})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "07_final_test_boxplot.png", dpi=160)
    plt.close(fig)

    # Diferencias pareadas por semilla.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, linewidth=1)
    ax.scatter(pair_df["seed"].astype(str), pair_df["difference_h_minus_c"] * 100.0)
    ax.set_xlabel("Semilla")
    ax.set_ylabel("Hibrido - Clasico (puntos porcentuales)")
    ax.set_title("Diferencia pareada en test por semilla")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "08_final_paired_differences.png", dpi=160)
    plt.close(fig)


def save_representative_reports(frozen_cfg: Config, final_results: Dict) -> None:
    """
    Guarda classification_report y matrices de confusion de una semilla representativa:
    la semilla cuyo accuracy hibrido esta mas cerca de la media hibrida final.
    """
    out_dir = Path(frozen_cfg.results_dir)
    pred_dir = out_dir / "predictions_final"
    pair_df = final_results["pairs"].copy()
    mean_h = pair_df["hybrid_test_accuracy"].mean()
    idx = (pair_df["hybrid_test_accuracy"] - mean_h).abs().idxmin()
    seed = int(pair_df.loc[idx, "seed"])

    h_npz = np.load(pred_dir / f"final_hybrid_seed{seed}.npz")
    c_npz = np.load(pred_dir / f"final_classical_seed{seed}.npz")

    y_true = h_npz["y_true"]
    y_pred_h = h_npz["y_pred"]
    y_pred_c = c_npz["y_pred"]

    report_h = classification_report(y_true, y_pred_h, output_dict=True, zero_division=0)
    report_c = classification_report(y_true, y_pred_c, output_dict=True, zero_division=0)
    pd.DataFrame(report_h).transpose().to_csv(out_dir / f"report_hybrid_seed{seed}.csv")
    pd.DataFrame(report_c).transpose().to_csv(out_dir / f"report_classical_seed{seed}.csv")

    cm_h = confusion_matrix(y_true, y_pred_h)
    cm_c = confusion_matrix(y_true, y_pred_c)
    pd.DataFrame(cm_h).to_csv(out_dir / f"confusion_hybrid_seed{seed}.csv", index=False)
    pd.DataFrame(cm_c).to_csv(out_dir / f"confusion_classical_seed{seed}.csv", index=False)

    metadata = {
        "representative_seed": seed,
        "criterion": "hybrid test accuracy closest to final hybrid mean",
        "hybrid_test_accuracy": float(pair_df.loc[idx, "hybrid_test_accuracy"]),
        "classical_test_accuracy": float(pair_df.loc[idx, "classical_test_accuracy"]),
    }
    save_json(metadata, out_dir / "representative_seed.json")


# ============================================================
# 17. CNN DE REFERENCIA OPCIONAL
#     SE EJECUTA SOLO DESPUES DE CONGELAR LA CONFIGURACION.
# ============================================================

def run_cnn_reference_after_freeze(
    frozen_cfg: Config,
    seed: int = 404,
) -> Dict:
    cfg = replace(
        frozen_cfg,
        seed=seed,
        run_name=f"cnn_reference_seed{seed}",
        encoding="angle",
        entangler="basic",
    )
    set_seed(cfg.seed)
    train_loader, val_loader = get_train_val_loaders(cfg)
    model = build_model(cfg, "cnn")
    model, history, t, diagnostics = train_model(
        model, cfg, train_loader, val_loader, cfg.run_name
    )
    test_loader = get_test_loader(cfg)
    acc, y_true, y_pred = predict_model(model, test_loader)

    record = make_record(
        cfg,
        "final_test",
        "cnn_reference",
        t,
        diagnostics,
        model,
        test_accuracy=acc,
    )
    upsert_experiment(record, cfg.results_csv)

    np.savez_compressed(
        Path(cfg.results_dir) / "cnn_reference_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "test_accuracy": acc,
        "history": history,
        "diagnostics": diagnostics,
        "n_params": count_params(model),
    }


# ============================================================
# 18. PIPELINE COMPLETO
# ============================================================

def run_full_pipeline(
    reset_results: bool = True,
    run_cnn_reference: bool = False,
) -> Dict:
    start = time.time()

    base_cfg = Config()
    results_dir = Path(base_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(base_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if reset_results:
        # Borra SOLO los resultados de ESTA nueva version.
        # Los notebooks/CSV antiguos permanecen intactos.
        for path in [Path(base_cfg.results_csv), results_dir / "final_paired_results.csv"]:
            if path.exists():
                path.unlink()

    manifest_path = save_environment_manifest(base_cfg)
    split_info = describe_data_split(base_cfg)
    save_json(split_info, results_dir / "data_split.json")

    print("\nEntorno guardado en:", manifest_path)
    print("Split fijo:", split_info)
    print("Semillas seleccion:", SELECTION_SEEDS)
    print("Semillas finales:", FINAL_SEEDS)
    print("TEST permanecerá cerrado hasta la fase final.\n")

    # --------------------------------------------------------
    # FASE 1A
    # --------------------------------------------------------
    coarse_df, top3_candidates = run_coarse_sweep(
        base_cfg,
        qubits_grid=(4, 6, 8),
        layers_grid=(2, 3, 4),
        sweep_epochs=6,
        seed=SELECTION_SEEDS[0],
    )

    # --------------------------------------------------------
    # FASE 1B
    # --------------------------------------------------------
    refine_df, best_n_qubits, best_n_layers = run_refine_sweep(
        base_cfg,
        top3_candidates,
        seeds=SELECTION_SEEDS,
        refine_epochs=10,
    )

    # --------------------------------------------------------
    # FASE 2
    # --------------------------------------------------------
    entangler_df, best_entangler = run_entangler_comparison(
        base_cfg,
        best_n_qubits,
        best_n_layers,
        seeds=SELECTION_SEEDS,
        full_epochs=15,
    )

    # --------------------------------------------------------
    # FASE 3
    # --------------------------------------------------------
    encoding_df, best_encoding = run_encoding_comparison(
        base_cfg,
        best_n_qubits,
        best_n_layers,
        best_entangler,
        seeds=SELECTION_SEEDS,
        full_epochs=15,
    )

    # --------------------------------------------------------
    # FASE 3B
    # --------------------------------------------------------
    hparam_df, best_lr, best_bs = run_training_hparam_grid(
        base_cfg,
        best_n_qubits,
        best_n_layers,
        best_entangler,
        best_encoding,
        lr_grid=(1e-3, 5e-4),
        batch_grid=(64, 128),
        seeds=SELECTION_SEEDS,
        screening_epochs=8,
    )

    # Graficas de seleccion. Todas usan VALIDATION.
    plot_selection_results(
        base_cfg,
        coarse_df,
        refine_df,
        entangler_df,
        encoding_df,
        hparam_df,
    )

    # --------------------------------------------------------
    # CONGELACION EXPLICITA ANTES DE TEST
    # --------------------------------------------------------
    frozen_cfg = freeze_final_config(
        base_cfg,
        n_qubits=best_n_qubits,
        n_layers=best_n_layers,
        entangler=best_entangler,
        encoding=best_encoding,
        learning_rate=best_lr,
        batch_size=best_bs,
        full_epochs=15,
    )

    # Parametros de la comparacion final ANTES de entrenar/testear.
    param_df = parameter_comparison(replace(frozen_cfg, seed=FINAL_SEEDS[0]))
    param_df.to_csv(results_dir / "final_parameter_comparison.csv", index=False)
    print("\nParametros de la comparacion final:")
    print(param_df.to_string(index=False))

    print("\nCONFIGURACION CONGELADA. A PARTIR DE AQUI SE PERMITE ABRIR TEST.")
    print(json.dumps(frozen_cfg.as_dict(), indent=2, ensure_ascii=False))

    # --------------------------------------------------------
    # FASE FINAL
    # --------------------------------------------------------
    final_results = run_multiseed_final_test(
        frozen_cfg,
        seeds=FINAL_SEEDS,
    )

    plot_final_results(frozen_cfg, final_results)
    save_representative_reports(frozen_cfg, final_results)

    cnn_result = None
    if run_cnn_reference:
        cnn_result = run_cnn_reference_after_freeze(frozen_cfg)
        print(
            f"\nCNN referencia: TEST accuracy={cnn_result['test_accuracy']*100:.2f}% "
            f"({cnn_result['n_params']} parametros)"
        )

    total_time = time.time() - start

    summary = {
        "best_n_qubits": best_n_qubits,
        "best_n_layers": best_n_layers,
        "best_entangler": best_entangler,
        "best_encoding": best_encoding,
        "best_learning_rate": best_lr,
        "best_batch_size": best_bs,
        "selection_seeds": list(SELECTION_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "split_seed": base_cfg.split_seed,
        "test_used_for_model_selection": False,
        "final_statistics": final_results["statistics"],
        "cnn_reference": cnn_result,
        "total_time_sec": total_time,
    }
    save_json(summary, results_dir / "study_summary.json")

    print("\n" + "=" * 70)
    print("ESTUDIO FINAL CORREGIDO COMPLETADO")
    print("=" * 70)
    print(f"Configuracion final:")
    print(f"  n_qubits      = {best_n_qubits}")
    print(f"  n_layers      = {best_n_layers}")
    print(f"  entangler     = {best_entangler}")
    print(f"  encoding      = {best_encoding}")
    print(f"  learning_rate = {best_lr}")
    print(f"  batch_size    = {best_bs}")
    print(f"Tiempo total: {total_time / 3600:.2f} horas")
    print(f"Resultados: {results_dir.resolve()}")
    print("\nUsa final_statistics.json y final_paired_results.csv para actualizar el TFM.")

    return {
        "coarse": coarse_df,
        "refine": refine_df,
        "entangler": entangler_df,
        "encoding": encoding_df,
        "hparams": hparam_df,
        "frozen_cfg": frozen_cfg,
        "final": final_results,
        "cnn": cnn_result,
        "summary": summary,
    }


# ============================================================
# 19. EJECUCION
# ============================================================

if __name__ == "__main__":
    # Si quieres conservar resultados parciales de una ejecucion previa,
    # cambia reset_results=False. Para el experimento definitivo recomiendo True.
    RESULTS = run_full_pipeline(
        reset_results=True,
        run_cnn_reference=False,
    )

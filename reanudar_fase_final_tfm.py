#!/usr/bin/env python3
"""
Reanuda SOLO la fase final del TFM corregido.

Requisitos:
- Ejecutar en la misma carpeta que estudio_sistematico_final_corregido.py
- Deben existir:
    ./results_final_corregido/frozen_final_config.json
    ./results_final_corregido/experiments_final_corregido.csv
    ./results_final_corregido/predictions_final/

Este script:
1. Lee la configuracion congelada original.
2. Detecta automaticamente las semillas finales ya completadas.
3. Ejecuta SOLO las semillas pendientes.
4. Reconstruye final_paired_results.csv con las 12 semillas.
5. Calcula final_statistics.json.
6. Genera graficas finales y reportes representativos.

NO repite las fases de seleccion y NO modifica la configuracion congelada.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import estudio_sistematico_final_corregido as tfm


def load_frozen_config():
    path = Path("./results_final_corregido/frozen_final_config.json")
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_ANY_FINAL_TEST_EVALUATION":
        raise RuntimeError("El frozen_final_config.json no tiene el estado esperado.")

    cfg = tfm.Config(**payload["config"])
    final_seeds = tuple(int(x) for x in payload["final_seeds"])
    return cfg, final_seeds, payload


def read_final_rows(cfg):
    csv_path = Path(cfg.results_csv)
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "phase" not in df.columns:
        return pd.DataFrame()

    df = df[df["phase"] == "final_test"].copy()
    if "run_name" in df.columns:
        df = df.drop_duplicates(subset=["run_name"], keep="last")
    return df


def seed_is_complete(seed, cfg, final_df):
    pred_dir = Path(cfg.results_dir) / "predictions_final"
    h_npz = pred_dir / f"final_hybrid_seed{seed}.npz"
    c_npz = pred_dir / f"final_classical_seed{seed}.npz"

    if final_df.empty:
        return False

    h_row = final_df[
        (final_df["seed"].astype(int) == int(seed))
        & (final_df["model_type"] == "hybrid")
    ]
    c_row = final_df[
        (final_df["seed"].astype(int) == int(seed))
        & (final_df["model_type"] == "classical_equivalent")
    ]

    return (
        len(h_row) >= 1
        and len(c_row) >= 1
        and h_npz.exists()
        and c_npz.exists()
        and pd.notna(h_row.iloc[-1]["test_accuracy"])
        and pd.notna(c_row.iloc[-1]["test_accuracy"])
    )


def run_one_seed(frozen_cfg, seed):
    print("\n" + "=" * 70, flush=True)
    print(f"REANUDACION - SEMILLA FINAL {seed}", flush=True)
    print("=" * 70, flush=True)

    cfg = replace(frozen_cfg, seed=int(seed))
    tfm.set_seed(cfg.seed)

    hybrid = tfm.HybridModel(cfg)
    classical = tfm.ClassicalEquivalentModel(cfg)
    tfm.assert_shared_initialization_equal(hybrid, classical)

    h_counts = tfm.count_params_by_component(hybrid)
    c_counts = tfm.count_params_by_component(classical)
    print("Parametros hibrido:", h_counts, flush=True)
    print("Parametros clasico:", c_counts, flush=True)

    train_h, val_h = tfm.get_train_val_loaders(cfg)
    train_c, val_c = tfm.get_train_val_loaders(cfg)

    pred_dir = Path(cfg.results_dir) / "predictions_final"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Hibrido
    run_h = f"final_hybrid_seed{seed}"
    cfg_h = replace(cfg, run_name=run_h)
    hybrid, hist_h, time_h, diag_h = tfm.train_model(
        hybrid, cfg_h, train_h, val_h, run_h, verbose=True
    )
    test_loader_h = tfm.get_test_loader(cfg_h)
    acc_h, y_true_h, y_pred_h = tfm.predict_model(hybrid, test_loader_h)
    print(f"[{run_h}] TEST accuracy = {acc_h * 100:.2f}%", flush=True)

    np.savez_compressed(
        pred_dir / f"{run_h}.npz",
        y_true=y_true_h,
        y_pred=y_pred_h,
    )

    rec_h = tfm.make_record(
        cfg_h,
        "final_test",
        "hybrid",
        time_h,
        diag_h,
        hybrid,
        test_accuracy=acc_h,
    )
    tfm.upsert_experiment(rec_h, cfg_h.results_csv)

    # Clasico equivalente
    run_c = f"final_classical_seed{seed}"
    cfg_c = replace(cfg, run_name=run_c)
    classical, hist_c, time_c, diag_c = tfm.train_model(
        classical, cfg_c, train_c, val_c, run_c, verbose=True
    )
    test_loader_c = tfm.get_test_loader(cfg_c)
    acc_c, y_true_c, y_pred_c = tfm.predict_model(classical, test_loader_c)
    print(f"[{run_c}] TEST accuracy = {acc_c * 100:.2f}%", flush=True)

    if not np.array_equal(y_true_h, y_true_c):
        raise RuntimeError("Los dos modelos no fueron evaluados sobre el mismo TEST.")

    np.savez_compressed(
        pred_dir / f"{run_c}.npz",
        y_true=y_true_c,
        y_pred=y_pred_c,
    )

    rec_c = tfm.make_record(
        cfg_c,
        "final_test",
        "classical_equivalent",
        time_c,
        diag_c,
        classical,
        test_accuracy=acc_c,
    )
    tfm.upsert_experiment(rec_c, cfg_c.results_csv)

    print(
        f"Semilla {seed} terminada | "
        f"Hibrido={acc_h*100:.2f}% | Clasico={acc_c*100:.2f}% | "
        f"Delta={(acc_h-acc_c)*100:+.2f} pp",
        flush=True,
    )

    del hybrid, classical
    del train_h, val_h, train_c, val_c, test_loader_h, test_loader_c
    gc.collect()

    return {
        "hybrid_history": hist_h,
        "classical_history": hist_c,
    }


def reconstruct_pairs(cfg, target_seeds):
    df = read_final_rows(cfg)
    rows = []

    for seed in target_seeds:
        h = df[
            (df["seed"].astype(int) == int(seed))
            & (df["model_type"] == "hybrid")
        ]
        c = df[
            (df["seed"].astype(int) == int(seed))
            & (df["model_type"] == "classical_equivalent")
        ]

        if len(h) == 0 or len(c) == 0:
            continue

        h = h.iloc[-1]
        c = c.iloc[-1]

        rows.append({
            "seed": int(seed),
            "hybrid_test_accuracy": float(h["test_accuracy"]),
            "classical_test_accuracy": float(c["test_accuracy"]),
            "difference_h_minus_c": float(h["test_accuracy"] - c["test_accuracy"]),
            "hybrid_val_accuracy": float(h["val_accuracy"]),
            "classical_val_accuracy": float(c["val_accuracy"]),
            "hybrid_time_sec": float(h["training_time_sec"]),
            "classical_time_sec": float(c["training_time_sec"]),
            "hybrid_total_params": int(h["n_trainable_params"]),
            "classical_total_params": int(c["n_trainable_params"]),
            "hybrid_middle_params": int(h["middle_params"]),
            "classical_middle_params": int(c["middle_params"]),
            "hybrid_encoder_grad_norm_first_batch": float(h["first_encoder_grad_norm"]),
        })

    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def save_final_plots(cfg, pair_df, stats_result):
    out_dir = Path(cfg.results_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [
            pair_df["hybrid_test_accuracy"].to_numpy(),
            pair_df["classical_test_accuracy"].to_numpy(),
        ],
        tick_labels=["Hibrido", "Clasico equivalente"],
    )
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        f"Resultado final (n={len(pair_df)}, p={stats_result['paired_t_pvalue']:.4g})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "07_final_test_boxplot.png", dpi=160)
    plt.close(fig)

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


def main():
    start = time.time()
    frozen_cfg, final_seeds, payload = load_frozen_config()

    print("Configuracion congelada cargada:", flush=True)
    print(json.dumps(frozen_cfg.as_dict(), indent=2, ensure_ascii=False), flush=True)
    print("Semillas objetivo:", final_seeds, flush=True)

    final_df = read_final_rows(frozen_cfg)
    completed = [
        s for s in final_seeds if seed_is_complete(s, frozen_cfg, final_df)
    ]
    missing = [s for s in final_seeds if s not in completed]

    print("Semillas ya completas:", completed, flush=True)
    print("Semillas pendientes:", missing, flush=True)

    # Reanuda unicamente lo que falta.
    for seed in missing:
        run_one_seed(frozen_cfg, seed)

    # Verificacion final.
    final_df = read_final_rows(frozen_cfg)
    completed_after = [
        s for s in final_seeds if seed_is_complete(s, frozen_cfg, final_df)
    ]
    missing_after = [s for s in final_seeds if s not in completed_after]

    print("\nSemillas completas tras reanudacion:", completed_after, flush=True)

    if missing_after:
        print("Aun faltan semillas:", missing_after, flush=True)
        raise RuntimeError("No se completaron todas las semillas finales.")

    pair_df = reconstruct_pairs(frozen_cfg, final_seeds)
    if len(pair_df) != len(final_seeds):
        raise RuntimeError(
            f"Se esperaban {len(final_seeds)} pares y se reconstruyeron {len(pair_df)}."
        )

    results_dir = Path(frozen_cfg.results_dir)
    pair_df.to_csv(results_dir / "final_paired_results.csv", index=False)

    stats_result = tfm.compute_final_statistics(pair_df)
    tfm.save_json(stats_result, results_dir / "final_statistics.json")
    tfm.print_final_statistics(stats_result)

    save_final_plots(frozen_cfg, pair_df, stats_result)

    final_results = {
        "pairs": pair_df,
        "statistics": stats_result,
        "histories_hybrid": [],
        "histories_classical": [],
    }
    tfm.save_representative_reports(frozen_cfg, final_results)

    summary = {
        "status": "RESUMED_AND_COMPLETED",
        "n_qubits": frozen_cfg.n_qubits,
        "n_layers": frozen_cfg.n_layers,
        "entangler": frozen_cfg.entangler,
        "encoding": frozen_cfg.encoding,
        "learning_rate": frozen_cfg.learning_rate,
        "batch_size": frozen_cfg.batch_size,
        "selection_seeds": payload.get("selection_seeds", []),
        "final_seeds": list(final_seeds),
        "split_seed": frozen_cfg.split_seed,
        "test_used_for_model_selection": False,
        "final_statistics": stats_result,
        "resume_time_sec": time.time() - start,
    }
    tfm.save_json(summary, results_dir / "study_summary.json")

    print("\n" + "=" * 70, flush=True)
    print("FASE FINAL REANUDADA Y COMPLETADA", flush=True)
    print("=" * 70, flush=True)
    print("Archivos principales:", flush=True)
    print(results_dir / "final_paired_results.csv", flush=True)
    print(results_dir / "final_statistics.json", flush=True)
    print(results_dir / "study_summary.json", flush=True)


if __name__ == "__main__":
    main()

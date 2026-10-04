# Clasificador de Imágenes Híbrido Cuántico-Clásico: Evaluación Comparativa de Arquitecturas Variacionales sobre MNIST

Código y resultados del Trabajo Fin de Máster de **Danna Marcela Ortiz Fonseca**
Máster en Formación Permanente en Inteligencia Artificial Aplicada — Universidad Europea de Madrid (curso 2025-2026)
Director: Ezequiel Luis Murina Moreno

## Qué estudia

Si un circuito cuántico variacional (VQC), colocado como bloque intermedio de un clasificador de imágenes, mejora el rendimiento frente a una transformación clásica comparable sobre MNIST.

| Modelo | Arquitectura | Papel |
|---|---|---|
| `HybridModel` | SharedEncoder → VQC → Classifier | Modelo bajo estudio |
| `ClassicalEquivalentModel` | SharedEncoder → Linear(8,8)+Tanh → Classifier | Baseline de ablación |
| `SimpleCNNBaseline` | Conv16 → Pool → Conv32 → Pool → FC64 → FC10 | Referencia contextual |

## Resultado principal

Configuración congelada con validación: 8 qubits, 4 capas, StronglyEntanglingLayers, AmplitudeEmbedding, learning rate 0,001, batch size 64. Evaluación en test con 12 semillas emparejadas.

| Modelo | Accuracy de test (media ± SD) | Parámetros | Tiempo medio por semilla |
|---|---|---|---|
| HybridModel | 94,73 % ± 0,31 | 50 946 | 799,5 s |
| ClassicalEquivalentModel | 95,95 % ± 0,39 | 50 922 | 126,3 s |
| SimpleCNNBaseline | 98,94 % ± 0,08 | 105 866 | 243,8 s |

Diferencia híbrido − clásico: −1,22 p.p., IC95 % [−1,54; −0,91], t(11) = −8,52, Wilcoxon p = 0,000488, Cohen dz = −2,46. El clásico equivalente supera al híbrido en las 12 semillas. Bajo este protocolo, el bloque cuántico estudiado no aporta una mejora predictiva. Fuente: `results_final_corregido/final_statistics.json` y `results_comparativa_3_modelos/pairwise_statistics.csv`.

## Estructura del repositorio

```
estudio_sistematico_final_corregido.py / .ipynb   Pipeline corregido: selección con validación y evaluación pareada
reanudar_fase_final_tfm.py                        Reanuda la fase final sin repetir la selección
comparativa_final_3_modelos_standalone.py         Evalúa la CNN con las mismas 12 semillas y compara los tres modelos
results_final_corregido/                          Resultados de selección y de la evaluación confirmatoria
results_comparativa_3_modelos/                    Resultados de la comparativa de tres modelos
exploratorio/                                     Estudio exploratorio de convergencia (18 configuraciones)
historico/                                        Versiones anteriores del código
```

## Protocolo del pipeline corregido

- El conjunto de test no interviene en ninguna decisión de selección; se abre con la configuración ya congelada (`frozen_final_config.json`).
- Split train/validation fijo (54 000 / 6 000) con `split_seed=2026`.
- Semillas de selección {42, 123, 2024} y semillas finales {7, 11, 17, 31, 73, 99, 256, 314, 512, 777, 1001, 2025}, disjuntas.
- Encoder idéntico (784 → 64 → n_qubits) para todas las ramas y para ambos encodings.
- Control explícito de que el gradiente llega al encoder con AmplitudeEmbedding.

## Reproducción

Python 3.12.3. Experimentos ejecutados en CPU con un solo hilo de PyTorch sobre el simulador `default.qubit` de PennyLane (ver `results_final_corregido/environment_manifest.json`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Selección y evaluación confirmatoria híbrido vs clásico equivalente
python estudio_sistematico_final_corregido.py

# 2. Solo si la fase final se interrumpe: completa las semillas pendientes
python reanudar_fase_final_tfm.py

# 3. CNN con las mismas 12 semillas y comparativa de tres modelos
python comparativa_final_3_modelos_standalone.py
```

MNIST se descarga automáticamente con `torchvision`. Los scripts deben ejecutarse desde la raíz del repositorio, porque leen y escriben en `./results_final_corregido/` y `./results_comparativa_3_modelos/`. Los checkpoints (`*.pt`) no se incluyen por tamaño; se regeneran al ejecutar.

## Correspondencia con los anexos de la memoria

| Anexo | Contenido | Ubicación |
|---|---|---|
| A | Código principal y scripts finales | Raíz del repositorio |
| B | Estudio exploratorio de convergencia | `exploratorio/` |
| C | Artefactos del pipeline corregido | `results_final_corregido/` (`experiments_final_corregido.csv`, `frozen_final_config.json`, `environment_manifest.json`) |
| D | Resultados confirmatorios | `results_final_corregido/` (`final_paired_results.csv`, `final_statistics.json`, `predictions_final/`) |
| E | Comparativa de tres modelos | `results_comparativa_3_modelos/` |

## Sobre `historico/`

Generaciones previas del pipeline, conservadas como historial metodológico. **No reproducen las cifras finales**: seleccionaban con accuracy de test, no fijaban el split, comparaban los encodings con encoders distintos y reutilizaban semillas entre selección y evaluación.

## Nota

En los archivos de resultados, las rutas absolutas del clúster se han sustituido por rutas relativas (`./`). Ninguna cifra ha sido modificada.

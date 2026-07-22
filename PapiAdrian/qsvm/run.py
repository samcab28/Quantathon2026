"""
Punto de entrada del QSVM cuántico (Challenge 2, Partes 3–4).

Solo modelo cuántico. El kernel se calcula en el emulador H2 de Quantinuum,
con dos backends intercambiables:  qnexus  o  pytket-quantinuum.

Consume los datos YA preparados por el equipo:
  - train: data/quantum_subset/quantum_subset_{16,32,64}.csv
  - test:  data/processed/X_test.csv + y_test.csv

Uso:
    python -m qsvm.run                                   # backend pytket-quantinuum, subset 16
    python run_qsvm.py                                   # equivalente (desde la raíz)
    python -m qsvm.run --backend qnexus                  # usar Nexus en vez de pytket-quantinuum
    python -m qsvm.run --n-subset 32 --n-test 40 --shots 512
    python -m qsvm.run --entrelazar --reps 2             # feature map ZZ
    python -m qsvm.run --device H2-1SC                   # syntax checker (gratis, para validar)

La primera corrida pide autenticación en Quantinuum (navegador o terminal).

Salidas (carpeta outputs/):
    - K_train_subsetN.npy / .png   matriz de kernel cuántico (heatmap)
    - qsvm_metrics.json            métricas del QSVM
"""
import argparse
import json
import os
import sys
import time

# Permite ejecutar este archivo DIRECTAMENTE (botón "Run" de VS Code hace
# `python qsvm/run.py`), no solo como módulo `python -m qsvm.run`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "qsvm"

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")   # tildes limpias en consola Windows
except (AttributeError, ValueError):
    pass

from .data import cargar_subset_cuantico, subconjunto_balanceado, RNG_SEED
from .kernel import matriz_kernel, matriz_kernel_cruzada
from .model import entrenar_qsvm

OUT_DIR = "outputs"


def _guardar_heatmap(K, path, titulo):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no instalado: se omite el PNG del heatmap)")
        return
    plt.figure(figsize=(5, 4))
    plt.imshow(K, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="K_ij = |<phi_i|phi_j>|^2")
    plt.title(titulo)
    plt.xlabel("muestra j"); plt.ylabel("muestra i")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Heatmap guardado en {path}")


def main():
    ap = argparse.ArgumentParser(
        description="QSVM cuántico (emulador H2 de Quantinuum) — Challenge 2.")
    ap.add_argument("--n-subset", type=int, default=16, choices=[16, 32, 64],
                    help="tamaño del subconjunto cuántico de train")
    ap.add_argument("--features", nargs="+", default=None,
                    help="columnas a usar (una por qubit). Default: las 9.")
    ap.add_argument("--n-test", type=int, default=30,
                    help="submuestreo balanceado del test (<=0 = todo el test)")
    ap.add_argument("--shots", type=int, default=512, help="shots por circuito")
    ap.add_argument("--entrelazar", action="store_true", help="feature map ZZ")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--backend", choices=["pytket-quantinuum", "qnexus"],
                    default="pytket-quantinuum",
                    help="librería de envío al emulador H2")
    ap.add_argument("--device", default=None,
                    help="device de Quantinuum. Default: H2-1E (pytket) / "
                         "H2-Emulator (qnexus). H2-1SC = syntax checker gratis.")
    ap.add_argument("--project", default="QSVM-Agua-Potable",
                    help="nombre del proyecto (solo aplica a qnexus)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # device por defecto según backend
    device = args.device or ("H2-Emulator" if args.backend == "qnexus" else "H2-1E")
    kb = dict(backend=args.backend, device_name=device, project_name=args.project)

    # ---------- PASO 1: datos ----------
    print(f"\n[1/3] Cargando subconjunto cuántico (n={args.n_subset}) ...")
    Xq, yq, X_test, y_test, feats = cargar_subset_cuantico(
        n_subset=args.n_subset, features=args.features)
    n_qubits = len(feats)
    Xte, yte = subconjunto_balanceado(X_test, y_test, n=args.n_test, seed=args.seed)
    print(f"  Features ({n_qubits} -> {n_qubits} qubits): {feats}")
    print(f"  Train cuántico: {len(Xq)}   Test evaluado: {len(Xte)}")

    # ---------- PASO 2: matriz de kernel cuántico (en H2) ----------
    print(f"\n[2/3] Kernel cuántico en H2  (backend={args.backend}, device={device}, "
          f"shots={args.shots}, entrelazar={args.entrelazar}, reps={args.reps})")
    print("  (la 1a vez se pedirá autenticación en Quantinuum)")
    n_pares_tr = len(Xq) * (len(Xq) - 1) // 2
    n_pares_te = len(Xte) * len(Xq)
    print(f"  Circuitos a enviar: {n_pares_tr} (train) + {n_pares_te} (test)")

    t0 = time.time()
    K_train = matriz_kernel(Xq, n_qubits, args.shots, args.entrelazar,
                            args.reps, **kb)
    print(f"  K_train {K_train.shape} lista en {time.time()-t0:.1f}s")
    np.save(f"{OUT_DIR}/K_train_subset{args.n_subset}.npy", K_train)
    _guardar_heatmap(K_train, f"{OUT_DIR}/K_train_subset{args.n_subset}.png",
                     f"Kernel cuántico (train, n={args.n_subset})")

    t0 = time.time()
    K_test = matriz_kernel_cruzada(Xte, Xq, n_qubits, args.shots,
                                   args.entrelazar, args.reps, **kb)
    print(f"  K_test  {K_test.shape} lista en {time.time()-t0:.1f}s")

    # ---------- PASO 3: entrenar y evaluar el QSVM ----------
    print("\n[3/3] Entrenando y evaluando el QSVM ...")
    _, m = entrenar_qsvm(K_train, yq, K_test, yte)

    print("\n================= RESULTADO QSVM =================")
    for k in ("exactitud", "precision", "recall", "f1"):
        print(f"  {k:<10}: {m[k]:.3f}")
    print(f"  (test evaluado: {len(Xte)} muestras)")

    salida = {
        "backend": args.backend, "device": device, "n_subset": args.n_subset,
        "n_qubits": n_qubits, "features": feats, "shots": args.shots,
        "entrelazar": args.entrelazar, "reps": args.reps,
        "n_test_evaluado": int(len(Xte)), "qsvm": m,
    }
    with open(f"{OUT_DIR}/qsvm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2)
    print(f"\nMétricas guardadas en {OUT_DIR}/qsvm_metrics.json")


if __name__ == "__main__":
    main()

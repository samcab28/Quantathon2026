"""
PASO 1 — Carga de datos (ya preparados por el equipo).

El pipeline del repo del equipo (src/data_prep/prepare_data.py) ya hizo la
imputación por clase, la estandarización, el balanceo y la extracción de los
subconjuntos cuánticos. Aquí SOLO se leen esos archivos:

  - train cuántico: data/quantum_subset/quantum_subset_{16,32,64}.csv
                    (features ya estandarizadas + columna Potability)
  - test:           data/processed/X_test.csv  +  data/processed/y_test.csv

Los subconjuntos son balanceados, con semilla fija (42) y extraídos SOLO del
train (nunca del test), tal como exige el enunciado.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42


def cargar_subset_cuantico(n_subset=16, features=None, data_dir="data"):
    """
    Devuelve (Xq, yq, X_test, y_test, features).

    - n_subset: tamaño del subconjunto cuántico de train (16, 32 o 64).
    - features: lista de columnas a usar (una por qubit). Si es None, usa las
      9 features completas del dataset.
    """
    sub_path = f"{data_dir}/quantum_subset/quantum_subset_{n_subset}.csv"
    df = pd.read_csv(sub_path)
    todas = [c for c in df.columns if c != "Potability"]
    feats = features if features else todas

    Xq = df[feats].values.astype(float)
    yq = df["Potability"].values.astype(int)

    X_test = pd.read_csv(f"{data_dir}/processed/X_test.csv")[feats].values.astype(float)
    y_test = pd.read_csv(f"{data_dir}/processed/y_test.csv").iloc[:, 0].values.astype(int)

    return Xq, yq, X_test, y_test, feats


def subconjunto_balanceado(X, y, n=30, seed=RNG_SEED):
    """
    Extrae n muestras balanceadas (n/2 por clase). Se usa para submuestrear
    el TEST (evaluar el QSVM contra todo el test son n_test*n_train circuitos,
    caro en el emulador). n<=0 significa 'usar todo el test'.
    """
    if n is None or n <= 0 or n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    idx0 = rng.permutation(np.where(y == 0)[0])[: n // 2]
    idx1 = rng.permutation(np.where(y == 1)[0])[: n // 2]
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return X[idx], y[idx]

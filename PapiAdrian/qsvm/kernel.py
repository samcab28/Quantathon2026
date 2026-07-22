"""
Ensamblaje de la matriz de kernel cuántico, con selector de backend.

Backends disponibles (ambos ejecutan en el emulador H2 de Quantinuum):
  - "qnexus"            -> plataforma Nexus (qsvm/nexus_backend.py)
  - "pytket-quantinuum" -> QuantinuumBackend directo (qsvm/backend_pytket.py)
"""
import numpy as np


def _dispatch(backend):
    if backend == "qnexus":
        from .nexus_backend import nexus_kernel_values
        return nexus_kernel_values
    elif backend == "pytket-quantinuum":
        from .backend_pytket import pytket_kernel_values
        return pytket_kernel_values
    raise ValueError(
        f"backend desconocido: {backend!r} "
        "(usa 'qnexus' o 'pytket-quantinuum')")


def matriz_kernel(X, n_qubits, shots_por_par=512, entrelazar=False, reps=1,
                  backend="pytket-quantinuum", device_name="H2-1E",
                  project_name="QSVM-Agua-Potable"):
    """
    Matriz de Gram simétrica NxN. Aprovecha simetría (K_ij = K_ji) y
    diagonal (K_ii = 1): solo se envían los pares del triángulo superior.
    """
    fn = _dispatch(backend)
    n = len(X)
    pares_i, pares_j, idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            pares_i.append(X[i]); pares_j.append(X[j]); idx.append((i, j))

    vals = fn(pares_i, pares_j, n_qubits, shots_por_par, entrelazar, reps,
              device_name, project_name, "qsvm_kernel_train")
    K = np.eye(n)
    for (i, j), v in zip(idx, vals):
        K[i, j] = K[j, i] = v
    return K


def matriz_kernel_cruzada(X_test, X_train, n_qubits, shots_por_par=512,
                          entrelazar=False, reps=1,
                          backend="pytket-quantinuum", device_name="H2-1E",
                          project_name="QSVM-Agua-Potable"):
    """K de forma (n_test, n_train): cada test contra cada train."""
    fn = _dispatch(backend)
    pares_i, pares_j = [], []
    for xi in X_test:
        for xj in X_train:
            pares_i.append(xi); pares_j.append(xj)
    vals = fn(pares_i, pares_j, n_qubits, shots_por_par, entrelazar, reps,
              device_name, project_name, "qsvm_kernel_test")
    return vals.reshape(len(X_test), len(X_train))

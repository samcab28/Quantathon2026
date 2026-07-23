"""
Circuitos del feature map cuántico (Pytket), compartidos por ambos backends
(qnexus y pytket-quantinuum). Ejecutados sobre el emulador H2 de Quantinuum.

Kernel:  K_ij = |<phi(x_i)|phi(x_j)>|^2  = P(|00...0>)
Se aplica  U(x_i)  seguido de  U(x_j)^dagger  y se mide: la probabilidad de
leer el estado todo-ceros es exactamente K_ij.
"""
import math
from pytket import Circuit

# Guppy usa radianes; pytket.Rz usa medias-vueltas (múltiplos de pi).
# Convertimos theta_rad -> theta/pi.
_HALF_TURN = 1.0 / math.pi


def build_feature_map(x, n_qubits, entrelazar=False, reps=1):
    """Circuito del mapa de características φ(x) (sin medición).

    Angle encoding: H + Rz(x_k) por qubit. Con `entrelazar`, añade una etapa
    ZZ (CX–Rz(x_k·x_{k+1})–CX) en cadena lineal. Se usa para el diagrama de
    circuito (deliverable de la Parte 3) y como bloque del kernel.
    """
    c = Circuit(n_qubits)
    for _ in range(reps):
        for k in range(n_qubits):
            c.H(k)
            c.Rz(x[k] * _HALF_TURN, k)
        if entrelazar:
            for k in range(n_qubits - 1):
                c.CX(k, k + 1)
                c.Rz(x[k] * x[k + 1] * _HALF_TURN, k + 1)
                c.CX(k, k + 1)
    return c


def build_overlap_circuit(xi, xj, n_qubits, entrelazar=False, reps=1):
    """Circuito pytket:  U(x_i) . U(x_j)^dagger  + medición de todos los qubits.

    - entrelazar: si True, añade una etapa ZZ (entanglement lineal).
    - reps: cuántas veces se repite el bloque del feature map.
    """
    c = Circuit(n_qubits, n_qubits)

    # ---- U(x_i): feature map forward ----
    for _ in range(reps):
        for k in range(n_qubits):
            c.H(k)
            c.Rz(xi[k] * _HALF_TURN, k)
        if entrelazar:
            for k in range(n_qubits - 1):
                c.CX(k, k + 1)
                c.Rz(xi[k] * xi[k + 1] * _HALF_TURN, k + 1)
                c.CX(k, k + 1)

    # ---- U(x_j)^dagger: todo invertido, ángulos negados ----
    for _ in range(reps):
        if entrelazar:
            for k in range(n_qubits - 1):
                c.CX(k, k + 1)
                c.Rz(-xj[k] * xj[k + 1] * _HALF_TURN, k + 1)
                c.CX(k, k + 1)
        for k in range(n_qubits):
            c.Rz(-xj[k] * _HALF_TURN, k)
            c.H(k)

    c.measure_all()
    return c


def all_zero_fraction(counts, n_qubits):
    """Fracción de shots en el estado |00...0> a partir de un dict de counts
    (llaves = tuplas de bits). Robusto a keys de longitud variable."""
    cero = tuple(0 for _ in range(n_qubits))
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get(cero, 0) / total

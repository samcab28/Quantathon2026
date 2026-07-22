"""
Backend B — pytket-quantinuum (`QuantinuumBackend`).

Envía los circuitos del kernel directamente al emulador H2 vía la API de
Quantinuum Systems (sin pasar por Nexus). Es el backend que lista el
requirements.txt del repo del equipo.

Login: pytket-quantinuum autentica de forma perezosa en la primera llamada a
`process_circuits` (pide credenciales de Quantinuum en la terminal / navegador
o usa un token cacheado). No hace falta llamar a login() explícitamente.

Device por defecto: "H2-1E" (emulador de H2-1). Alternativas: "H2-2E",
"H2-1SC" (syntax checker, gratis) o "H2-1"/"H2-2" (hardware real).
"""
import numpy as np

from .circuits import build_overlap_circuit, all_zero_fraction


def pytket_kernel_values(pairs_i, pairs_j, n_qubits, shots_por_par,
                         entrelazar=False, reps=1, device_name="H2-1E",
                         project_name=None, job_name="qsvm_kernel"):
    """Devuelve un np.array con K para cada par (mismo orden de entrada).

    `project_name` se ignora (solo aplica a Nexus); se acepta para mantener
    una firma común entre backends.
    """
    from pytket.extensions.quantinuum import QuantinuumBackend

    n_pairs = len(pairs_i)
    if n_pairs == 0:
        return np.zeros(0)

    backend = QuantinuumBackend(device_name=device_name)

    circuits = [build_overlap_circuit(xi, xj, n_qubits, entrelazar, reps)
                for xi, xj in zip(pairs_i, pairs_j)]
    # compilar al conjunto de compuertas/topología del device
    compiled = backend.get_compiled_circuits(circuits)
    # process_circuits dispara el login la 1a vez
    handles = backend.process_circuits(compiled, n_shots=shots_por_par)

    K = np.zeros(n_pairs)
    for k, h in enumerate(handles):
        counts = backend.get_result(h).get_counts()
        K[k] = all_zero_fraction(counts, n_qubits)
    return K

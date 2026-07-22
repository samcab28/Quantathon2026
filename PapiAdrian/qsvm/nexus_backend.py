"""
Backend A — Quantinuum Nexus (paquete `qnexus`).

Envía los circuitos del kernel al emulador H2 vía la plataforma Nexus.
Construye un circuito pytket por par (i, j) y los manda todos como lista en
un solo `start_execute_job`.

Login: la primera llamada abre el navegador para autenticar en Nexus; el token
queda cacheado en ~/.qnx/auth/.
"""
import numpy as np

from .circuits import build_overlap_circuit, all_zero_fraction


def nexus_kernel_values(pairs_i, pairs_j, n_qubits, shots_por_par,
                        entrelazar=False, reps=1, device_name="H2-Emulator",
                        project_name="QSVM-Agua-Potable", job_name="qsvm_kernel"):
    """Devuelve un np.array con K para cada par (mismo orden de entrada)."""
    import qnexus as qnx  # import perezoso

    n_pairs = len(pairs_i)
    if n_pairs == 0:
        return np.zeros(0)

    qnx.login()
    project = qnx.projects.get_or_create(name=project_name)
    config = qnx.QuantinuumConfig(device_name=device_name)

    refs = []
    for k, (xi, xj) in enumerate(zip(pairs_i, pairs_j)):
        circ = build_overlap_circuit(xi, xj, n_qubits, entrelazar, reps)
        refs.append(qnx.circuits.upload(
            circuit=circ, project=project, name=f"{job_name}_p{k}"))

    job = qnx.start_execute_job(
        programs=refs, n_shots=shots_por_par, backend_config=config,
        name=job_name, project=project)

    qnx.jobs.wait_for(job)
    result_refs = qnx.jobs.results(job)

    K = np.zeros(n_pairs)
    for k, rref in enumerate(result_refs):
        counts = rref.download_result().get_counts()
        K[k] = all_zero_fraction(counts, n_qubits)
    return K

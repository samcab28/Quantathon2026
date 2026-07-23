"""Kernel cuántico calculado en un backend real vía Quantinuum Nexus.

Kij se estima con el truco de "overlap" (Havlicek et al. 2019): se aplica
el mapa de características de x_i seguido del inverso (dagger) del mapa
de x_j, y se mide la probabilidad de leer el estado |0...0>. Esa
probabilidad es exactamente |<phi(xi)|phi(xj)>|^2. A diferencia de
`quantum_model.quantum_kernel`, aquí cada entrada se estima por
frecuencia de shots en un backend de Nexus (H2-E o SelenePlus), así que
sirve para cuantificar cuánto degradan el kernel el ruido real y el
muestreo finito frente a la simulación ideal.

`qnexus` se importa de forma perezosa dentro de cada función que lo
necesita, para que el resto del pipeline (y las pruebas locales de este
módulo) sigan funcionando sin esa dependencia opcional instalada.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pytket.circuit.display import render_circuit_as_html
from pytket.qasm import circuit_to_qasm_str

from quantum_model import build_feature_map


def build_overlap_circuit(row_left, row_right, name="custom", repetitions=2):
    left = build_feature_map(row_left, name, repetitions)
    right = build_feature_map(row_right, name, repetitions)
    circuit = left.copy()
    circuit.append(right.dagger())
    circuit.measure_all()
    return circuit


def zero_outcome_probability(counts, n_qubits):
    total = sum(counts.values())
    if total == 0:
        raise ValueError("El resultado no contiene shots")
    return counts.get(tuple([0] * n_qubits), 0) / total


def _pairs(n_left, n_right, symmetric):
    if symmetric:
        return [(i, j) for i in range(n_left) for j in range(i, n_right)]
    return [(i, j) for i in range(n_left) for j in range(n_right)]


def login():
    """Inicia sesión en Nexus (flujo interactivo por navegador)."""
    import qnexus as qnx

    qnx.login()


def list_devices():
    """Backends disponibles en la cuota de Nexus de la cuenta activa."""
    import qnexus as qnx

    return qnx.devices.get_all().df()


_AER_CONFIGS = {
    "aer_simulator": "AerConfig",
    "aer_simulator_statevector": "AerStateConfig",
    "aer_simulator_unitary": "AerUnitaryConfig",
}


def _backend_config(device_name):
    """Elige la clase de `BackendConfig` de qnexus según `device_name`.

    `QuantinuumConfig(device_name=...)` sólo es válido para backends de la
    familia Quantinuum (H1-*, H2-*, Helios-*). Pasarle un nombre de otra
    familia (p.ej. "aer_simulator") hace que Nexus busque una pasada de
    compilación de Quantinuum para ese nombre y falle con
    "Error retrieving compilation pass: <device_name>".
    """
    import qnexus as qnx

    config_name = _AER_CONFIGS.get(device_name)
    if config_name is not None:
        return getattr(qnx, config_name)()
    return qnx.QuantinuumConfig(device_name=device_name)


def quantum_kernel_nexus(
    X_left,
    X_right=None,
    *,
    name="custom",
    repetitions=2,
    device_name,
    n_shots=200,
    project_name="Quantathon-Challenge2",
    optimisation_level=1,
    timeout=1800,
    job_label=None,
):
    """Equivalente a `quantum_model.quantum_kernel`, ejecutado en Nexus.

    Mismo contrato de entrada/salida que la versión local: si `X_right`
    es None se calcula el kernel simétrico de `X_left` consigo mismo
    (explotando la simetría para pedir la mitad de los circuitos);
    si no, se calcula el kernel rectangular `X_left` x `X_right`.
    """
    import qnexus as qnx

    X_left = np.asarray(X_left, dtype=float)
    symmetric = X_right is None
    X_right = X_left if symmetric else np.asarray(X_right, dtype=float)
    n_qubits = X_left.shape[1]

    project = qnx.projects.get_or_create(name=project_name)
    qnx.context.set_active_project(project)
    config = _backend_config(device_name)

    pairs = _pairs(len(X_left), len(X_right), symmetric)
    label = job_label or datetime.now().strftime("%Y%m%d-%H%M%S")

    refs = [
        qnx.circuits.upload(
            circuit=build_overlap_circuit(X_left[i], X_right[j], name, repetitions),
            name=f"qkernel-{label}-{index:04d}",
        )
        for index, (i, j) in enumerate(pairs)
    ]

    compiled = qnx.compile(
        programs=refs,
        name=f"qkernel-{label}-compile",
        optimisation_level=optimisation_level,
        backend_config=config,
        project=project,
        timeout=timeout,
    )
    results = qnx.execute(
        programs=compiled,
        name=f"qkernel-{label}-execute",
        n_shots=[n_shots] * len(compiled),
        backend_config=config,
        project=project,
        timeout=timeout,
    )

    kernel = np.zeros((len(X_left), len(X_right)))
    for (i, j), result in zip(pairs, results):
        counts = result.get_counts()
        probability = zero_outcome_probability(counts, n_qubits)
        kernel[i, j] = probability
        if symmetric:
            kernel[j, i] = probability
    return kernel


def save_overlap_circuit_example(row_left, row_right, feature_map, repetitions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit = build_overlap_circuit(row_left, row_right, feature_map, repetitions)
    (output_dir / "overlap_circuit.html").write_text(
        str(render_circuit_as_html(circuit, jupyter=False)),
        encoding="utf-8",
    )
    (output_dir / "overlap_circuit.qasm").write_text(
        circuit_to_qasm_str(circuit),
        encoding="utf-8",
    )


def save_kernel_comparison_plot(simulator_kernel, nexus_kernel_matrix, output_path):
    difference = np.abs(simulator_kernel - nexus_kernel_matrix)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        ("Simulador (statevector exacto)", simulator_kernel),
        ("Nexus (emulador real)", nexus_kernel_matrix),
        ("|Diferencia|", difference),
    ]
    for ax, (title, matrix) in zip(axes, panels):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Muestra")
        fig.colorbar(image, ax=ax, fraction=0.046)
    axes[0].set_ylabel("Muestra")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

"""Pytket-native equivalents of ZZ, Pauli, and custom feature maps."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Pytket creates a configuration file on first import. Keep that side effect
# inside the repository instead of depending on or mutating the user's profile.
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("XDG_CONFIG_HOME", str(_REPO_ROOT / ".runtime_config"))

from pytket import Circuit
from pytket.circuit.display import render_circuit_as_html
from pytket.qasm import circuit_to_qasm_str


@dataclass(frozen=True)
class FeatureMapSpec:
    name: str
    repetitions: int = 2
    entanglement: str = "linear"
    feature_clip: float = 3.0
    angle_scale: float = 1.0


def _angles(values: np.ndarray, spec: FeatureMapSpec) -> np.ndarray:
    clipped = np.clip(
        np.asarray(values, dtype=float), -spec.feature_clip, spec.feature_clip
    )
    return clipped / spec.feature_clip * np.pi * spec.angle_scale


def entanglement_pairs(n_qubits: int, pattern: str) -> list[tuple[int, int]]:
    if pattern == "linear":
        return [(index, index + 1) for index in range(n_qubits - 1)]
    if pattern == "ring":
        pairs = [(index, index + 1) for index in range(n_qubits - 1)]
        if n_qubits > 2:
            pairs.append((n_qubits - 1, 0))
        return pairs
    if pattern == "full":
        return [
            (left, right)
            for left in range(n_qubits)
            for right in range(left + 1, n_qubits)
        ]
    raise ValueError(f"Unknown entanglement pattern: {pattern}")


def _rz_radians(circuit: Circuit, radians: float, qubit: int) -> None:
    circuit.Rz(float(radians / np.pi), qubit)


def _rx_radians(circuit: Circuit, radians: float, qubit: int) -> None:
    circuit.Rx(float(radians / np.pi), qubit)


def _ry_radians(circuit: Circuit, radians: float, qubit: int) -> None:
    circuit.Ry(float(radians / np.pi), qubit)


def build_feature_map(values: np.ndarray, spec: FeatureMapSpec) -> Circuit:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Feature map expects one non-empty feature vector")
    if spec.repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    angles = _angles(values, spec)
    pairs = entanglement_pairs(len(angles), spec.entanglement)
    circuit = Circuit(len(angles), name=f"{spec.name}_feature_map")

    if spec.name == "zz":
        for qubit in range(len(angles)):
            circuit.H(qubit)
        for repetition in range(spec.repetitions):
            for qubit, angle in enumerate(angles):
                _rz_radians(circuit, angle, qubit)
            for left, right in pairs:
                pair_angle = angles[left] * angles[right] / np.pi
                circuit.CX(left, right)
                _rz_radians(circuit, pair_angle, right)
                circuit.CX(left, right)
            if repetition + 1 < spec.repetitions:
                for qubit in range(len(angles)):
                    circuit.H(qubit)
    elif spec.name == "pauli":
        for qubit in range(len(angles)):
            circuit.H(qubit)
        for repetition in range(spec.repetitions):
            for qubit, angle in enumerate(angles):
                _rz_radians(circuit, angle, qubit)
                _rx_radians(circuit, 0.5 * angle, qubit)
            for left, right in pairs:
                pair_angle = (angles[left] + angles[right]) / 2
                circuit.CX(left, right)
                _rx_radians(circuit, pair_angle, right)
                circuit.CX(left, right)
            if repetition + 1 < spec.repetitions:
                for qubit in range(len(angles)):
                    circuit.H(qubit)
    elif spec.name == "custom":
        ring_pairs = entanglement_pairs(len(angles), "ring")
        for repetition in range(spec.repetitions):
            for qubit, angle in enumerate(angles):
                _ry_radians(circuit, angle, qubit)
                _rz_radians(circuit, angle * angle / np.pi, qubit)
            for left, right in ring_pairs:
                circuit.CX(left, right)
            # Data re-uploading with a deterministic cyclic permutation.
            angles = np.roll(angles, 1)
    else:
        raise ValueError(f"Unknown feature map: {spec.name}")
    return circuit


def circuit_resources(circuit: Circuit) -> dict[str, object]:
    commands = list(circuit.get_commands())
    two_qubit = sum(1 for command in commands if len(command.qubits) == 2)
    operation_counts: dict[str, int] = {}
    for command in commands:
        name = command.op.type.name
        operation_counts[name] = operation_counts.get(name, 0) + 1
    return {
        "n_qubits": circuit.n_qubits,
        "n_gates": len(commands),
        "depth": circuit.depth(),
        "two_qubit_gates": two_qubit,
        "operation_counts": operation_counts,
    }


def save_representative_circuit(
    circuit: Circuit,
    spec: FeatureMapSpec,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"feature_map_{spec.name}"
    html = render_circuit_as_html(circuit, jupyter=False)
    (output_dir / f"{stem}.html").write_text(str(html), encoding="utf-8")
    (output_dir / f"{stem}.qasm").write_text(
        circuit_to_qasm_str(circuit), encoding="utf-8"
    )
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "spec": asdict(spec),
                "resources": circuit_resources(circuit),
                "circuit": circuit.to_dict(),
            },
            handle,
            indent=2,
        )
    return circuit_resources(circuit)

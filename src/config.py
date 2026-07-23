"""Typed configuration for reproducible classical and quantum experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    raw_path: str = "data/raw/water_potability.csv"
    test_size: float = 0.20
    split_seed: int = 20260802
    subset_sizes: tuple[int, ...] = (16, 32, 64)
    subset_repeats: int = 5
    subset_seed_start: int = 20261000


@dataclass(frozen=True)
class ClassicalConfig:
    cv_splits: int = 5
    cv_seed: int = 20260723
    optuna_trials: int = 30
    nested_outer_splits: int = 5
    bootstrap_iterations: int = 2000
    primary_metric: str = "f1"
    accelerator: str = "cuda"


@dataclass(frozen=True)
class QuantumConfig:
    enabled: bool = True
    feature_maps: tuple[str, ...] = ("zz", "pauli", "custom")
    repetitions: int = 2
    entanglement: str = "linear"
    feature_clip: float = 3.0
    angle_scale: float = 1.0
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    cv_splits: int = 4
    psd_tolerance: float = 1e-10
    psd_epsilon: float = 1e-8
    backend: str = "pytket-statevector"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "full"
    data: DataConfig = field(default_factory=DataConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_tuple(values: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(values) if values is not None else default


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_raw = raw.get("data", {})
    classical_raw = raw.get("classical", {})
    quantum_raw = raw.get("quantum", {})

    data = DataConfig(
        raw_path=data_raw.get("raw_path", DataConfig.raw_path),
        test_size=float(data_raw.get("test_size", DataConfig.test_size)),
        split_seed=int(data_raw.get("split_seed", DataConfig.split_seed)),
        subset_sizes=_as_tuple(data_raw.get("subset_sizes"), DataConfig.subset_sizes),
        subset_repeats=int(data_raw.get("subset_repeats", DataConfig.subset_repeats)),
        subset_seed_start=int(
            data_raw.get("subset_seed_start", DataConfig.subset_seed_start)
        ),
    )
    classical = ClassicalConfig(
        cv_splits=int(classical_raw.get("cv_splits", ClassicalConfig.cv_splits)),
        cv_seed=int(classical_raw.get("cv_seed", ClassicalConfig.cv_seed)),
        optuna_trials=int(
            classical_raw.get("optuna_trials", ClassicalConfig.optuna_trials)
        ),
        nested_outer_splits=int(
            classical_raw.get(
                "nested_outer_splits", ClassicalConfig.nested_outer_splits
            )
        ),
        bootstrap_iterations=int(
            classical_raw.get(
                "bootstrap_iterations", ClassicalConfig.bootstrap_iterations
            )
        ),
        primary_metric=classical_raw.get(
            "primary_metric", ClassicalConfig.primary_metric
        ),
        accelerator=classical_raw.get(
            "accelerator", ClassicalConfig.accelerator
        ),
    )
    quantum = QuantumConfig(
        enabled=bool(quantum_raw.get("enabled", QuantumConfig.enabled)),
        feature_maps=_as_tuple(
            quantum_raw.get("feature_maps"), QuantumConfig.feature_maps
        ),
        repetitions=int(
            quantum_raw.get("repetitions", QuantumConfig.repetitions)
        ),
        entanglement=quantum_raw.get(
            "entanglement", QuantumConfig.entanglement
        ),
        feature_clip=float(
            quantum_raw.get("feature_clip", QuantumConfig.feature_clip)
        ),
        angle_scale=float(
            quantum_raw.get("angle_scale", QuantumConfig.angle_scale)
        ),
        c_grid=_as_tuple(quantum_raw.get("c_grid"), QuantumConfig.c_grid),
        cv_splits=int(quantum_raw.get("cv_splits", QuantumConfig.cv_splits)),
        psd_tolerance=float(
            quantum_raw.get("psd_tolerance", QuantumConfig.psd_tolerance)
        ),
        psd_epsilon=float(
            quantum_raw.get("psd_epsilon", QuantumConfig.psd_epsilon)
        ),
        backend=quantum_raw.get("backend", QuantumConfig.backend),
    )
    return ExperimentConfig(
        name=raw.get("name", config_path.stem),
        data=data,
        classical=classical,
        quantum=quantum,
    )

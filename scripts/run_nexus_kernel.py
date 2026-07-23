"""Calcula el kernel cuántico en un emulador real de Quantinuum vía Nexus
y lo compara contra la simulación local de statevector (extensión
opcional A: ejecución en hardware / análisis de ruido).

Requiere una cuenta de Nexus:
    pip install qnexus
    python scripts/run_nexus_kernel.py --list-devices
para ver los backends disponibles en tu cuota (p.ej. H2-1E, H2-2E,
Helios-1E) antes de correrlo en serio. Usa --dry-run para ver cuántos
circuitos y shots se enviarían sin consumir cuota ni requerir login.

Este script no forma parte de `python main.py`: necesita login
interactivo (`qnx.login()`), tiene latencia de cola y consume cuota de
Nexus, así que queda como paso manual y opcional.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nexus_kernel
from metrics import calculate_metrics
from prepare_data import load_data
from quantum_model import (
    _quantum_search,
    balanced_subset,
    preprocess_subset,
    quantum_kernel,
)


def _resolve(cli_value, config_dict, key, default=None):
    if cli_value is not None:
        return cli_value
    return config_dict.get(key, default)


def run(
    config,
    *,
    device_name,
    n_shots,
    subset_size,
    project_name,
    optimisation_level,
    timeout,
    output_dir,
    dry_run,
):
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    X_train, X_test, y_train, y_test = load_data(
        config["data_path"], config["test_size"], config["seed"]
    )
    feature_map = config.get("nexus", {}).get("feature_map", "custom")
    repetitions = config["circuit_repetitions"]

    X_subset, y_subset = balanced_subset(X_train, y_train, subset_size, config["seed"])
    X_scaled, X_test_scaled = preprocess_subset(X_subset, X_test)

    eval_size = min(len(X_test_scaled), subset_size)
    rng = np.random.default_rng(config["seed"])
    eval_idx = rng.choice(len(X_test_scaled), eval_size, replace=False)
    X_eval = X_test_scaled[eval_idx]
    y_eval = y_test.to_numpy()[eval_idx]

    train_pairs = subset_size * (subset_size + 1) // 2
    test_pairs = eval_size * subset_size
    print(
        f"Circuitos a enviar: {train_pairs} (kernel de entrenamiento) + "
        f"{test_pairs} (kernel de prueba) = {train_pairs + test_pairs}, "
        f"{n_shots} shots cada uno "
        f"(~{(train_pairs + test_pairs) * n_shots} shots totales)."
    )
    if dry_run:
        print("Dry-run: no se envía nada a Nexus.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Kernel de referencia (statevector local exacto)...")
    simulator_train_kernel = quantum_kernel(X_scaled, name=feature_map, repetitions=repetitions)
    simulator_test_kernel = quantum_kernel(
        X_eval, X_scaled, name=feature_map, repetitions=repetitions
    )

    print(f"2. Kernel de entrenamiento en Nexus ({device_name})...")
    nexus_train_kernel = nexus_kernel.quantum_kernel_nexus(
        X_scaled,
        name=feature_map,
        repetitions=repetitions,
        device_name=device_name,
        n_shots=n_shots,
        project_name=project_name,
        optimisation_level=optimisation_level,
        timeout=timeout,
        job_label=f"{run_stamp}-train",
    )
    print("3. Kernel de prueba en Nexus...")
    nexus_test_kernel = nexus_kernel.quantum_kernel_nexus(
        X_eval,
        X_scaled,
        name=feature_map,
        repetitions=repetitions,
        device_name=device_name,
        n_shots=n_shots,
        project_name=project_name,
        optimisation_level=optimisation_level,
        timeout=timeout,
        job_label=f"{run_stamp}-test",
    )

    train_diff = np.abs(simulator_train_kernel - nexus_train_kernel)
    diagonal_error = float(np.mean(np.abs(np.diag(nexus_train_kernel) - 1)))
    print(f"   Error absoluto medio (kernel de entrenamiento): {train_diff.mean():.4f}")
    print(f"   Desviación media de la diagonal respecto a 1: {diagonal_error:.4f}")

    nexus_kernel.save_kernel_comparison_plot(
        simulator_train_kernel,
        nexus_train_kernel,
        output_dir / "kernel_simulator_vs_nexus.png",
    )
    nexus_kernel.save_overlap_circuit_example(
        X_scaled[0], X_scaled[1], feature_map, repetitions, output_dir
    )

    print("4. Entrenando QSVM (precomputed) con cada kernel...")
    summary = {
        "device_name": device_name,
        "n_shots": n_shots,
        "subset_size": subset_size,
        "kernel_mean_absolute_error": float(train_diff.mean()),
        "kernel_diagonal_deviation": diagonal_error,
    }
    for label, train_kernel, test_kernel in [
        ("simulator", simulator_train_kernel, simulator_test_kernel),
        ("nexus", nexus_train_kernel, nexus_test_kernel),
    ]:
        search = _quantum_search(train_kernel, y_subset.to_numpy(), config["cv_folds"], config["seed"])
        predictions = search.predict(test_kernel)
        scores = search.decision_function(test_kernel)
        summary[label] = calculate_metrics(y_eval, predictions, scores)

    with (output_dir / "nexus_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {output_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--device-name",
        default=None,
        help="P.ej. H2-1E, H2-2E. Corre --list-devices para ver las opciones de tu cuota.",
    )
    parser.add_argument("--n-shots", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--optimisation-level", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Inicia sesión, imprime los backends disponibles en tu cuota y termina.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula cuántos circuitos/shots se enviarían, sin conectar a Nexus.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        import pandas as pd

        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", None)
        nexus_kernel.login()
        print(nexus_kernel.list_devices())
        return

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    nexus_config = config.get("nexus", {})

    device_name = _resolve(args.device_name, nexus_config, "device_name")
    if device_name is None and not args.dry_run:
        raise SystemExit(
            "Falta device_name. Corre 'python scripts/run_nexus_kernel.py --list-devices' "
            "para ver las opciones de tu cuota y pásalo con --device-name o en "
            "configs/full.yaml (sección nexus.device_name)."
        )

    if not args.dry_run:
        nexus_kernel.login()

    run(
        config,
        device_name=device_name,
        n_shots=_resolve(args.n_shots, nexus_config, "n_shots", 200),
        subset_size=_resolve(args.subset_size, nexus_config, "subset_size", 8),
        project_name=_resolve(
            args.project_name, nexus_config, "project_name", "Quantathon-Challenge2"
        ),
        optimisation_level=_resolve(args.optimisation_level, nexus_config, "optimisation_level", 1),
        timeout=_resolve(args.timeout, nexus_config, "timeout", 1800),
        output_dir=_resolve(args.output_dir, nexus_config, "output_dir", "results/output/nexus"),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

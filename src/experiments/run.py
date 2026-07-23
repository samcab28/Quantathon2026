"""Single, cross-platform entry point for every reported figure and number."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import optuna
import pandas as pd
import yaml

from src.classical.baseline import run_required_baseline
from src.classical.controls import run_classical_controls
from src.classical.optuna_search import run_optuna_baseline
from src.config import ROOT, ExperimentConfig, load_config
from src.data_prep.prepare_data import load_prepared, prepare_data
from src.evaluation.metrics import write_json
from src.evaluation.plotting import plot_metric_comparison
from src.quantum.experiment import run_feature_map_study, run_paired_comparison

TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "matplotlib",
    "seaborn",
    "optuna",
    "pytket",
    "pytket-quantinuum",
    "xgboost",
    "PyYAML",
]


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest() -> dict[str, object]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status_short": _git_value("status", "--short"),
    }


def make_run_id(config: ExperimentConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config.name}"


def create_run_dir(run_id: str) -> Path:
    run_dir = ROOT / "results" / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a new --run-id."
        )
    run_dir.mkdir(parents=True)
    return run_dir


def _summary_rows(
    required: dict[str, object],
    optuna_result: dict[str, object],
    controls: dict[str, object],
) -> pd.DataFrame:
    records = []
    for model, metrics in (
        ("Dummy", required["dummy"]),
        ("SVM requerida", required["required"]),
        ("SVM Optuna", optuna_result["metrics"]),
        (
            f"Control: {controls['winner_name']}",
            controls["winner_metrics"],
        ),
    ):
        records.append(
            {
                "model": model,
                **{
                    metric: metrics[metric]
                    for metric in (
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "specificity",
                        "false_positive_rate",
                        "mcc",
                        "roc_auc",
                        "average_precision",
                    )
                    if metric in metrics
                },
            }
        )
    return pd.DataFrame(records)


def run_experiment(config: ExperimentConfig, run_id: str | None = None) -> Path:
    run_id = run_id or make_run_id(config)
    run_dir = create_run_dir(run_id)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    manifest = environment_manifest()
    manifest["run_id"] = run_id
    manifest["config_name"] = config.name
    write_json(run_dir / "manifest.json", manifest)

    print("[1/7] Validating data and freezing split/subset manifests...")
    data_audit = prepare_data(config.data)
    write_json(run_dir / "data_audit.json", data_audit)
    train, test, subset_manifest = load_prepared()
    subset_manifest.to_csv(run_dir / "subset_manifest.csv", index=False)
    pd.read_csv(ROOT / "data/processed/split_manifest.csv").to_csv(
        run_dir / "split_manifest.csv", index=False
    )

    primary_map: str | None = None
    feature_study: pd.DataFrame | None = None
    if config.quantum.enabled:
        print("[2/7] Selecting the feature map using training-only CV...")
        feature_study, primary_map = run_feature_map_study(
            train,
            subset_manifest,
            config.quantum,
            run_dir,
        )
    else:
        print("[2/7] Quantum experiments disabled by configuration.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("[3/7] Running the exact required RBF-SVM grid and dummy control...")
    required = run_required_baseline(
        train,
        test,
        config.classical,
        run_dir,
        run_id,
    )
    print("[4/7] Running nested-CV Optuna RBF-SVM...")
    optuna_result = run_optuna_baseline(
        train,
        test,
        config.classical,
        run_dir,
        run_id,
    )
    print("[5/7] Benchmarking additional classical controls...")
    controls = run_classical_controls(
        train,
        test,
        config.classical,
        run_dir,
        run_id,
    )

    paired_summary: pd.DataFrame | None = None
    if config.quantum.enabled and primary_map is not None:
        print("[6/7] Running paired RBF-SVM/QSVM scaling comparison...")
        _, paired_summary = run_paired_comparison(
            train,
            test,
            subset_manifest,
            primary_map,
            config.quantum,
            config.classical,
            run_dir,
            run_id,
        )
    else:
        print("[6/7] Paired quantum comparison skipped.")

    print("[7/7] Building consolidated artifacts...")
    classical_summary = _summary_rows(required, optuna_result, controls)
    classical_summary.to_csv(
        run_dir / "metrics" / "classical_summary.csv", index=False
    )
    plot_metric_comparison(
        classical_summary,
        run_dir / "figures" / "classical_model_comparison.png",
        "Modelos clásicos sobre el test bloqueado",
    )
    summary_payload = {
        "run_id": run_id,
        "primary_feature_map": primary_map,
        "classical_models": classical_summary.to_dict(orient="records"),
        "paired_scaling": (
            [] if paired_summary is None else paired_summary.to_dict(orient="records")
        ),
        "feature_map_study_rows": (
            0 if feature_study is None else int(len(feature_study))
        ),
        "limitations": [
            "The dataset is small and observational.",
            "Water potability predictions are not a substitute for laboratory certification.",
            "The quantum experiment uses exact local statevectors, not noisy hardware.",
            "No quantum advantage is claimed.",
        ],
    }
    write_json(run_dir / "summary.json", summary_payload)
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Experiment run `{run_id}`",
                "",
                "This directory is immutable evidence for one complete run.",
                "",
                f"- Primary feature map selected on training CV: `{primary_map}`",
                "- `manifest.json`: environment, git commit, and package versions.",
                "- `split_manifest.csv`: frozen train/test membership.",
                "- `subset_manifest.csv`: paired nested subsets.",
                "- `metrics/`: CV, test, kernel, and scaling tables.",
                "- `predictions/`: per-sample predictions and scores.",
                "- `figures/`: regenerated figures.",
                "- `circuits/`: Pytket HTML/QASM/JSON circuit artifacts.",
                "",
                "The test is evaluated only by predeclared workflows after all "
                "training-only selection logic has completed.",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        ROOT / "results" / "latest_run.json",
        {"run_id": run_id, "path": str(run_dir.relative_to(ROOT))},
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full.yaml",
        help="YAML configuration relative to the repository root.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional immutable run identifier. Defaults to a UTC timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = run_experiment(config, args.run_id)
    print(f"Complete run: {run_dir}")


if __name__ == "__main__":
    main()

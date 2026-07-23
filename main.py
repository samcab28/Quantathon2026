"""Ejecuta el modelo clásico y la comparación con QSVM."""

import argparse
import json
from pathlib import Path

import yaml

from classical_model import (
    evaluate_classical_svm,
    save_confusion_matrix,
    train_classical_svm,
)
from prepare_data import data_summary, load_data
from quantum_model import (
    choose_feature_map,
    compare_models,
    preprocess_subset,
    save_circuit_example,
    save_comparison_plot,
)


def run(config_path="configs/full.yaml", quick=False):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if quick:
        config["feature_maps"] = ["custom"]
        config["subset_sizes"] = [16]
        config["repeats"] = 1
        config["map_repeats"] = 1
        config["cv_folds"] = 2

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(
        config["data_path"],
        config["test_size"],
        config["seed"],
    )

    print("1. Entrenando SVM-RBF clásica...")
    classical = train_classical_svm(
        X_train,
        y_train,
        config["cv_folds"],
        config["seed"],
    )
    classical_metrics, _ = evaluate_classical_svm(
        classical,
        X_test,
        y_test,
    )
    with (output / "classical_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(classical_metrics, file, indent=2, ensure_ascii=False)
    save_confusion_matrix(
        classical_metrics["confusion_matrix"],
        output / "classical_confusion_matrix.png",
    )

    print("2. Comparando mapas de características...")
    selected_map, map_results = choose_feature_map(
        X_train,
        y_train,
        config["feature_maps"],
        max(config["subset_sizes"]),
        config["map_repeats"],
        config["cv_folds"],
        config["seed"] + 1000,
        config["circuit_repetitions"],
    )
    map_results.to_csv(output / "feature_map_results.csv", index=False)

    print(f"3. Comparando SVM-RBF y QSVM con el mapa {selected_map}...")
    comparison = compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
        selected_map,
        config["subset_sizes"],
        config["repeats"],
        config["cv_folds"],
        config["seed"],
        config["circuit_repetitions"],
    )
    comparison.to_csv(output / "model_comparison.csv", index=False)
    save_comparison_plot(comparison, output / "model_comparison.png")

    example, _ = preprocess_subset(X_train.head(16), X_train.head(16))
    save_circuit_example(
        example[0],
        selected_map,
        config["circuit_repetitions"],
        output,
    )

    summary = {
        "data": data_summary(config["data_path"]),
        "selected_feature_map": selected_map,
        "classical": classical_metrics,
        "comparison_mean": comparison.groupby(
            ["model", "subset_size"]
        )[
            ["accuracy", "balanced_accuracy", "f1", "false_positive_rate"]
        ]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Resultados guardados en {output}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ejecuta una versión pequeña para comprobar que todo funciona.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.quick)

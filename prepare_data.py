"""Carga y separación de los datos de potabilidad."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET = "Potability"


def load_data(
    csv_path: str | Path,
    test_size: float = 0.2,
    seed: int = 20260802,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(csv_path)
    expected = FEATURES + [TARGET]
    if list(data.columns) != expected:
        raise ValueError(f"Columnas esperadas: {expected}")
    if set(data[TARGET].dropna().unique()) != {0, 1}:
        raise ValueError("Potability debe contener solamente 0 y 1")

    X = data[FEATURES]
    y = data[TARGET].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def data_summary(csv_path: str | Path) -> dict[str, object]:
    data = pd.read_csv(csv_path)
    return {
        "rows": len(data),
        "features": len(FEATURES),
        "class_counts": data[TARGET].value_counts().sort_index().to_dict(),
        "missing_values": data[FEATURES].isna().sum().to_dict(),
        "duplicates": int(data.duplicated().sum()),
    }

"""
Métricas de evaluación del QSVM (las 5 que pide la rúbrica del reto).
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)


def reportar_metricas(y_true, y_pred, titulo=None):
    """Imprime y devuelve un dict con las métricas."""
    if titulo:
        print(f"\n=== {titulo} ===")
    m = {
        "exactitud": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"  Exactitud : {m['exactitud']:.3f}")
    print(f"  Precisión : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1        : {m['f1']:.3f}")
    print(f"  Matriz de confusión:\n{confusion_matrix(y_true, y_pred)}")
    return m

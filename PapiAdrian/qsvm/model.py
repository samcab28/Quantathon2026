"""
PASO 4 — QSVM con kernel precomputado + comparación.
"""
import numpy as np
from sklearn.svm import SVC

from .metrics import reportar_metricas


def entrenar_qsvm(K_train, y_train, K_test, y_test, C=1.0):
    """
    Entrena un SVC con kernel PRECOMPUTADO.

    - K_train: (n_train, n_train) matriz de Gram del subconjunto cuántico.
    - K_test:  (n_test, n_train) similitud de cada test contra cada train.
    """
    # regularización PSD: un kernel con ruido de muestreo puede no ser
    # exactamente semidefinido positivo -> se estabiliza la diagonal.
    K_train = K_train + 1e-8 * np.eye(len(K_train))

    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    m = reportar_metricas(y_test, y_pred, titulo="QSVM (kernel cuántico)")
    return clf, m

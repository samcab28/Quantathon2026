# src/classical/

Línea base clásica (Parte 2 del enunciado) — referencia obligatoria contra la
que se compara la QSVM.

## Qué debe contener

- Entrenamiento de **SVM con kernel RBF** (`sklearn.svm.SVC`).
- **Ajuste de hiperparámetros por validación cruzada de 5 particiones** sobre
  la grilla completa: `C ∈ {0.1, 1, 10}`, `gamma ∈ {scale, auto, 0.01}`
  (usar `GridSearchCV` o equivalente, sin recortar la grilla).
- Evaluación sobre el conjunto de prueba reservado reportando **las cinco
  métricas obligatorias**: exactitud, precisión, exhaustividad (recall), F1 y
  matriz de confusión.
- Referencia citada explícitamente en el informe: Cortes & Vapnik (1995).

## Salidas esperadas

Este módulo debe dejar en `results/metrics/` una tabla con las cinco métricas
y en `results/figures/` la matriz de confusión, para que el punto de entrada
único regenere ambas.

## Estado actual

Implementado en [`baseline.py`](baseline.py): `GridSearchCV` con la grilla
completa (9 combinaciones `C × gamma`), CV de 5 particiones, `scoring="f1"`.
Total de SVMs entrenadas por corrida: 9×5 = 45 durante la búsqueda + 1
reentrenamiento final sobre todo el train balanceado (`refit=True`) = 46.
Ejecutado vía `notebooks/02_classical_baseline.ipynb`. Resultado actual:
`C=10, gamma=auto`, F1 de CV = 0.626, test → accuracy 0.60 / precisión 0.489 /
recall 0.504 / F1 0.496 (brecha CV vs. test a documentar como limitación
honesta en el informe).

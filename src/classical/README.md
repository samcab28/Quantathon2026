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
recall 0.504 / F1 0.496 (brecha CV vs. test documentada como limitación
honesta en el informe).

## Baseline extendido: [`optuna_search.py`](optuna_search.py)

Este baseline requerido por la rúbrica tenía dos debilidades metodológicas:
balancea las clases **antes** de la CV (arriesga fuga de información, sobre
todo con SMOTE) y usa una grilla fija de solo 9 combinaciones. `optuna_search.py`
las corrige:

- Usa `data/processed/X_train_raw.csv` / `y_train_raw.csv` (imputado y
  estandarizado, **sin balancear**).
- Balanceo dentro de un `imblearn.pipeline.Pipeline` (submuestreo / SMOTE /
  `class_weight='balanced'` sin resamplear), re-ajustado en cada fold de
  `RepeatedStratifiedKFold(5 splits × 3 repeats = 15 folds)` — nunca antes
  del split de CV.
- Búsqueda bayesiana con **Optuna** (TPE) sobre `C` y `gamma` en escala
  logarítmica continua, más la estrategia de balanceo como hiperparámetro
  (60 trials por defecto).

Resultado actual (mejor combinación: `C≈1.44, gamma=auto,
balance_strategy=class_weight`): F1 de CV = 0.575 (mucho más cercano al F1 de
test que el baseline de grilla, señal de una estimación menos optimista/más
honesta) → test: accuracy 0.622 / precisión 0.516 / recall 0.508 / **F1 0.512**
(vs. 0.496 del baseline de grilla). Este modelo optimizado, no el de grilla
fija, es el que se debe usar como referencia "clásico más fuerte" al comparar
contra la QSVM (criterio general de rúbrica). Ejecutado vía
`notebooks/02_classical_baseline.ipynb`.

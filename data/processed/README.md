# data/processed/

Salidas del pipeline de preparación de datos (`src/data_prep/`), generadas a
partir de `data/raw/water_potability.csv`. Nada aquí se edita a mano; todo se
regenera ejecutando el pipeline.

## Qué debe contener

- Dataset con imputación por **mediana por clase** (pH, Sulfate, Trihalomethanes).
- Dataset **estandarizado** (media 0, varianza 1) sobre las 9 features.
- Dataset **balanceado** (submuestreo o SMOTE) — original ≈60% no potable / 40% potable.
- Splits **estratificados 80/20**, guardados de forma reproducible, por ejemplo:
  - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
  - o un único `train.csv` / `test.csv` con la columna objetivo incluida.
- Metadatos de la corrida (semilla usada, método de balanceo elegido, tamaños
  resultantes) en un `metadata.json` o similar, para que el punto de entrada
  reproduzca exactamente el mismo split.

## Reglas importantes (errores comunes del enunciado)

- El split train/test se hace **antes** de cualquier balanceo/oversampling que
  dependa de las etiquetas, para evitar fuga de datos.
- El subconjunto para el experimento cuántico (16–64 muestras) se extrae
  **solo del conjunto de entrenamiento** — ver `data/quantum_subset/README.md`.

## Estado actual

Generado por `src/data_prep/prepare_data.py`, ejecutado desde
`notebooks/02_classical_baseline.ipynb`. Contiene `X_train.csv` (2044 filas,
balanceado 1022/1022 por submuestreo), `X_test.csv` (656 filas, distribución
original 400 no potable / 256 potable), `y_train.csv`, `y_test.csv` y
`metadata.json` (semilla=42, medianas de imputación por clase, conteos antes/
después de balancear).

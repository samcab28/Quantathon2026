# src/data_prep/

Código de preparación de datos (Parte 1 del enunciado). Toma
`data/raw/water_potability.csv` y produce todo lo que vive en
`data/processed/` y `data/quantum_subset/`.

## Qué debe contener

- **Carga** del CSV crudo.
- **Imputación** de NaNs por mediana por clase (pH: 491 NaN, Sulfato: 781 NaN,
  Trihalometanos: 162 NaN).
- **Estandarización** de las 9 features (media 0, varianza 1) — ajustada solo
  con el conjunto de entrenamiento y aplicada a test.
- **Split estratificado 80/20**.
- **Balanceo de clases** (submuestreo o SMOTE), aplicado después del split
  para no filtrar información del test al train.
- **Selección del subconjunto cuántico** (16–64 muestras) desde el train,
  con función que documente/parametrice la estrategia (aleatoria
  estratificada, semilla fija, etc.).

Pensar este módulo como funciones puras reutilizables (`load_raw()`,
`impute(df)`, `scale(X_train, X_test)`, `balance(X, y)`,
`select_quantum_subset(X_train, y_train, n)`) que el punto de entrada
principal (`main.py` / notebook `00_run_all`) orquesta paso a paso.

## Estado actual

Implementado en [`prepare_data.py`](prepare_data.py): `load_raw`, `split`,
`impute_median_by_class`, `standardize`, `balance` (submuestreo por defecto,
SMOTE disponible vía parámetro) y `select_quantum_subset` (muestreo aleatorio
estratificado con semilla fija, tamaños 16/32/64). Rutas ancladas a
`Path(__file__).resolve().parents[2]`, no al directorio de trabajo, para que
el resultado sea idéntico si se corre por terminal (`python -m
src.data_prep.prepare_data`) o desde un notebook. Ejecutado y verificado vía
`notebooks/02_classical_baseline.ipynb`.

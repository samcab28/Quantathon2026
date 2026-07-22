# data/quantum_subset/

Subconjunto balanceado de **16 a 64 muestras**, extraído únicamente del
**conjunto de entrenamiento** (`data/processed/`), usado para todos los
experimentos cuánticos (Partes 3 y 4 del enunciado).

## Por qué existe esta carpeta separada

El kernel cuántico requiere O(N²) ejecuciones de circuito, así que no se usa
el dataset completo. El enunciado exige:
- Documentar explícitamente la **estrategia de selección** del subconjunto.
- Preservar el **balance de clases** (potable/no potable) dentro del subconjunto.
- **Nunca** incluir muestras del conjunto de prueba (fuga de datos = error común
  señalado en el enunciado).

## Qué debe contener

- El subconjunto en sí (`quantum_subset.csv` o `.npz`), con features ya
  estandarizadas (los mapas cuánticos son sensibles a escala).
- Un archivo corto (`selection_strategy.md` o dentro de `metadata.json`)
  describiendo cómo se eligieron las muestras (aleatorio estratificado con
  semilla fija, u otro criterio) y por qué ese tamaño (16, 32, 64...).
- Si se hace el estudio de **escalado en ≥2 tamaños de problema** (requisito
  de la rúbrica, 20%), guardar aquí un subset por cada tamaño evaluado
  (ej. `quantum_subset_16.csv`, `quantum_subset_32.csv`).

## Estado actual

Generado por `src/data_prep/prepare_data.py` (función `select_quantum_subset`),
ejecutado desde `notebooks/02_classical_baseline.ipynb`. Estrategia: muestreo
aleatorio estratificado sin reemplazo, semilla fija (42), mismo número de
muestras por clase, extraído únicamente de `data/processed/X_train.csv` (ya
balanceado). Archivos generados: `quantum_subset_16.csv`, `_32.csv`, `_64.csv`
— pensados para el estudio de escalado en ≥2 tamaños (Comparación y escalado,
20% de la rúbrica). Detalle en `metadata.json`.

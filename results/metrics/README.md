# results/metrics/

Cifras numéricas reportadas en el informe, en formato tabular (CSV/JSON),
regeneradas por el punto de entrada único.

## Qué debe contener

- Tabla de métricas de la línea base clásica (exactitud, precisión, recall,
  F1) + resultados de la grilla de CV (`C × gamma`).
- Tabla de métricas de la QSVM, por mapa de características evaluado en la
  Parte 4 (ZZFeatureMap, PauliFeatureMap, custom).
- Resultados de las **ablaciones** (repeticiones, entrelazamiento, términos de
  Pauli, escalado de features, capas de data-reuploading).
- Resultados de **escalado** (métricas por tamaño de problema, ≥2 tamaños).
- Media y desviación estándar de ≥3 ejecuciones para cada configuración
  reportada.
- (Opcional) métricas de fidelidad del kernel antes/después de mitigación de
  ruido, y comparación simulador vs. hardware real.

Mantener estas tablas como la fuente numérica única que citan tanto el
informe técnico (`report/`) como las diapositivas (`slides/`).

## Estado actual

`classical_baseline.json` generado por `notebooks/02_classical_baseline.ipynb`
(vía `src/classical/baseline.py`): mejores hiperparámetros, F1 de CV, y las
cinco métricas sobre el test set. Pendiente: métricas de la QSVM (Parte 3),
tabla del estudio de mapas (Parte 4) y resultados de escalado.

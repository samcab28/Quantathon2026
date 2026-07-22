# src/utils/

Utilidades compartidas entre `data_prep/`, `classical/` y `quantum/`, para no
duplicar lógica entre el pipeline clásico y el cuántico.

## Qué debe contener

- **Métricas**: funciones que calculen y formateen exactitud, precisión,
  recall, F1 y matriz de confusión de forma consistente para ambos
  clasificadores (misma definición de métrica para que la comparación sea
  justa, tal como exige el enunciado).
- **Comparación clásico vs. cuántico**: función que arme la tabla/figura
  lado a lado de las cinco métricas para el mismo test set, y que soporte
  el escalado en ≥2 tamaños de problema.
- **Estadística de múltiples corridas**: helper para correr un experimento
  N≥3 veces y devolver media ± desviación estándar (barras de error).
- **Plotting**: heatmaps de kernel, matrices de confusión, espectros de
  valores propios — con estilo consistente para todas las figuras de
  `results/figures/`.
- **Semillas / reproducibilidad**: fijación centralizada de semillas
  aleatorias usadas por todo el pipeline.

## Estado actual

Implementado en [`plotting.py`](plotting.py): `plot_2d_decision_boundary(model,
X, y, title, out_path)` — proyecta datos de alta dimensión a 2D con PCA y
dibuja la frontera de decisión **real** del modelo ya entrenado (evalúa
`decision_function`/`predict_proba` sobre una malla en el plano PC1-PC2,
reconstruida a la dimensión original vía `PCA.inverse_transform`), no la de un
modelo sustituto entrenado directamente en 2D. Funciona igual con un `SVC` o
con un `imblearn.pipeline.Pipeline` (los samplers se ignoran en predicción).
Usado en `notebooks/02_classical_baseline.ipynb`; pensado para reutilizarse
también con la QSVM.

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

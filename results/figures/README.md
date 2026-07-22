# results/figures/

Todas las figuras reportadas en el informe técnico y en la presentación,
regeneradas automáticamente por el punto de entrada único (nada se edita a
mano ni se pega aquí manualmente).

## Qué debe contener

- Diagrama(s) del circuito del mapa de características cuántico.
- Heatmap(s) de la matriz de kernel cuántico (y, si aplica, por cada mapa
  estudiado en la Parte 4).
- Matrices de confusión: SVM clásica y QSVM.
- Gráfico comparativo lado a lado de las cinco métricas (clásico vs.
  cuántico) con barras de error (media ± std de ≥3 corridas).
- Gráfico de escalado (métrica vs. tamaño del problema, ≥2 tamaños).
- Espectro de valores propios / rango efectivo por mapa de características.
- (Opcional) gráficos de comparación simulador vs. hardware real, y de mejora
  por mitigación de ruido (ZNE, readout).

## Convención de nombres sugerida

`fig_<tema>_<detalle>.png`, ej. `fig_kernel_heatmap_zzfeaturemap.png`,
`fig_confusion_matrix_qsvm.png`, `fig_metric_comparison.png`.

## Estado actual

`fig_confusion_matrix_classical.png` generado por
`notebooks/02_classical_baseline.ipynb`. Pendiente: diagrama de circuito,
heatmap(s) de kernel, matriz de confusión de la QSVM, comparativas y gráfico
de escalado.

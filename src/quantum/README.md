# src/quantum/

Núcleo cuántico del reto (Partes 3 y 4). Implementado en **Pytket y/o Guppy**,
ejecutado sobre el **emulador H2 de Quantinuum** (hasta 26 qubits exactos).

## Qué debe contener

### Parte 3 — Kernel cuántico
- Definición de al menos un **mapa de características φ(x)** (circuito
  parametrizado que codifica cada muestra en un estado cuántico).
- Cálculo de la **matriz de kernel** `Kij = |⟨φ(xi)|φ(xj)⟩|²` sobre el
  `data/quantum_subset/`.
- Regularización si el kernel no es semidefinido positivo (sumar `epsilon * I`
  antes de pasarlo a `SVC(kernel='precomputed')`).
- Utilidad para generar el **diagrama del circuito** y el **heatmap** de K
  (guardar en `results/figures/`).
- Registro del **número de qubits** usado y explicación del cálculo (para el
  informe).

### Parte 4 — Estudio de mapas de características
- Implementación de **ZZFeatureMap**, **PauliFeatureMap** y un **mapa
  personalizado**.
- Para cada mapa, calcular: exactitud / exactitud balanceada / F1 vía CV,
  alineación kernel-objetivo, similitudes intra/interclase, distribución y
  heatmap del kernel, espectro de valores propios y rango efectivo,
  profundidad del circuito transpilado, número de compuertas de dos qubits,
  sensibilidad a muestreo finito/ruido/topología de entrelazamiento, y costo
  computacional.
- **Ablaciones**: variar repeticiones, entrelazamiento, términos de Pauli,
  escalado de features y capas de data-reuploading.

### Extensiones opcionales
- `hardware/` o módulo equivalente para ejecución en backend real y
  comparación con el simulador.
- Mitigación de ruido: ZNE + mitigación de errores de lectura, reportando
  mejora en la fidelidad del kernel.

## Errores comunes a evitar (del enunciado)

- Nunca usar muestras de test en el subconjunto cuántico.
- Estandarizar features antes de codificar (sensibilidad a escala).
- Reportar media ± desviación estándar de ≥3 ejecuciones, no una sola corrida.

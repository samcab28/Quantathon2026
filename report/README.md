# report/

Informe técnico escrito, entregable obligatorio del reto.

## Qué debe contener

- **Fuente del informe** (Markdown/LaTeX/Word) y el **PDF final**, máximo
  **8 páginas**.
- Secciones obligatorias:
  1. Planteamiento del problema (contexto ODS 6, dataset, objetivo).
  2. Línea base clásica (SVM-RBF, metodología de CV, resultados con las
     cinco métricas, cita a Cortes & Vapnik 1995).
  3. Resumen de la implementación cuántica (mapa de características, cálculo
     del kernel, número de qubits, resultados de la QSVM).
  4. Comparación clásico vs. cuántico y estudio de mapas (Parte 4), con
     resultados y **barras de error** (media ± std de ≥3 corridas).
  5. Conexión con el ODS 6: submeta específica, cadena causal hacia un
     resultado real, consideración de escala/costo/implementación, y
     mención de ≥2 ODS adicionales.
  6. **Sección honesta de limitaciones** (obligatoria): dataset pequeño,
     qubits limitados, efectos de ruido, ausencia de ventaja cuántica
     demostrada, sobrecarga O(N²).
  7. (Opcional) resultados de hardware real y/o mitigación de ruido.
- **Declaración de SDK** (≤200 palabras): qué funcionó, qué no, y qué faltó
  usando Pytket/Guppy — puede ir como anexo del mismo PDF o como archivo
  separado (`sdk_statement.md`) en esta carpeta.

Todas las cifras y figuras citadas aquí deben venir de `results/metrics/` y
`results/figures/`, generadas por el punto de entrada único (no se recalculan
a mano para el informe).

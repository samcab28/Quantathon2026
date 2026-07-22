# Quantathon CR 2026 · Challenge 2 — Agua limpia (QSVM vs SVM)

Clasificador de potabilidad del agua: comparación de una SVM clásica (kernel RBF)
contra una QSVM (kernel cuántico) sobre el dataset [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
de Kaggle. Ejecutado sobre el emulador H2 de Quantinuum (Pytket/Guppy). Conecta con el ODS 6.

El enunciado completo del reto está en [`Enunciado/`](Enunciado/) y el análisis de
requerimientos derivado de él se resume más abajo.

## Estructura del proyecto

```
Enunciado/            enunciado original (.md, .docx, .pdf) — no modificar
data/
  raw/                CSV original de Kaggle, sin tocar
  processed/          datos imputados, estandarizados y balanceados (train/test)
  quantum_subset/      subconjunto de 16-64 muestras para el experimento cuántico
notebooks/            exploración y prototipado (EDA, pruebas de mapas de features)
src/
  data_prep/          carga, imputación, escalado, balanceo, split, selección del subset cuántico
  classical/          SVM-RBF baseline + tuning por CV
  quantum/             feature maps, cálculo de la matriz de kernel, QSVM, estudio de mapas
  utils/               métricas, plotting, utilidades compartidas
results/
  figures/            diagramas de circuito, heatmaps de kernel, matrices de confusión, comparativas
  metrics/            tablas de métricas, resultados de CV y de ablación (csv/json)
report/               informe técnico (PDF ≤ 8 páginas) y declaración de SDK (≤200 palabras)
slides/               presentación de 5 minutos
scripts/              scripts de PowerShell: setup, ejecución y reset del entorno
requirements.txt      dependencias para reproducibilidad
```

Cada carpeta tiene su propio `README.md` explicando qué debe contener.

## Punto de entrada de reproducibilidad

El reto exige **un único script o notebook de punto de entrada** que reproduzca
cada figura y cifra reportada. Ese punto de entrada es
**[`scripts/run_all.ps1`](scripts/run_all.ps1)**: crea/actualiza `.venv`,
instala `requirements.txt` y ejecuta en orden todos los `notebooks/NN_*.ipynb`
(`01_eda` → `02_classical_baseline` → `03_quantum_kernel` → `04_feature_map_study`
a medida que se van creando), cada uno llamando a funciones de `src/` sin
duplicar lógica. Ver [`scripts/README.md`](scripts/README.md) para el resto de
scripts (`setup.ps1`, `run_notebooks.ps1`, `activate.ps1`, `reset.ps1`). Para
volver a un estado limpio y repetir todo desde cero cuantas veces se quiera:

```powershell
.\scripts\reset.ps1 -Force
.\scripts\run_all.ps1
```

## Estado de los datos

El CSV de Kaggle ya está descargado manualmente en
`data/raw/water_potability.csv` (ver [`data/raw/README.md`](data/raw/README.md)
para volver a obtenerlo desde cero). No se sube a git.

## Estado actual del proyecto

- [x] Entorno virtual `.venv` + `requirements.txt` instalados.
- [x] `notebooks/01_eda.ipynb` — EDA del CSV crudo, ejecutado y verificado.
- [x] `notebooks/02_classical_baseline.ipynb` — ejecuta `src/data_prep/prepare_data.py`
      (split 80/20, imputación por mediana por clase, estandarización, subsets
      cuánticos 16/32/64) y dos clasificadores clásicos:
      - `src/classical/baseline.py` — SVM-RBF, `GridSearchCV` 5-fold sobre la
        grilla completa (requisito de rúbrica). Test: accuracy 0.60, F1 0.496
        (`C=10, gamma=auto`).
      - `src/classical/optuna_search.py` — baseline extendido: balanceo dentro
        de cada fold de CV (en vez de antes, evita fuga) + búsqueda bayesiana
        con Optuna (60 trials) sobre C/gamma/estrategia de balanceo. Test:
        accuracy 0.622, **F1 0.512** (`C≈1.44, gamma=auto, class_weight`) — este
        es el modelo a usar como referencia "clásico más fuerte" frente a la QSVM.
      - `src/utils/plotting.py` — proyección PCA 2D con la frontera de decisión
        real de cada modelo (`results/figures/fig_decision_boundary_*_pca.png`).
- [ ] `notebooks/03_quantum_kernel.ipynb` — mapa de características + matriz de
      kernel + diagrama de circuito + heatmap (Parte 3, siguiente paso).
- [ ] `notebooks/04_feature_map_study.ipynb` — estudio comparativo de mapas
      (ZZFeatureMap, PauliFeatureMap, custom) con ablaciones (Parte 4).

## Checklist de entrega (ver enunciado para el detalle completo)

- [x] Línea base SVM-RBF con CV de 5 particiones sobre la grilla completa + 5 métricas + matriz de confusión
- [ ] Mapa de características cuántico + matriz de kernel + diagrama de circuito + heatmap
- [ ] Estudio comparativo de mapas (ZZFeatureMap, PauliFeatureMap, custom) con ablaciones
- [ ] QSVM entrenada y comparada lado a lado con la baseline (mismas métricas, mismo test set)
- [ ] Escalado en ≥2 tamaños de problema + media/std de ≥3 corridas
- [ ] Sección honesta de limitaciones
- [ ] (Opcional) ejecución en hardware real y/o mitigación de ruido (ZNE, readout)
- [ ] Repo público + `requirements.txt` + punto de entrada único + `README.md`
- [ ] Informe técnico PDF (≤8 páginas), presentación (5 min), declaración de SDK (≤200 palabras)

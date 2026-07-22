# notebooks/

La lógica reutilizable vive en `src/` (funciones puras); estos notebooks son
el **registro de ejecución**: importan esas funciones (nunca duplican su
lógica) y muestran/verifican sus resultados de forma visible e inspeccionable.
También sirven para EDA y prototipado antes de mover código a `src/`.

## Convención (numeración = orden de ejecución)

- [x] `01_eda.ipynb` — exploración del dataset crudo (`data/raw/`). Ejecutado;
      confirma shape (3276×10), NaNs por columna (pH 491, Sulfato 781,
      Trihalometanos 162) y balance de clases (~61/39%) frente a lo esperado
      por el enunciado.
- [x] `02_classical_baseline.ipynb` — llama a `src/data_prep/prepare_data.main()`,
      `src/classical/baseline.train_and_evaluate(...)` (baseline de grilla fija,
      Parte 2) y `src/classical/optuna_search.main(...)` (baseline extendido:
      balanceo dentro de CV + búsqueda bayesiana con Optuna, F1 test 0.496 →
      0.512), más `src/utils/plotting.plot_2d_decision_boundary(...)` para la
      proyección PCA 2D con la frontera de decisión real de cada modelo.
      Ejecutado; regenera `data/processed/`, `data/quantum_subset/`, ambos
      `results/metrics/classical_*.json` y las figuras de confusión/frontera
      en `results/figures/`.
- [ ] `03_quantum_kernel.ipynb` — mapa de características + matriz de kernel
      cuántico (Parte 3, próximo paso), llamando a `src/quantum/`.
- [ ] `04_feature_map_study.ipynb` — comparación ZZFeatureMap / PauliFeatureMap /
      custom con ablaciones (Parte 4).

Cada notebook nuevo se ejecuta con `jupyter nbconvert --to notebook --execute
--inplace notebooks/0X_....ipynb` y se revisan sus outputs antes de darlo por
terminado (no basta con que el código "se vea bien", tiene que correr limpio).

Al cerrar el proyecto, un `00_run_all.ipynb` (o `main.py` en la raíz) deberá
ejecutar 01→04 en secuencia como el **punto de entrada único** exigido por el
enunciado para reproducibilidad.

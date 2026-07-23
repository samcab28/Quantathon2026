# QSVM cuántico — potabilidad del agua (Challenge 2, Partes 3–4)

Contribución al Quantathon CR 2026. Entrena y evalúa **únicamente un SVM con
kernel cuántico (QSVM)**; la comparación clásica ya vive en `src/classical/`
del repo del equipo. El kernel `K_ij = |⟨φ(xᵢ)|φ(xⱼ)⟩|²` se calcula en el
**emulador H2 de Quantinuum**, con **dos backends intercambiables**:

- `pytket-quantinuum` — `QuantinuumBackend(device_name="H2-1E")` (default)
- `qnexus` — plataforma Quantinuum Nexus (`device_name="H2-Emulator"`)

Consume los datos **ya preparados** por el equipo (imputados, estandarizados,
balanceados): los subconjuntos cuánticos de `data/quantum_subset/` para train y
`data/processed/X_test.csv` para test.

## Instalación

```bash
pip install -r requirements.txt
```

Usa **Python 3.12** (donde están instaladas las librerías), no el 3.14.
Necesitas una **cuenta de Quantinuum** para autenticar al enviar circuitos.

## Uso rápido

```bash
# Corrida por defecto: backend pytket-quantinuum, subset 16, device H2-1E
python run_qsvm.py

# Validar circuitos sin gastar (syntax checker, gratis)
python run_qsvm.py --device H2-1SC --n-subset 16 --n-test 8

# Usar el backend Nexus en vez de pytket-quantinuum
python run_qsvm.py --backend qnexus

# Estudio de escalado (rúbrica): repetir con 16, 32 y 64
python run_qsvm.py --n-subset 32 --n-test 40 --shots 512
python run_qsvm.py --n-subset 64 --n-test 40 --shots 512

# Feature map ZZ con entanglement y 2 repeticiones
python run_qsvm.py --entrelazar --reps 2

# Evaluar contra TODO el test (caro): --n-test 0
python run_qsvm.py --n-test 0
```

`run_qsvm.py` (raíz) equivale a `python -m qsvm.run`.

## Backends (ambos ejecutan en el emulador H2)

| Backend | Librería | Device default | Login |
|---------|----------|----------------|-------|
| `pytket-quantinuum` (default) | `QuantinuumBackend` | `H2-1E` | perezoso, en la 1a llamada |
| `qnexus` | plataforma Nexus | `H2-Emulator` | navegador, token en `~/.qnx/auth/` |

Devices: `H2-1E`/`H2-2E` (emulador), `H2-1SC` (syntax checker, **gratis**, para
validar circuitos sin gastar), `H2-1`/`H2-2` (hardware real, gasta créditos).

Cada circuito es estático, así que se construye **un circuito pytket por par**
(i, j) y se envían todos juntos. El feature map por defecto es *angle encoding*
(una `Rz` por qubit); con `--entrelazar` se añade una etapa `ZZ` con
entanglement lineal.

Salidas en `outputs/`: `K_train_subsetN.npy` + `.png` (heatmap del kernel,
entregable de la rúbrica) y `qsvm_metrics.json` (métricas del QSVM).

## Estructura del paquete

```
qsvm/
├── data.py            # Paso 1: carga subset cuántico (train) + test del equipo
├── circuits.py        # feature map en Pytket (compartido por ambos backends)
├── backend_pytket.py  # Backend A: pytket-quantinuum (QuantinuumBackend)
├── nexus_backend.py   # Backend B: qnexus (plataforma Nexus)
├── kernel.py          # Paso 2: ensambla la matriz K, despacha al backend
├── model.py           # Paso 3: QSVM (SVC con kernel precomputado)
├── metrics.py         # métricas (exactitud, precisión, recall, F1)
└── run.py             # orquestador (python -m qsvm.run)
```

## Parámetros (`python -m qsvm.run --help`)

| Flag | Qué hace | Default |
|------|----------|---------|
| `--backend` | `pytket-quantinuum` o `qnexus` | pytket-quantinuum |
| `--n-subset` | subconjunto cuántico de train (16, 32, 64) | 16 |
| `--features` | columnas a usar (1 por qubit). Default: las 9 | las 9 |
| `--n-test` | submuestreo balanceado del test (0 = todo) | 30 |
| `--shots` | shots por circuito | 512 |
| `--entrelazar` | activa la etapa ZZ (entanglement) | off |
| `--reps` | repeticiones del feature map | 1 |
| `--device` | device de Quantinuum | H2-1E / H2-Emulator |
| `--project` | proyecto en Nexus (solo `qnexus`) | QSVM-Agua-Potable |

> **Costo/tiempo**: train envía `n·(n-1)/2` circuitos y test `n_test·n_train`.
> Con `--n-subset 16 --n-test 30` son ~600 circuitos. Valida antes con
> `--device H2-1SC` (gratis) y sube el tamaño despacio en el emulador.

## Notas del enunciado respetadas

- El subconjunto cuántico se extrae **solo del train** (nunca del test).
- Las features van **estandarizadas** antes de codificar (el equipo ya lo hizo).
- Para el informe: reportar **media ± desviación de ≥3 corridas** (variar `--seed`).

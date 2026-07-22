# Aqua Collector — recolector multi-país de calidad de agua

Herramienta base para el Quantathon CR 2026 (Track 02). Recolecta datos reales
de calidad de agua de portales oficiales latinoamericanos y los estandariza a un
esquema común, listo para alimentar el modelo de ML/QML.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# 1) Ver países disponibles
python main.py --list

# 2) La PRIMERA vez con un país: ver las columnas crudas de la fuente
python main.py --country colombia --inspect

# 3) Recolectar y mostrar en terminal
python main.py --country colombia --year 2019 --limit 200

# 4) Guardar en Excel o CSV
python main.py --country colombia --region ANTIOQUIA --output excel --file antioquia.xlsx
python main.py --country colombia --year 2019 --output csv --file datos.csv
```

## Estructura

```
aqua_collector/
├── main.py            # CLI (elige país, filtros, salida terminal/excel/csv)
├── schema.py          # esquema común + regla de etiqueta (IRCA -> potable 0/1)
├── requirements.txt
└── sources/
    ├── __init__.py    # registro de países
    ├── base.py        # clase base (fetch_raw + normalize)
    ├── colombia.py    # FUNCIONAL: SIVICAP/INS vía API Socrata
    └── stubs.py       # plantillas Brasil y México (por completar)
```

## Pendiente / para el equipo

1. **Confirmar columnas de Colombia.** Corre `--inspect` una vez y ajusta el
   `COLUMN_MAP` en `sources/colombia.py` con los nombres reales de la API.
   (Los valores actuales son la mejor conjetura y están marcados `# TODO`.)
2. **Brasil** (`stubs.py::BrazilSource`): implementar descarga del SISAGUA y
   derivar la etiqueta con la norma brasileña (no trae IRCA).
3. **México** (`stubs.py::MexicoSource`): descargar CSV de RENAMECA/SINA.
   Ojo: es agua superficial, no de red — no mezclar sin aclararlo.
4. **Costa Rica**: los datos del LNA/AyA están en PDF, no en portal descargable.
   Requiere un extractor de PDF aparte (dejar para el final si sobra tiempo).

## Nota sobre las etiquetas

Colombia trae el IRCA ya calculado, así que la potabilidad sale directa:
`IRCA <= 5` = potable, `IRCA > 5` = no potable (Resolución 2115/2007).
Países sin índice oficial derivan la etiqueta aplicando umbrales OMS o la norma
local dentro de su `normalize()`.

---

# Modelo QSVM (potabilidad con kernel cuántico)

El paquete `qsvm/` entrena y evalúa **únicamente un SVM con kernel cuántico
(QSVM)**. El kernel se calcula **siempre en Quantinuum Nexus** (device `H1-1E`).
No entrena ningún modelo clásico. Es la segunda mitad del pipeline: consume el
CSV del recolector (o el dataset de Kaggle) con una columna `Potability` (0/1).

## Instalación

```bash
pip install -r requirements.txt
```

Necesitas además una **cuenta activa de Quantinuum Nexus** para autenticar.

## Uso rápido

```bash
# 1) Generar un dataset de respaldo (esquema Kaggle) si aún no hay datos reales
python -m qsvm.make_dataset            # crea data/water_potability.csv

# 2) Correr el QSVM (rápido, para verificar de punta a punta)
python -m qsvm.run --n-train 8 --n-test 8 --shots 256

# 3) Corrida "de verdad" para el informe
python -m qsvm.run --n-train 16 --n-test 30 --shots 512

# 4) Subir complejidad: feature map ZZ con entanglement y 2 repeticiones
python -m qsvm.run --entrelazar --reps 2

# 5) Usar el CSV real del recolector (debe tener columna 'Potability')
python -m qsvm.run --data data/colombia.csv --features ph Sulfate Chloramines Solids

# 6) Correr en hardware real en vez del emulador
python -m qsvm.run --device H1-1 --n-train 8 --n-test 8
```

## Backend: Quantinuum Nexus

El kernel se ejecuta en Nexus. Devices:
- `H1-1E` / `H2-Emulator` — **emuladores** (default, recomendado para desarrollar)
- `H1-1` / `H2-1` — **hardware real** (gasta créditos y hace cola)

**Login**: la primera corrida abre el navegador para autenticar en Nexus. El
token queda cacheado en `~/.qnx/auth/`. Para entornos headless usa
`qnx.login_with_token(...)` (ver docstring en `qsvm/nexus_backend.py`).

En Nexus cada circuito es estático, así que el backend construye **un circuito
pytket por par** (i, j) y los envía todos como lista en un solo
`start_execute_job` (ver `qsvm/nexus_backend.py`).

Salidas en `outputs/`: `K_train.npy` + `K_train.png` (heatmap de la matriz de
kernel, entregable de la rúbrica) y `metrics.json` (métricas del QSVM).

## Cómo funciona el kernel cuántico

`K_ij = |⟨φ(x_i)|φ(x_j)⟩|²`. Se calcula aplicando `U(x_i)` seguido de
`U(x_j)†` y midiendo: la probabilidad de leer `|00…0⟩` es exactamente `K_ij`.
El feature map por defecto es *angle encoding* (una rotación `Rz` por qubit);
con `--entrelazar` se añade una etapa `ZZ` con entanglement.

## Estructura del paquete

```
qsvm/
├── data.py            # Paso 1: carga, imputa por clase, escala, split
├── nexus_backend.py   # Paso 2: circuitos pytket + matriz K en Quantinuum Nexus
├── model.py           # Paso 3: QSVM (SVC con kernel precomputado)
├── metrics.py         # métricas de evaluación (exactitud, precisión, recall, F1)
├── run.py             # orquesta los 3 pasos (python -m qsvm.run)
└── make_dataset.py    # genera dataset de respaldo tipo Kaggle
```

## Parámetros clave (`python -m qsvm.run --help`)

| Flag | Qué hace | Default |
|------|----------|---------|
| `--n-qubits` / `--features` | features usadas (1 por qubit) | 4 |
| `--n-train` | tamaño del subconjunto cuántico (matriz K es O(n²)) | 16 |
| `--n-test` | subconjunto balanceado para evaluar el QSVM | 30 |
| `--shots` | shots por par (más = menos ruido en K) | 512 |
| `--entrelazar` | activa la etapa ZZ (entanglement) | off |
| `--reps` | repeticiones del feature map | 1 |
| `--device` | device de Nexus (H1-1E, H2-Emulator, H1-1, ...) | H1-1E |
| `--project` | nombre del proyecto en Nexus | QSVM-Agua-Potable |

**Empieza simple**: angle encoding, pocos qubits, n pequeño. Solo cuando salgan
números sube `--n-train`, activa `--entrelazar` y sube `--reps`.

> **Ojo con el costo/tiempo en Nexus**: la matriz de train envía `n·(n-1)/2`
> circuitos y la de test `n_test·n_train`. Con `--n-train 16 --n-test 30` son
> ~600 circuitos. En emulador es manejable; en hardware real sube el número
> despacio.

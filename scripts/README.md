# scripts/

Scripts de PowerShell (Windows PowerShell 5.1+) para instalar el entorno,
ejecutar el pipeline completo y resetear todo para volver a empezar. Ninguno
depende del directorio desde donde se invoque: todos resuelven las rutas a
partir de `$PSScriptRoot` (la carpeta donde vive el script), nunca del
directorio de trabajo actual, así que funcionan igual llamados desde la raíz
del repo, desde dentro de `scripts\`, o con ruta absoluta desde cualquier
lado.

## Scripts

| Script | Qué hace |
|---|---|
| `setup.ps1` | Crea `.venv\` si no existe, instala `requirements.txt`, registra un kernel de Jupyter propio (`quantathon-ch2`) apuntando a ese venv. Idempotente: correrlo de nuevo no rehace lo que ya está. |
| `activate.ps1` | Activa el venv en tu shell **actual** (para trabajar interactivamente). Debe invocarse con `. ` (dot-source) al inicio, si no la activación desaparece al terminar el script. |
| `run_notebooks.ps1` | Descubre todos los `notebooks\NN_*.ipynb` (orden numérico) y los ejecuta en orden con `jupyter nbconvert --execute --inplace`, usando siempre el Python del venv y el kernel `quantathon-ch2` (nunca "el python3 que Jupyter encuentre primero" — evita que en la máquina de un compañero se ejecute con un Python/Anaconda distinto por accidente). Se detiene en el primer notebook que falle. |
| `run_all.ps1` | Orquestador: `setup.ps1` + `run_notebooks.ps1` en una sola llamada. Este es el **punto de entrada único de reproducibilidad** que exige el enunciado. |
| `reset.ps1` | Borra todo lo generado (`data/processed/*`, `data/quantum_subset/*`, `results/metrics/*`, `results/figures/*` — conservando cada `README.md`) y limpia los outputs de los notebooks, para poder correr `run_all.ps1` desde cero cuantas veces se quiera. Nunca toca `data/raw/water_potability.csv` (es la descarga manual de Kaggle) ni el código fuente. |
| `check_notebook_errors.py` | Helper de Python usado por `run_notebooks.ps1`: revisa un `.ipynb` ya ejecutado y falla (exit code 1) si alguna celda quedó con un error, como chequeo independiente además del exit code de `nbconvert`. |

## Uso típico

Desde la raíz del repo (o con ruta completa, da igual):

```powershell
# Primera vez / después de clonar el repo
.\scripts\run_all.ps1

# Solo instalar/actualizar dependencias
.\scripts\setup.ps1

# Solo correr el pipeline de notebooks (venv ya existe)
.\scripts\run_notebooks.ps1

# Trabajar interactivamente en el venv
. .\scripts\activate.ps1

# Empezar de cero (pide confirmación)
.\scripts\reset.ps1

# Empezar de cero sin preguntar, y además borrar el venv
.\scripts\reset.ps1 -Force -RemoveVenv
```

Si Windows bloquea la ejecución de scripts (`.ps1 is not digitally signed` o
similar), correr una sola vez, o anteponer a cualquier llamada:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

## Qué NO hacen estos scripts

- No tocan `data/raw/water_potability.csv` (la descarga manual de Kaggle) bajo
  ninguna circunstancia.
- No tocan `src/`, `notebooks/*.ipynb` (el código fuente de las celdas),
  `Enunciado/`, `report/` ni `slides/`.
- No suben ni descargan nada de git/GitHub — son solo entorno local y
  ejecución del pipeline.

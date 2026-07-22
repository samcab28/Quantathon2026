"""
Lanzador del QSVM desde la raíz del proyecto.

Sirve para el botón "Run" de VS Code o `python run_qsvm.py`. Es equivalente
a `python -m qsvm.run` pero sin tener que acordarse del `-m`.

    python run_qsvm.py                              # corrida por defecto (Selene)
    python run_qsvm.py --backend nexus --device H1-1E
    python run_qsvm.py --n-train 16 --n-test 30 --shots 512

IMPORTANTE: usa el Python 3.12 (donde están instaladas las librerías), no el 3.14.
"""
from qsvm.run import main

if __name__ == "__main__":
    main()

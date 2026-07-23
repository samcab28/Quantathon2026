"""Lista los devices de Quantinuum disponibles para tu cuenta.

    python run_devices.py                 # backend pytket-quantinuum
    python run_devices.py --backend qnexus

Equivale a `python -m qsvm.list_devices`. Corre esto para saber qué nombre de
device usar con run_qsvm.py --device <NOMBRE>.
"""
from qsvm.list_devices import main

if __name__ == "__main__":
    main()

"""
Lista los devices de Quantinuum disponibles para TU cuenta.

El nombre correcto del emulador (p. ej. H2-1E, H2-1SC, H2-2E...) depende de a
qué te dé acceso tu cuenta. Corre esto una vez, mira los nombres y usa el que
aparezca con  --device  en run.py.

    python -m qsvm.list_devices                 # backend pytket-quantinuum
    python -m qsvm.list_devices --backend qnexus
    python run_devices.py                        # equivalente (desde la raíz)

La primera vez pide autenticación en Quantinuum.
"""
import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "qsvm"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def listar_pytket():
    from pytket.extensions.quantinuum import QuantinuumBackend
    print("Devices vía pytket-quantinuum (dispara login la 1a vez):\n")
    infos = QuantinuumBackend.available_devices()
    if not infos:
        print("  (ninguno — ¿la cuenta no tiene acceso o falló el login?)")
        return
    for info in infos:
        name = getattr(info, "device_name", None) or getattr(info, "name", "?")
        nq = getattr(info, "n_nodes", None)
        if nq is None:
            nq = getattr(info, "n_qubits", "?")
        print(f"  {str(name):18s}  (qubits: {nq})")
    print("\nUsa uno de esos nombres con:  python run_qsvm.py --device <NOMBRE>")


def listar_qnexus():
    import qnexus as qnx
    print("Devices vía qnexus (dispara login en el navegador la 1a vez):\n")
    qnx.login()
    try:
        devs = qnx.devices.get_all()
    except Exception as e:
        print(f"  No se pudo listar: {e}")
        return
    for d in devs:
        name = getattr(d, "device_name", None) or getattr(d, "name", str(d))
        print(f"  {name}")
    print("\nUsa uno de esos nombres con:  "
          "python run_qsvm.py --backend qnexus --device <NOMBRE>")


def main():
    ap = argparse.ArgumentParser(description="Lista devices de Quantinuum.")
    ap.add_argument("--backend", choices=["pytket-quantinuum", "qnexus"],
                    default="pytket-quantinuum")
    args = ap.parse_args()
    if args.backend == "qnexus":
        listar_qnexus()
    else:
        listar_pytket()


if __name__ == "__main__":
    main()

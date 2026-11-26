"""
Los archivos de models/ reproducen el score de validación en un proceso nuevo (T03 · HU-09).

Carga models/modelo.pt y models/estadisticas.npz con app/predict.predecir() (CPU), arma las
ventanas con la función `windows` del notebook de entrenamiento y evalúa con train/validacion.py.
Los valores de referencia son los que imprimió el entrenamiento en Kaggle (ver models/FICHA.md).

Se omite si faltan PyTorch, pandas, los archivos del modelo o la data oficial.

Se ejecuta con:  pytest -q -s tests/test_reproduce_score.py
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(RAIZ / "train"))

import predict  # noqa: E402
import validacion  # noqa: E402

NOTEBOOK = RAIZ / "train" / "t02_linea_base.ipynb"
FICHA = RAIZ / "models" / "FICHA.md"
RUTA_TRAIN = RAIZ / "data" / "train.npz"
TOLERANCIA = 1e-4
REFERENCIA = {"score_local": 0.3376796841621399, "score_oficial": 0.44092414598231033}


def sha_en_ficha(nombre):
    fila = next(f for f in FICHA.read_text(encoding="utf-8").splitlines() if f.startswith(f"| `{nombre}`"))
    return re.search(r"[0-9a-f]{64}", fila).group(0)


@pytest.mark.parametrize("ruta", [predict.RUTA_PESOS, predict.RUTA_ESTADISTICAS], ids=lambda r: r.name)
def test_archivo_es_el_de_la_ficha(ruta):
    if not ruta.exists():
        pytest.skip(f"falta {ruta.name}")
    assert hashlib.sha256(ruta.read_bytes()).hexdigest() == sha_en_ficha(ruta.name)


def ventanas_del_notebook():
    """La función `windows` tal cual está en el notebook, para no mantener otra copia."""
    pd = pytest.importorskip("pandas")
    for celda in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        codigo = "".join(celda["source"])
        if celda["cell_type"] == "code" and "def windows(" in codigo:
            funcion = next(n for n in ast.parse(codigo).body
                           if isinstance(n, ast.FunctionDef) and n.name == "windows")
            ns = {"np": np, "pd": pd}
            exec(ast.get_source_segment(codigo, funcion), ns)
            return ns["windows"]
    raise AssertionError("El notebook no define windows().")


def test_los_archivos_reproducen_el_score():
    pytest.importorskip("torch")
    faltan = [r.name for r in (predict.RUTA_PESOS, predict.RUTA_ESTADISTICAS, predict.RUTA_STATIC, RUTA_TRAIN)
              if not r.exists()]
    if faltan:
        pytest.skip(f"faltan {', '.join(faltan)}")

    datos = np.load(RUTA_TRAIN)
    X, Y, t = datos["X"], datos["Y"], datos["times"].astype(str)
    _, val = validacion.corte_temporal(t)
    ventanas = ventanas_del_notebook()(X, t)[val]
    res = validacion.validar(predict.predecir(ventanas, validacion.horas(t[val])), Y, t)

    print()
    for clave, ref in REFERENCIA.items():
        print(f"T03 · {clave}: {res[clave]:.8f} · referencia {ref:.8f} · diferencia {res[clave] - ref:+.2e}")
    print("T03 · RMSE:", res["rmse"])
    print("T03 · RMSE horas evaluadas:", res["rmse_horas_evaluadas"])
    for clave, ref in REFERENCIA.items():
        assert abs(res[clave] - ref) < TOLERANCIA, clave

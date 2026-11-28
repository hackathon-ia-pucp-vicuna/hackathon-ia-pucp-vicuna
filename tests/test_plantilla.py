"""La plantilla de experimentos (T20 · HU-09) lleva las copias exactas y no importa app/. Sin PyTorch."""
import ast
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "train" / "plantilla_experimento.ipynb"


def celdas():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("modulo", ["validacion.py", "ventanas.py"])
def test_lleva_la_copia_exacta_del_modulo(modulo):
    copias = [c for c in celdas() if c.startswith(f"# COPIA EXACTA de train/{modulo}")]
    assert len(copias) == 1
    assert copias[0].split("\n", 1)[1] == (RAIZ / "train" / modulo).read_text(encoding="utf-8").rstrip("\n")


def test_sintaxis_valida_sin_app_y_sin_salidas():
    for c in celdas():
        for nodo in ast.walk(ast.parse(c)):
            if isinstance(nodo, ast.Import):
                assert not any(a.name.split(".")[0] == "app" for a in nodo.names)
            if isinstance(nodo, ast.ImportFrom):
                assert (nodo.module or "").split(".")[0] != "app"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert not any(c.get("outputs") for c in nb["cells"])


def test_no_corre_en_modo_prueba_por_defecto():
    assert "QUICK = False " in celdas()[0]

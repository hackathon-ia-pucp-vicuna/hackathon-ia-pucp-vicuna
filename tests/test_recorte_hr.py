"""
La HR sale recortada a 0–100 % de predecir() (T10 · HU-09).

La red se anula (todos los pesos en cero), así que predecir() devuelve la climatología: se le pone
HR fuera de rango y se comprueba que solo la HR cambia, y que cambia igual que en el notebook.
Se omite si falta PyTorch.

Se ejecuta con:  pytest -q tests/test_recorte_hr.py
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))

import predict  # noqa: E402

NOTEBOOK = RAIZ / "train" / "t08_experimentos.ipynb"


def recortar_del_notebook():
    """La función `recortar_hr` tal cual está en el notebook, para no mantener otra copia."""
    for celda in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        codigo = "".join(celda["source"])
        if celda["cell_type"] == "code" and "def recortar_hr(" in codigo:
            funcion = next(n for n in ast.parse(codigo).body
                           if isinstance(n, ast.FunctionDef) and n.name == "recortar_hr")
            ns = {"np": np}
            exec(ast.get_source_segment(codigo, funcion), ns)
            return ns["recortar_hr"]
    raise AssertionError("El notebook no define recortar_hr().")


@pytest.fixture
def modelo_nulo(tmp_path, monkeypatch):
    """Red en cero y climatología con HR por debajo, dentro y por encima de 0–100. Devuelve `clim`."""
    torch = pytest.importorskip("torch")
    from test_paridad import escribir_manifiesto, static_sintetico

    st = static_sintetico()
    np.savez(tmp_path / "static.npz", **st)
    estado = {k: torch.zeros_like(v) for k, v in predict.construir_red(st).state_dict().items()}
    torch.save(estado, tmp_path / "modelo.pt")
    clim = np.empty((24, 70, 35, 4), np.float32)
    clim[..., :] = [70.0, 280.0, 150.0, -50.0]     # U y V fuera de 0–100 a propósito: no se tocan
    clim[:, :10, :, 0] = -30.0
    clim[:, 10:20, :, 0] = 130.0
    est = {"mu_x": np.zeros(4, np.float32), "sd_x": np.ones(4, np.float32), "clim": clim,
           "anom_sd": np.ones(4, np.float32)}
    np.savez(tmp_path / "estadisticas.npz", **est)
    # T49: predecir() carga lo que dice el manifiesto (variante starter, un miembro)
    ruta = escribir_manifiesto(tmp_path, "starter", [tmp_path / "modelo.pt"], est)
    monkeypatch.setattr(predict, "RUTA_STATIC", tmp_path / "static.npz")
    monkeypatch.setattr(predict, "RUTA_MODELOS", tmp_path)
    monkeypatch.setattr(predict, "RUTA_MANIFIESTO", ruta)
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()
    yield clim
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()


def test_predecir_recorta_la_hr_a_0_100(modelo_nulo):
    horas = np.arange(24)
    Y = predict.predecir(np.zeros((24, 3, 17, 13, 4), np.float32), horas, lote=5)
    assert Y.dtype == np.float32
    assert Y[..., 0].min() == 0.0 and Y[..., 0].max() == 100.0
    np.testing.assert_array_equal(Y[:, :10, :, 0], 0.0)
    np.testing.assert_array_equal(Y[:, 10:20, :, 0], 100.0)
    np.testing.assert_array_equal(Y[:, 20:, :, 0], 70.0)
    np.testing.assert_array_equal(Y[..., 1:], modelo_nulo[horas][..., 1:])


def test_predecir_recorta_igual_que_el_notebook(modelo_nulo):
    horas = np.arange(24)
    Y = predict.predecir(np.zeros((24, 3, 17, 13, 4), np.float32), horas)
    np.testing.assert_array_equal(Y, recortar_del_notebook()(modelo_nulo[horas]))

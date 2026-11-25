"""
Paridad entre la red del notebook de entrenamiento y la copia de app/predict.py (T12 · HU-01).

- Sintética (siempre): extrae `Net` del notebook, le pone la misma grilla, DEM y pesos aleatorios
  que a la copia de predict.py y compara la salida en 5 muestras (diferencia < 1e-5).
- Real (si existen models/modelo.pt y data/static.npz): lo mismo con los pesos y el relieve reales.

Se ejecuta con:  pytest -q tests/test_paridad.py -s   (-s muestra el tiempo de 24 h en CPU)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))

import predict  # noqa: E402

NOTEBOOK = RAIZ / "train" / "starter_notebook.ipynb"
TOLERANCIA = 1e-5


def static_sintetico(semilla=0):
    """Grillas con la misma geometría que la competencia: GFS 17×13 (fila 0 = norte), WRF 70×35 (fila 0 = sur)."""
    rng = np.random.default_rng(semilla)
    return {
        "gfs_lats": np.round(np.arange(-7.25, -11.25 - 1e-9, -0.25), 2),
        "gfs_lons": np.round(np.arange(-79.0, -76.0 + 1e-9, 0.25), 2),
        "wrf_lats": np.linspace(-10.4, -8.3, 70),
        "wrf_lons": np.linspace(-78.3, -77.2, 35),
        "dem": rng.uniform(1500, 6700, (70, 35)).astype(np.float32),
    }


def red_del_notebook(st):
    """Ejecuta la definición de `Net` tal cual está en el notebook, con GRID y DEM como en su celda."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "class Net" in "".join(c["source"]))
    clase = re.search(r"^class Net\(nn\.Module\):.*?(?=^\S)", codigo, re.S | re.M).group(0)

    glat, glon, wlat, wlon, dem = (st[k] for k in ("gfs_lats", "gfs_lons", "wrf_lats", "wrf_lons", "dem"))
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    ns = {
        "np": np, "torch": torch, "nn": nn, "F": F,
        "GRID": torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32),
        "DEM": torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32)[None, None],
    }
    exec(clase, ns)
    return ns["Net"]()


def muestras(n=5, semilla=1):
    rng = np.random.default_rng(semilla)
    x = torch.tensor(rng.standard_normal((n, 3, 17, 13, 4)), dtype=torch.float32)
    h = torch.tensor(rng.integers(0, 24, n), dtype=torch.float32)
    return x, h


def comparar(st, estado=None):
    torch.manual_seed(0)
    red_nb = red_del_notebook(st)
    if estado is None:
        with torch.no_grad():                         # pesos aleatorios, incluido el parámetro `static`
            for p in red_nb.parameters():
                p.normal_(0, 0.1)
        estado = red_nb.state_dict()
    else:
        red_nb.load_state_dict(estado)
    red_app = predict.construir_red(st)
    red_app.load_state_dict(estado)                  # mismo state_dict: mismas claves en ambas redes
    red_nb.eval(), red_app.eval()
    x, h = muestras()
    with torch.no_grad():
        a, b = red_nb(x, h).numpy(), red_app(x, h).numpy()
    return float(np.abs(a - b).max()), a.shape


def test_paridad_sintetica():
    dif, forma = comparar(static_sintetico())
    assert forma == (5, 70, 35, 4)
    assert dif < TOLERANCIA, f"diferencia máxima {dif:.2e}"


@pytest.mark.skipif(not (predict.RUTA_PESOS.exists() and predict.RUTA_STATIC.exists()),
                    reason="faltan models/modelo.pt o data/static.npz (T03)")
def test_paridad_con_pesos_reales():
    estado = torch.load(predict.RUTA_PESOS, map_location="cpu", weights_only=True)
    dif, _ = comparar(predict.cargar_static(), estado)
    assert dif < TOLERANCIA, f"diferencia máxima {dif:.2e}"


@pytest.fixture
def modelo_temporal(tmp_path, monkeypatch):
    """Archivos del modelo con pesos aleatorios y estadísticas plausibles, para probar predecir() de punta a punta."""
    st = static_sintetico()
    np.savez(tmp_path / "static.npz", **st)
    torch.manual_seed(0)
    torch.save(predict.construir_red(st).state_dict(), tmp_path / "modelo.pt")
    clim = np.zeros((24, 70, 35, 4), np.float32)
    clim[..., :] = [70.0, 280.0, 1.0, -1.0]
    np.savez(tmp_path / "estadisticas.npz",
             mu_x=np.array([70.0, 285.0, 0.5, -0.5], np.float32), sd_x=np.array([20.0, 6.0, 3.0, 3.0], np.float32),
             clim=clim, anom_sd=np.array([15.0, 2.0, 2.5, 2.0], np.float32))
    monkeypatch.setattr(predict, "RUTA_STATIC", tmp_path / "static.npz")
    monkeypatch.setattr(predict, "RUTA_PESOS", tmp_path / "modelo.pt")
    monkeypatch.setattr(predict, "RUTA_ESTADISTICAS", tmp_path / "estadisticas.npz")
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()
    yield
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()


def test_predecir_devuelve_70x35x4_en_unidades_fisicas(modelo_temporal):
    rng = np.random.default_rng(2)
    Xw = np.empty((24, 3, 17, 13, 4), np.float32)
    Xw[...] = [70.0, 285.0, 0.5, -0.5]
    Xw += rng.normal(0, 1, Xw.shape).astype(np.float32)
    Y = predict.predecir(Xw, np.arange(24))
    assert Y.shape == (24, 70, 35, 4)
    assert np.isfinite(Y).all()
    assert 200 < Y[..., 1].mean() < 330                # T en K, no normalizada
    assert predict.predecir(Xw[:0], []).shape == (0, 70, 35, 4)


def test_predecir_valida_la_entrada(modelo_temporal):
    with pytest.raises(ValueError):
        predict.predecir(np.zeros((2, 17, 13, 4)), [0, 1])
    with pytest.raises(ValueError):
        predict.predecir(np.zeros((2, 3, 17, 13, 4)), [0])


def test_tiempo_24_horas_en_cpu(modelo_temporal):
    seg = predict.medir_24h()
    print(f"\npredecir() en CPU · 24 horas: {seg:.3f} s (pesos aleatorios, misma arquitectura)")
    assert seg < 10


def test_falta_archivo_da_error_claro(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "RUTA_PESOS", tmp_path / "no_existe.pt")
    monkeypatch.setattr(predict, "RUTA_STATIC", tmp_path / "no_existe.npz")
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()
    with pytest.raises(predict.ModeloNoDisponible, match="No se encontró"):
        predict.predecir(np.zeros((1, 3, 17, 13, 4)), [0])
    predict.cargar_static.cache_clear()

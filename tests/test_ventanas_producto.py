"""
La app y el aviso de Telegram solo arman ventanas con horas consecutivas (T35 · HU-01).

Sin red ni pesos. Con las 04 y 05 UTC ausentes, las horas 04–07 de la noche no se predicen (les falta
t, t-1 o t-2) y la app y el aviso lo advierten, en vez de rellenar la ventana con horas anteriores.

Se ejecuta con:  pytest -q tests/test_ventanas_producto.py
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aviso  # noqa: E402
import entrada_gfs  # noqa: E402
import gfs_forecast  # noqa: E402
from test_streamlit_app import NOCHE, app, pronostico_sintetico, sin_trazas  # noqa: E402,F401

SIN_PREDICCION = [f"{NOCHE}{h:02d}" for h in (4, 5, 6, 7)]


def con_hueco():
    X, t = pronostico_sintetico()
    quedan = ~np.isin(t, [f"{NOCHE}04", f"{NOCHE}05"])
    X = X[quedan].copy()
    t = t[quedan]
    X[..., 1] = 270.0 + np.arange(len(t), dtype=np.float32)[:, None, None]    # cada hora, una T distinta
    return X, t


def test_ninguna_ventana_cruza_un_hueco():
    X, t = con_hueco()
    Xw, t_noche, X_t, faltan = entrada_gfs.ventanas_de_la_noche(X, t, NOCHE)
    assert faltan == SIN_PREDICCION
    assert not set(t_noche) & set(SIN_PREDICCION) and len(t_noche) == 9
    pos = {h: i for i, h in enumerate(t)}
    for ventana, gfs, hora in zip(Xw, X_t, t_noche):
        i = pos[hora]
        np.testing.assert_array_equal(ventana, X[i - 2:i + 1])
        np.testing.assert_array_equal(gfs, X[i])


def test_advertencia_en_hora_de_lima():
    texto = entrada_gfs.advertencia_horas_faltantes(SIN_PREDICCION)
    assert "23:00, 00:00, 01:00, 02:00" in texto
    assert entrada_gfs.advertencia_horas_faltantes([]) is None


def test_la_app_advierte_las_horas_que_faltan(app):  # noqa: F811
    at, pred = app
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", return_value=con_hueco()):
        at.run()
    sin_trazas(at)
    assert any("Faltan datos del pronóstico" in w.value for w in at.warning)
    assert len(pred.call_args[0][0]) == 9


def test_el_aviso_de_telegram_advierte_las_horas_que_faltan(app):  # noqa: F811
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", return_value=con_hueco()):
        avisos = aviso.calcular_avisos(NOCHE, "papa", ["carhuaz"])
    assert "Faltan datos del pronóstico" in avisos[0]["mensaje"]

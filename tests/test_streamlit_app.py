"""
App de Streamlit de punta a punta, sin red (T17 · HU-02, HU-04, HU-07).

Se ejecuta app/web/streamlit_app.py con streamlit.testing (AppTest), con el pronóstico de Open-Meteo,
el relieve y predecir() sustituidos por datos sintéticos. Comprueba:
- se elige zona y cultivo y se ven nivel, hora de inicio y duración;
- se ven la tabla por hora y la T mínima del GFS crudo y del modelo, con su diferencia;
- cambiar de zona o de cultivo no vuelve a correr el modelo (caché);
- los errores salen en español y sin trazas técnicas (sin red, sin modelo, error inesperado, noche no cubierta).
Con la data oficial y los pesos, además mide la carga con caché usando el modelo real (< 5 s).

Se ejecuta con:  pytest -q -s tests/test_streamlit_app.py
"""
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gfs_forecast  # noqa: E402
import predict  # noqa: E402
import streamlit as st  # noqa: E402
from frost import comparar, rules  # noqa: E402
from test_paridad import static_sintetico  # noqa: E402

APP = str(RAIZ / "app" / "web" / "streamlit_app.py")
NOCHE = "20260701"                               # noche fija: horas 00–12 UTC del 1 jul
K = 273.15
T_GFS_C = 5.0                                    # GFS crudo constante: 5 °C en toda la grilla


def pronostico_sintetico():
    """Horario del 30 jun 12 UTC al 1 jul 23 UTC (con t-2 y t-1 para la noche), T del GFS a 5 °C."""
    times = np.array([f"202606{30}{h:02d}" for h in range(12, 24)] + [f"{NOCHE}{h:02d}" for h in range(24)])
    X = np.empty((len(times), 17, 13, 4), np.float32)
    X[...] = [60.0, T_GFS_C + K, 1.0, -1.0]
    return X, times


def predecir_sintetico(Xw, horas):
    """El 'modelo' enfría la noche: T = 2 °C − 1 °C por hora UTC (00 UTC: 2 °C … 12 UTC: −10 °C)."""
    Y = np.empty((len(Xw), 70, 35, 4), np.float32)
    Y[...] = [80.0, 0.0, 1.0, -1.0]
    Y[..., 1] = (2.0 - np.asarray(horas, np.float32))[:, None, None] + K
    return Y


@pytest.fixture
def app(tmp_path, monkeypatch):
    """AppTest con red, relieve y modelo sintéticos. Devuelve (AppTest, mock de predecir)."""
    monkeypatch.setenv("HELADA_CACHE_GFS", str(tmp_path / "gfs_cache.npz"))   # sin caché guardada
    st.cache_data.clear()
    pred = mock.Mock(side_effect=predecir_sintetico)
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", return_value=pronostico_sintetico()), \
            mock.patch.object(predict, "cargar_static", return_value=static_sintetico()), \
            mock.patch.object(predict, "predecir", pred), \
            mock.patch.object(rules, "noche_objetivo", return_value=NOCHE):
        yield AppTest.from_file(APP, default_timeout=60), pred
    st.cache_data.clear()


def sin_trazas(at):
    assert not at.exception, [e.value for e in at.exception]


def metricas(at):
    return {m.label: m.value for m in at.metric}


def test_zona_y_cultivo_muestran_nivel_inicio_y_duracion(app):
    at, _ = app
    at.run()
    sin_trazas(at)
    assert [s.label for s in at.selectbox] == ["Localidad", "Cultivo"]
    assert at.selectbox[1].options == ["Papa"]
    m = metricas(at)
    # papa: helada_c −1 °C. T del modelo 2 − h: baja de −1 °C a las 03 UTC (22:00 Lima) hasta las 12 UTC
    assert m["Nivel de riesgo"] == "Helada probable"
    assert m["Inicio bajo -1 °C"] == "22:00"
    assert m["Duración bajo -1 °C"] == "10 h"
    assert "Carhuaz · Papa" in at.subheader[0].value

    at.selectbox[0].set_value("huaraz").run()
    sin_trazas(at)
    assert "Huaraz · Papa" in at.subheader[0].value


def test_tabla_por_hora_y_tmin_del_gfs_crudo_y_del_modelo(app):
    at, _ = app
    at.run()
    sin_trazas(at)
    m = metricas(at)
    assert m["Modelo (3 km)"] == "-10.0 °C"
    assert m["GFS crudo (27 km)"] == "5.0 °C"
    assert m["Diferencia"] == "-15.0 °C"
    tabla = at.dataframe[0].value
    assert list(tabla.columns) == ["Hora (Lima)", "Modelo (°C)", "GFS crudo (°C)", "Diferencia (°C)", "Bajo -1 °C"]
    assert len(tabla) == 13                                       # 00–12 UTC = 19:00–07:00 Lima
    assert tabla["Hora (Lima)"].iloc[0] == "30/06 19:00"
    assert (tabla["GFS crudo (°C)"] == T_GFS_C).all()
    assert tabla["Bajo -1 °C"].tolist().count("sí") == 10


def test_cambiar_zona_o_cultivo_no_vuelve_a_correr_el_modelo(app):
    at, pred = app
    at.run()
    at.selectbox[0].set_value("yungay").run()
    at.selectbox[0].set_value("huaraz").run()
    at.selectbox[1].set_value("papa").run()
    sin_trazas(at)
    assert pred.call_count == 1


def test_sin_red_y_sin_caché_da_mensaje_en_español(app):
    at, _ = app
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", side_effect=ConnectionError("Max retries exceeded")):
        at.run()
    sin_trazas(at)
    assert at.error[0].value.startswith("No se pudo obtener el pronóstico: revisa tu conexión")
    assert "Max retries" not in at.error[0].value


def test_sin_red_con_caché_usa_el_último_pronóstico(app, tmp_path):
    at, _ = app
    X, times = pronostico_sintetico()
    np.savez(tmp_path / "gfs_cache.npz", X=X, times=times)
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", side_effect=ConnectionError("sin red")):
        at.run()
    sin_trazas(at)
    assert "último pronóstico guardado" in at.warning[0].value
    assert metricas(at)["Nivel de riesgo"] == "Helada probable"


def test_sin_modelo_da_mensaje_en_español(app):
    at, _ = app
    falta = predict.ModeloNoDisponible("No se encontró el manifiesto del modelo (modelo.json).")
    with mock.patch.object(predict, "predecir", side_effect=falta):
        at.run()
    sin_trazas(at)
    assert at.error[0].value.startswith("El modelo no está disponible. No se encontró el manifiesto")


def test_error_inesperado_no_muestra_trazas(app):
    at, _ = app
    with mock.patch.object(predict, "predecir", side_effect=ValueError("operands could not be broadcast")):
        at.run()
    sin_trazas(at)
    assert at.error[0].value.startswith("Ocurrió un problema al preparar el aviso")
    assert "broadcast" not in at.error[0].value
    assert not at.metric


def test_noche_no_cubierta_da_mensaje_en_español(app):
    at, _ = app
    with mock.patch.object(rules, "noche_objetivo", return_value="20260705"):
        at.run()
    sin_trazas(at)
    assert at.error[0].value == "El pronóstico disponible no cubre la noche de hoy. Intenta de nuevo más tarde."


def test_la_app_no_muestra_trazas_por_configuración():
    config = (RAIZ / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "showErrorDetails = false" in config


def test_interpolación_igual_a_la_de_t36():
    sys.path.insert(0, str(RAIZ / "scripts"))
    import evaluar_tmin_nocturna as t36

    st_ = static_sintetico()
    X = np.random.default_rng(0).normal(280, 5, (3, 17, 13)).astype(np.float64)
    np.testing.assert_array_equal(comparar.interpolar_gfs(X, st_), t36.interpolar_gfs(X, st_))


DATA = RAIZ / "data"


@pytest.mark.skipif(not ((DATA / "test.npz").exists() and (DATA / "static.npz").exists()
                         and predict.RUTA_MANIFIESTO.exists() and (RAIZ / "models" / "estadisticas_ensamble.npz").exists()),
                    reason="sin la data oficial, los pesos o models/estadisticas_ensamble.npz")
def test_con_caché_carga_en_menos_de_5_segundos(tmp_path, monkeypatch):
    """Modelo real (los miembros del manifiesto). El GFS sale de test.npz (sin red); se mide la recarga con la caché llena."""
    te = np.load(DATA / "test.npz")
    X, times = te["X"], te["times"].astype(str)
    noche = "20240710"
    monkeypatch.setenv("HELADA_CACHE_GFS", str(tmp_path / "gfs_cache.npz"))
    st.cache_data.clear()
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", return_value=(X, times)), \
            mock.patch.object(rules, "noche_objetivo", return_value=noche):
        at = AppTest.from_file(APP, default_timeout=300)
        t0 = time.perf_counter()
        at.run()
        primera = time.perf_counter() - t0
        sin_trazas(at)
        tiempos = []
        for zona in ("huaraz", "yungay", "carhuaz"):
            t0 = time.perf_counter()
            at.selectbox[0].set_value(zona).run()
            tiempos.append(time.perf_counter() - t0)
            sin_trazas(at)
    st.cache_data.clear()
    print(f"\nT17 · primera carga {primera:.2f} s · con caché {', '.join(f'{t:.2f}' for t in tiempos)} s · "
          f"{metricas(at)}")
    assert max(tiempos) < 5

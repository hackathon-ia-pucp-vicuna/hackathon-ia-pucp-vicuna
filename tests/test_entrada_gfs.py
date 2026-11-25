"""
Pruebas del adaptador Open-Meteo → entrada del modelo (T35 · HU-01). Sin red.

La fixture tests/fixtures/open_meteo_8h.json es una descarga real de Open-Meteo (GFS, 221 puntos,
8 horas: 2 pasadas + 6 de pronóstico) guardada una vez; aquí se reemplaza requests.get por ella.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import gfs_forecast  # noqa: E402
from entrada_gfs import EntradaInvalida, obtener_entrada, validar_grilla, ventanas_por_hora  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "open_meteo_8h.json"


class _Respuesta:
    status_code = 200

    def __init__(self, datos):
        self._datos = datos

    def json(self):
        return self._datos


@pytest.fixture
def sin_red(monkeypatch):
    """Reemplaza requests.get por la fixture; cuenta las llamadas. Permite borrar una hora de un punto."""
    puntos = json.loads(FIXTURE.read_text(encoding="utf-8"))
    estado = {"llamadas": 0, "puntos": puntos}

    def falso_get(url, params, timeout):
        estado["llamadas"] += 1
        lats = params["latitude"].split(",")
        lons = params["longitude"].split(",")
        return _Respuesta([estado["puntos"][f"{float(a)},{float(b)}"] for a, b in zip(lats, lons)])

    monkeypatch.setattr(gfs_forecast.requests, "get", falso_get)
    return estado


def test_ventanas_con_forma_orden_y_unidades_del_entrenamiento(sin_red, tmp_path):
    e = obtener_entrada(forecast_hours=6, cache_path=tmp_path / "c.npz")
    Xw, times = e["Xw"], e["times"]
    assert Xw.shape == (6, 3, 17, 13, 4) and Xw.dtype == np.float32
    assert len(e["descartadas"]) == 2                     # las 2 horas pasadas no tienen t-2 propio
    assert e["horas_utc"].tolist() == [int(t[8:10]) for t in times]
    # La ventana i es [t-2, t-1, t]: su último paso es el primero de la ventana i+2
    np.testing.assert_array_equal(Xw[0, 2], Xw[2, 0])
    # Unidades: RH en %, T en K, viento en m/s
    assert 0 <= Xw[..., 0].min() and Xw[..., 0].max() <= 100
    assert 200 < Xw[..., 1].mean() < 310


def test_fila_cero_es_el_norte_y_conversion_de_viento(sin_red, tmp_path):
    e = obtener_entrada(forecast_hours=6, cache_path=tmp_path / "c.npz")
    p_norte = sin_red["puntos"][f"{gfs_forecast.LATS[0]},{gfs_forecast.LONS[0]}"]["hourly"]
    p_sur = sin_red["puntos"][f"{gfs_forecast.LATS[-1]},{gfs_forecast.LONS[-1]}"]["hourly"]
    # Ventana 0 → hora t = índice 2 de la descarga
    assert e["Xw"][0, 2, 0, 0, 1] == pytest.approx(p_norte["temperature_2m"][2] + 273.15, abs=1e-3)
    assert e["Xw"][0, 2, -1, -1, 0] == pytest.approx(p_sur["relative_humidity_2m"][2], abs=1e-3)
    ws, wd = p_norte["wind_speed_10m"][2], np.deg2rad(p_norte["wind_direction_10m"][2])
    assert e["Xw"][0, 2, 0, 0, 2] == pytest.approx(-ws * np.sin(wd), abs=1e-3)   # U
    assert e["Xw"][0, 2, 0, 0, 3] == pytest.approx(-ws * np.cos(wd), abs=1e-3)   # V


def test_hora_faltante_descarta_las_ventanas_que_la_cruzan(sin_red, tmp_path):
    # Open-Meteo devuelve una hora vacía en un punto → gfs_forecast descarta esa hora para toda la grilla
    clave = next(iter(sin_red["puntos"]))
    hueco = sin_red["puntos"][clave]["hourly"]["time"][4]
    sin_red["puntos"][clave]["hourly"]["temperature_2m"][4] = None
    e = obtener_entrada(forecast_hours=6, cache_path=tmp_path / "c.npz")
    h = hueco.replace("-", "").replace("T", "")[:10]
    assert h not in e["times"]
    # Sin la hora 4 se pierden la 4, la 5 (le falta t-1) y la 6 (le falta t-2)
    assert len(e["times"]) == 3
    assert len(e["descartadas"]) == 2 + 2


def test_cache_con_ttl(sin_red, tmp_path):
    cache = tmp_path / "gfs_cache.npz"
    obtener_entrada(forecast_hours=6, cache_path=cache, ttl_h=3)
    llamadas = sin_red["llamadas"]
    assert cache.exists() and llamadas > 0
    obtener_entrada(forecast_hours=6, cache_path=cache, ttl_h=3)
    assert sin_red["llamadas"] == llamadas                # caché vigente: sin red
    viejo = time.time() - 4 * 3600
    os.utime(cache, (viejo, viejo))
    obtener_entrada(forecast_hours=6, cache_path=cache, ttl_h=3)
    assert sin_red["llamadas"] > llamadas                 # caché vencida: vuelve a descargar


def test_ventanas_sinteticas_con_huecos_y_desorden():
    times = np.array(["2024061600", "2024061601", "2024061602", "2024061604", "2024061605", "2024061606"])
    X = np.zeros((6, 17, 13, 4), np.float32)
    X[..., 1] = 280.0 + np.arange(6)[:, None, None]       # T distinta por hora para rastrearla
    orden = [3, 0, 5, 1, 4, 2]
    Xw, t_ok, horas, desc = ventanas_por_hora(X[orden], times[orden])
    assert t_ok.tolist() == ["2024061602", "2024061606"]
    assert desc == ["2024061600", "2024061601", "2024061604", "2024061605"]
    assert Xw[0, :, 0, 0, 1].tolist() == [280.0, 281.0, 282.0]
    assert horas.tolist() == [2, 6]


def test_sin_ventanas_posibles_devuelve_vacio():
    X = np.full((2, 17, 13, 4), [60.0, 280.0, 1.0, -1.0], np.float32)
    Xw, t_ok, horas, desc = ventanas_por_hora(X, ["2024061600", "2024061601"])
    assert Xw.shape == (0, 3, 17, 13, 4) and len(t_ok) == 0 and len(desc) == 2


@pytest.mark.parametrize("cambio,mensaje", [
    (lambda X: X[:, :13], "17, 13, 4"),
    (lambda X: X - 273.15 * np.array([0, 1, 0, 0], np.float32), "unidades"),   # T en °C
    (lambda X: np.where(X == X.max(), np.nan, X), "NaN"),
])
def test_entrada_invalida(cambio, mensaje):
    X = np.full((3, 17, 13, 4), [60.0, 280.0, 1.0, -1.0], np.float32)
    with pytest.raises(EntradaInvalida, match=mensaje):
        validar_grilla(cambio(X), ["2024061600", "2024061601", "2024061602"])


def test_past_hours_minimo():
    with pytest.raises(ValueError):
        obtener_entrada(past_hours=1, descargar=lambda **k: None)

"""
Modo sin red y manejo de fallos (T37 · HU-11). Sin red: Open-Meteo y Telegram se simulan.

- Si falla Open-Meteo, se usa el último pronóstico guardado, con la fecha y hora en que se descargó y
  una advertencia; sin pronóstico guardado, un mensaje en español. Vale para la app y para el aviso.
- Si falla Telegram, el script no se cae, registra el error (sin el token) y la app sigue funcionando.

Se ejecuta con:  pytest -q tests/test_fallos_red.py
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import requests

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(RAIZ / "app" / "notify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aviso  # noqa: E402
import gfs_forecast  # noqa: E402
import predict  # noqa: E402
import pronostico  # noqa: E402
import telegram  # noqa: E402
from test_paridad import static_sintetico  # noqa: E402
from test_streamlit_app import app, pronostico_sintetico, predecir_sintetico  # noqa: E402,F401
from test_telegram import TOKEN, aviso_de  # noqa: E402

# 26 nov 2026, 18:00 de Lima = 23:00 UTC
GUARDADO = datetime(2026, 11, 26, 23, 0, tzinfo=timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def sin_env_real(monkeypatch, tmp_path):
    """Las pruebas nunca leen el .env real de quien las corre (telegram.main() carga .env)."""
    monkeypatch.setattr(telegram, "RUTA_ENV", tmp_path / "no_hay.env")


def guardar_cache(ruta):
    X, times = pronostico_sintetico()
    np.savez(ruta, X=X, times=times)
    os.utime(ruta, (GUARDADO, GUARDADO))
    return ruta


def test_open_meteo_falla_y_se_usa_la_caché_con_su_hora(tmp_path):
    ruta = guardar_cache(tmp_path / "gfs_cache.npz")
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", side_effect=requests.ConnectionError("sin red")):
        X, times, adv = pronostico.obtener_pronostico(ruta, 3)
    assert X.shape[1:] == (17, 13, 4) and len(times) == len(X)
    assert "descargado el 26/11 a las 18:00 (hora de Lima)" in adv
    assert "Puede estar desactualizado" in adv


def test_open_meteo_falla_sin_caché_da_mensaje_en_español(tmp_path):
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", side_effect=RuntimeError("HTTP 429")), \
            pytest.raises(pronostico.SinPronostico, match="No se pudo obtener el pronóstico"):
        pronostico.obtener_pronostico(tmp_path / "no_hay.npz", 3)


def test_caché_dañada_se_trata_como_si_no_hubiera(tmp_path):
    ruta = tmp_path / "gfs_cache.npz"
    ruta.write_bytes(b"no es un npz")
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", side_effect=requests.Timeout("lento")), \
            pytest.raises(pronostico.SinPronostico):
        pronostico.obtener_pronostico(ruta, 3)


def test_con_red_no_hay_advertencia(tmp_path):
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", return_value=pronostico_sintetico()):
        assert pronostico.obtener_pronostico(tmp_path / "c.npz", 3)[2] is None


def test_la_app_muestra_la_caché_con_su_hora_si_falla_open_meteo(app, tmp_path):  # noqa: F811
    at, _ = app
    guardar_cache(tmp_path / "gfs_cache.npz")                  # la fixture apunta HELADA_CACHE_GFS aquí
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", side_effect=requests.ConnectionError("sin red")):
        at.run()
    assert not at.exception
    assert "descargado el 26/11 a las 18:00 (hora de Lima)" in at.warning[0].value
    assert any(m.label == "Nivel de riesgo" for m in at.metric)        # el aviso se muestra igual


def test_el_aviso_de_telegram_usa_la_caché_y_lo_advierte(tmp_path, monkeypatch):
    monkeypatch.setattr(aviso, "CACHE_GFS", guardar_cache(tmp_path / "gfs_cache.npz"))
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", side_effect=requests.ConnectionError("sin red")), \
            mock.patch.object(predict, "cargar_static", return_value=static_sintetico()), \
            mock.patch.object(predict, "predecir", side_effect=predecir_sintetico):
        avisos = aviso.calcular_avisos("20260701")
    assert all(a["advertencia"] for a in avisos)
    texto = telegram.armar_mensaje(avisos[0])
    assert telegram.ADVERTENCIA in texto and len(texto) <= 600


def test_mensaje_con_advertencia_sigue_en_600_caracteres():
    a = aviso_de()
    a["advertencia"] = "x"
    a["mensaje"] = "Texto largo. " * 30
    texto = telegram.armar_mensaje(a)
    assert len(texto) <= 600 and telegram.ADVERTENCIA in texto        # la advertencia no se quita


def test_telegram_falla_el_script_no_se_cae_y_registra_el_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@canal")
    log = tmp_path / "errores.log"
    monkeypatch.setenv("HELADA_LOG_TELEGRAM", str(log))
    monkeypatch.setattr(telegram, "RUTA_REGISTRO", tmp_path / "r.sqlite")
    error = requests.ConnectionError(f"HTTPSConnectionPool /bot{TOKEN}/sendMessage")
    with mock.patch.object(aviso, "calcular_avisos", return_value=[aviso_de(), aviso_de(clave="huaraz", zona="Huaraz")]), \
            mock.patch.object(requests, "post", side_effect=error):
        assert telegram.main([]) == 1                                  # sale con código 1, sin excepción
    lineas = log.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 2
    assert all("No se pudo conectar con Telegram" in x for x in lineas)
    assert TOKEN not in log.read_text(encoding="utf-8")
    assert datetime.fromisoformat(lineas[0].split(" · ")[0])


def test_si_falla_el_cálculo_también_se_registra(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@canal")
    log = tmp_path / "errores.log"
    monkeypatch.setenv("HELADA_LOG_TELEGRAM", str(log))
    with mock.patch.object(aviso, "calcular_avisos", side_effect=pronostico.SinPronostico(pronostico.MSJ_SIN_PRONOSTICO)):
        assert telegram.main([]) == 1
    assert "SinPronostico" in log.read_text(encoding="utf-8")


def test_log_que_no_se_puede_escribir_no_rompe(tmp_path):
    telegram.registrar_error("prueba", ruta=tmp_path / "no_existe" / "x.log")   # no lanza


def test_la_app_no_depende_de_telegram(app):  # noqa: F811
    at, _ = app
    with mock.patch.object(requests, "post", side_effect=requests.ConnectionError("Telegram caído")):
        at.run()
    assert not at.exception
    assert [e.value.split(" · ")[0] for e in at.error] == ["**Helada probable**"]   # solo el aviso, en rojo
    assert any(m.label == "Nivel de riesgo" for m in at.metric)
    fuente = (RAIZ / "app" / "web" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "telegram" not in fuente.lower()

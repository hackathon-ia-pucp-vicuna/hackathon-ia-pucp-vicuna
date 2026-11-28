"""
Aviso por el canal de Telegram (T18 · HU-05). Sin red: Telegram, Open-Meteo y el modelo se sustituyen.

- El mensaje tiene como máximo 600 caracteres, en los tres niveles y en el peor caso.
- No duplica el aviso de la misma zona y noche (registro en SQLite); un envío fallido no se registra.
- El token y el canal se leen de variables de entorno; el modo de prueba no envía, no pide token ni registra.
- Los errores nunca muestran el token.

Se ejecuta con:  pytest -q tests/test_telegram.py
"""
import sys
from datetime import datetime
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
import telegram  # noqa: E402
from frost.advice import mensaje  # noqa: E402
from frost.rules import LIMA, umbral_de_cultivo  # noqa: E402

TOKEN = "123456:ABC-token-de-prueba"
CANAL = "@canal_de_prueba"


def aviso_de(nivel="helada probable", zona="Carhuaz", clave="carhuaz", noche="20261127", tmin=-3.4):
    hora = lambda h: datetime(2026, 11, 26 if h >= 19 else 27, h, 0, tzinfo=LIMA)  # noqa: E731
    r = {"nivel": nivel, "tmin_c": tmin, "hora_tmin": hora(5),
         "inicio": hora(1) if nivel == "helada probable" else None,
         "duracion_h": 4.0 if nivel == "helada probable" else 0.0}
    return {"clave": clave, "zona": zona, "noche": noche, "cultivo": "Papa", "umbral": umbral_de_cultivo("papa"),
            "resultado": r, "mensaje": mensaje(nivel, r, zona), "inicio_noche": hora(19)}


def respuesta(ok=True, message_id=7, descripcion=None, status=200):
    resp = mock.Mock(status_code=status)
    resp.json.return_value = ({"ok": True, "result": {"message_id": message_id}} if ok
                              else {"ok": False, "description": descripcion})
    return resp


@pytest.fixture(autouse=True)
def sin_env_real(monkeypatch, tmp_path):
    """Las pruebas nunca leen el .env real de quien las corre."""
    monkeypatch.setattr(telegram, "RUTA_ENV", tmp_path / "no_hay.env")


@pytest.fixture
def entorno(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CANAL)


@pytest.mark.parametrize("nivel", ["sin riesgo", "vigilancia", "helada probable"])
def test_mensaje_de_600_caracteres_como_maximo(nivel):
    texto = telegram.armar_mensaje(aviso_de(nivel))
    assert len(texto) <= 600
    assert "Carhuaz" in texto
    assert "lluvia" in texto                         # declara los límites del modelo


def test_mensaje_de_helada_trae_inicio_duracion_y_minima():
    texto = telegram.armar_mensaje(aviso_de())
    assert "Helada probable" in texto
    assert "noche del 26/11" in texto
    assert "Mínima: -3 °C hacia las 05:00" in texto
    assert "desde las 01:00, unas 4 h" in texto


def test_mensaje_largo_se_recorta_a_600():
    a = aviso_de(zona="Z" * 400)
    a["mensaje"] = "Texto muy largo. " * 40
    assert len(telegram.armar_mensaje(a)) <= 600


def test_envia_un_mensaje_por_zona_al_canal(entorno, tmp_path):
    avisos = [aviso_de(clave=c, zona=c.capitalize()) for c in ("carhuaz", "huaraz", "yungay")]
    with mock.patch.object(requests, "post", return_value=respuesta()) as post:
        res = telegram.avisar(avisos, ruta_registro=tmp_path / "r.sqlite", salida=lambda s: None)
    assert res["enviados"] == ["carhuaz", "huaraz", "yungay"]
    assert post.call_count == 3
    url, kw = post.call_args.args[0], post.call_args.kwargs
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert kw["json"]["chat_id"] == CANAL and len(kw["json"]["text"]) <= 600


def test_no_duplica_la_misma_zona_y_noche(entorno, tmp_path):
    registro = tmp_path / "r.sqlite"
    with mock.patch.object(requests, "post", return_value=respuesta()) as post:
        telegram.avisar([aviso_de()], ruta_registro=registro, salida=lambda s: None)
        res = telegram.avisar([aviso_de(), aviso_de(clave="huaraz", zona="Huaraz")], ruta_registro=registro,
                              salida=lambda s: None)
        otra_noche = telegram.avisar([aviso_de(noche="20261128")], ruta_registro=registro, salida=lambda s: None)
    assert res == {"enviados": ["huaraz"], "omitidos": ["carhuaz"], "fallidos": []}
    assert otra_noche["enviados"] == ["carhuaz"]       # otra noche sí se envía
    assert post.call_count == 3


def test_envio_fallido_no_se_registra_y_no_muestra_el_token(entorno, tmp_path):
    registro, lineas = tmp_path / "r.sqlite", []
    fallo = respuesta(ok=False, descripcion=f"Unauthorized: bot{TOKEN} invalid", status=401)
    with mock.patch.object(requests, "post", return_value=fallo):
        res = telegram.avisar([aviso_de()], ruta_registro=registro, salida=lineas.append)
    assert res["fallidos"] == ["carhuaz"]
    assert TOKEN not in " ".join(lineas) and "***" in " ".join(lineas)
    with mock.patch.object(requests, "post", return_value=respuesta()):
        assert telegram.avisar([aviso_de()], ruta_registro=registro, salida=lambda s: None)["enviados"] == ["carhuaz"]


def test_sin_conexion_el_error_no_muestra_el_token(entorno, tmp_path):
    lineas = []
    error = requests.ConnectionError(f"HTTPSConnectionPool: /bot{TOKEN}/sendMessage")
    with mock.patch.object(requests, "post", side_effect=error):
        res = telegram.avisar([aviso_de()], ruta_registro=tmp_path / "r.sqlite", salida=lineas.append)
    assert res["fallidos"] == ["carhuaz"]
    assert "No se pudo conectar con Telegram" in lineas[0] and TOKEN not in lineas[0]


def test_modo_de_prueba_no_envia_no_pide_token_ni_registra(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    lineas = []
    with mock.patch.object(requests, "post") as post:
        res = telegram.avisar([aviso_de()], prueba=True, ruta_registro=tmp_path / "r.sqlite", salida=lineas.append)
    post.assert_not_called()
    assert not (tmp_path / "r.sqlite").exists()
    assert res == {"enviados": [], "omitidos": [], "fallidos": []}
    assert "prueba, no se envía" in lineas[0] and "Helada Andina · Carhuaz" in lineas[0]


def test_el_token_se_lee_de_variables_de_entorno(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CANAL)
    with pytest.raises(telegram.ConfiguracionFaltante, match="TELEGRAM_BOT_TOKEN"):
        telegram.configuracion()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    assert telegram.configuracion() == (TOKEN, CANAL)
    fuente = (RAIZ / "app" / "notify" / "telegram.py").read_text(encoding="utf-8")
    assert "os.environ.get(\"TELEGRAM_BOT_TOKEN\"" in fuente


def test_sin_token_el_script_falla_antes_de_calcular(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with mock.patch.object(aviso, "calcular_avisos") as calc:
        assert telegram.main([]) == 1
    calc.assert_not_called()
    assert "Faltan las variables de entorno" in capsys.readouterr().err


NOCHE = "20260701"


def pronostico_sintetico():
    """Horario del 30 jun 12 UTC al 1 jul 23 UTC, con T del GFS a 5 °C."""
    times = np.array([f"20260630{h:02d}" for h in range(12, 24)] + [f"{NOCHE}{h:02d}" for h in range(24)])
    X = np.empty((len(times), 17, 13, 4), np.float32)
    X[...] = [60.0, 278.15, 1.0, -1.0]
    return X, times


def predecir_sintetico(Xw, horas):
    """T = 2 °C − 1 °C por hora UTC: helada desde las 03 UTC."""
    Y = np.empty((len(Xw), 70, 35, 4), np.float32)
    Y[...] = [80.0, 0.0, 1.0, -1.0]
    Y[..., 1] = (2.0 - np.asarray(horas, np.float32))[:, None, None] + 273.15
    return Y


def test_calcular_avisos_da_uno_por_zona(monkeypatch):
    from test_paridad import static_sintetico

    with mock.patch.object(gfs_forecast, "get_gfs_forecast", return_value=pronostico_sintetico()), \
            mock.patch.object(predict, "cargar_static", return_value=static_sintetico()), \
            mock.patch.object(predict, "predecir", side_effect=predecir_sintetico):
        avisos = aviso.calcular_avisos(NOCHE)
    assert [a["clave"] for a in avisos] == ["carhuaz", "huaraz", "yungay"]
    assert all(a["resultado"]["nivel"] == "helada probable" for a in avisos)
    assert all(len(telegram.armar_mensaje(a)) <= 600 for a in avisos)
    with mock.patch.object(gfs_forecast, "get_gfs_forecast", return_value=pronostico_sintetico()), \
            mock.patch.object(predict, "cargar_static", return_value=static_sintetico()), \
            mock.patch.object(predict, "predecir", side_effect=predecir_sintetico), \
            pytest.raises(aviso.NocheNoCubierta):
        aviso.calcular_avisos("20260705")
    assert np.isfinite([a["resultado"]["tmin_c"] for a in avisos]).all()


def test_env_se_carga_sin_pisar_el_entorno(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@ya_definido")
    monkeypatch.delenv("HELADA_PRUEBA_VACIA", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# comentario\n\nexport TELEGRAM_BOT_TOKEN=\"{TOKEN}\"\nTELEGRAM_CHAT_ID=@del_archivo\n"
                   "HELADA_PRUEBA_VACIA=\nlinea sin igual\n", encoding="utf-8")
    cargadas = telegram.cargar_env(env)
    assert cargadas == ["TELEGRAM_BOT_TOKEN"]                       # devuelve nombres, nunca valores
    assert telegram.configuracion() == (TOKEN, "@ya_definido")      # el entorno manda sobre .env
    assert "HELADA_PRUEBA_VACIA" not in __import__("os").environ    # valores vacíos no se cargan
    assert telegram.cargar_env(tmp_path / "no_existe.env") == []


def test_main_lee_el_env_de_la_raiz(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"TELEGRAM_BOT_TOKEN={TOKEN}\nTELEGRAM_CHAT_ID={CANAL}\n", encoding="utf-8")
    monkeypatch.setattr(telegram, "RUTA_ENV", env)
    monkeypatch.setattr(telegram, "RUTA_REGISTRO", tmp_path / "r.sqlite")
    with mock.patch.object(aviso, "calcular_avisos", return_value=[aviso_de()]), \
            mock.patch.object(requests, "post", return_value=respuesta()) as post:
        assert telegram.main([]) == 0
    assert post.call_args.kwargs["json"]["chat_id"] == CANAL
    assert TOKEN not in capsys.readouterr().out


@pytest.mark.parametrize("args, texto", [
    (["--zonas", "lima"], "Zona desconocida: lima. Zonas disponibles: carhuaz, huaraz, yungay."),
    (["--noche", "27/11/2026"], "La noche «27/11/2026» no es válida: usa el formato AAAAMMDD"),
    (["--noche", "20261340"], "La noche «20261340» no es válida"),
])
def test_zona_o_noche_invalidas_dan_mensaje_en_español(args, texto, capsys):
    with mock.patch.object(gfs_forecast, "get_gfs_forecast") as descarga:
        assert telegram.main(["--prueba", *args]) == 1
    descarga.assert_not_called()                                    # falla antes de descargar
    err = capsys.readouterr().err
    assert texto in err and "Traceback" not in err and "KeyError" not in err


def test_aviso_usa_la_misma_variable_de_caché_que_la_app(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("HELADA_CACHE_GFS", str(tmp_path / "otra_cache.npz"))
    importlib.reload(aviso)
    try:
        assert aviso.CACHE_GFS == tmp_path / "otra_cache.npz"
    finally:
        monkeypatch.delenv("HELADA_CACHE_GFS")
        importlib.reload(aviso)
    assert aviso.CACHE_GFS == RAIZ / "app" / "gfs_cache.npz"

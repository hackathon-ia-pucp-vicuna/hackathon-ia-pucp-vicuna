"""
Confirmaciones sí/no de helada en SQLite (T19 · HU-06).

- El formulario de la app guarda zona, fecha y respuesta en SQLite (AppTest, sin red).
- Una segunda respuesta de la misma zona y fecha reemplaza a la anterior y no duplica el registro.
- No se guardan datos personales: la tabla tiene solo zona, fecha, respuesta y hora de guardado, y el
  formulario no pide texto libre.

Se ejecuta con:  pytest -q tests/test_confirmaciones.py
"""
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict  # noqa: E402
import storage  # noqa: E402
from frost.rules import LIMA  # noqa: E402
from test_streamlit_app import app  # noqa: E402,F401  (fixture: AppTest con red, relieve y modelo sintéticos)

ZONAS = {"carhuaz": {}, "huaraz": {}, "yungay": {}}
HOY = date(2026, 11, 27)


def filas(ruta):
    con = sqlite3.connect(ruta)
    try:
        return con.execute("SELECT zona, fecha, helada FROM confirmaciones ORDER BY zona, fecha").fetchall()
    finally:
        con.close()


def test_guarda_zona_fecha_y_respuesta(tmp_path):
    db = tmp_path / "c.sqlite"
    assert storage.guardar_confirmacion("carhuaz", date(2026, 11, 27), True, ZONAS, ruta=db, hoy=HOY) is False
    assert storage.guardar_confirmacion("huaraz", "2026-11-26", False, ZONAS, ruta=db, hoy=HOY) is False
    assert filas(db) == [("carhuaz", "2026-11-27", 1), ("huaraz", "2026-11-26", 0)]
    leidas = storage.leer_confirmaciones(db)
    assert [x["zona"] for x in leidas] == ["huaraz", "carhuaz"]       # por fecha
    assert all(datetime.fromisoformat(x["actualizado_utc"]) for x in leidas)


def test_segunda_respuesta_reemplaza_y_no_duplica(tmp_path):
    db = tmp_path / "c.sqlite"
    storage.guardar_confirmacion("carhuaz", HOY, True, ZONAS, ruta=db, hoy=HOY)
    assert storage.guardar_confirmacion("carhuaz", HOY, False, ZONAS, ruta=db, hoy=HOY) is True
    assert storage.guardar_confirmacion("carhuaz", "2026-11-27", False, ZONAS, ruta=db, hoy=HOY) is True
    assert filas(db) == [("carhuaz", "2026-11-27", 0)]                 # una fila, vale la última
    storage.guardar_confirmacion("carhuaz", HOY - timedelta(days=1), True, ZONAS, ruta=db, hoy=HOY)
    storage.guardar_confirmacion("yungay", HOY, True, ZONAS, ruta=db, hoy=HOY)
    assert len(filas(db)) == 3                                          # otra fecha u otra zona sí suman


def test_la_clave_unica_esta_en_la_base(tmp_path):
    db = tmp_path / "c.sqlite"
    storage.conectar(db).close()
    con = sqlite3.connect(db)
    con.execute("INSERT INTO confirmaciones VALUES ('carhuaz', '2026-11-27', 1, 'x')")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO confirmaciones VALUES ('carhuaz', '2026-11-27', 0, 'y')")
    con.close()


@pytest.mark.parametrize("zona, fecha, helada, texto", [
    ("lima", HOY, True, "no está en la lista"),
    ("carhuaz", "27/11/2026", True, "fecha no es válida"),
    ("carhuaz", HOY + timedelta(days=1), True, "todavía no pasó"),
    ("carhuaz", HOY, "sí", "sí o no"),
])
def test_datos_invalidos_dan_error_en_español(tmp_path, zona, fecha, helada, texto):
    with pytest.raises(storage.ConfirmacionInvalida, match=texto):
        storage.guardar_confirmacion(zona, fecha, helada, ZONAS, ruta=tmp_path / "c.sqlite", hoy=HOY)


def test_no_se_guardan_datos_personales(tmp_path):
    db = tmp_path / "c.sqlite"
    storage.guardar_confirmacion("carhuaz", HOY, True, ZONAS, ruta=db, hoy=HOY)
    con = sqlite3.connect(db)
    tablas = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    columnas = [c[1] for c in con.execute("PRAGMA table_info(confirmaciones)")]
    con.close()
    assert tablas == ["confirmaciones"]
    assert columnas == ["zona", "fecha", "helada", "actualizado_utc"]
    firma = storage.guardar_confirmacion.__code__.co_varnames[:storage.guardar_confirmacion.__code__.co_argcount]
    assert set(firma) == {"zona", "fecha", "helada", "zonas_validas", "ruta", "hoy"}


def test_el_formulario_guarda_en_sqlite(app, tmp_path, monkeypatch):  # noqa: F811
    at, _ = app
    db = tmp_path / "confirmaciones.sqlite"
    monkeypatch.setenv("HELADA_DB", str(db))
    at.run()
    assert not at.exception
    assert not at.text_input and not at.text_area                     # sin texto libre: no hay datos personales
    radio = next(r for r in at.radio if r.label.startswith("¿Heló esa noche"))
    noche = next(s for s in at.selectbox if s.label == "Noche")
    hoy = datetime.now(LIMA).date()
    assert noche.options[0] == f"Del {hoy - timedelta(days=1):%d/%m} al {hoy:%d/%m}" and len(noche.options) == 7
    radio.set_value("Sí")
    at.button[0].click().run()
    assert not at.exception
    assert any("Guardamos tu respuesta" in s.value for s in at.success)
    assert filas(db) == [("carhuaz", hoy.isoformat(), 1)]

    radio = next(r for r in at.radio if r.label.startswith("¿Heló esa noche"))
    radio.set_value("No")
    at.button[0].click().run()
    assert any("Actualizamos tu respuesta" in s.value for s in at.success)
    assert filas(db) == [("carhuaz", hoy.isoformat(), 0)]             # la segunda no duplica

    at.selectbox[0].set_value("yungay").run()
    radio = next(r for r in at.radio if r.label.startswith("¿Heló esa noche"))
    radio.set_value("Sí")
    next(s for s in at.selectbox if s.label == "Noche").set_value(hoy - timedelta(days=2))
    at.button[0].click().run()
    assert filas(db) == [("carhuaz", hoy.isoformat(), 0), ("yungay", (hoy - timedelta(days=2)).isoformat(), 1)]


def test_sin_respuesta_no_guarda(app, tmp_path, monkeypatch):  # noqa: F811
    at, _ = app
    db = tmp_path / "confirmaciones.sqlite"
    monkeypatch.setenv("HELADA_DB", str(db))
    at.run()
    at.button[0].click().run()
    assert any("Elige sí o no" in w.value for w in at.warning)
    assert not db.exists() or filas(db) == []


def test_el_formulario_funciona_aunque_falle_el_modelo(app, tmp_path, monkeypatch):  # noqa: F811
    at, _ = app
    db = tmp_path / "confirmaciones.sqlite"
    monkeypatch.setenv("HELADA_DB", str(db))
    with mock.patch.object(predict, "predecir", side_effect=predict.ModeloNoDisponible("Falta el modelo.")):
        at.run()
        assert at.error[0].value.startswith("El modelo no está disponible")
        next(r for r in at.radio if r.label.startswith("¿Heló esa noche")).set_value("No")
        at.button[0].click().run()
    assert not at.exception
    assert filas(db) == [("carhuaz", datetime.now(LIMA).date().isoformat(), 0)]


def test_si_falla_la_base_avisa_sin_trazas(app, tmp_path, monkeypatch):  # noqa: F811
    at, _ = app
    monkeypatch.setenv("HELADA_DB", str(tmp_path / "no_existe" / "c.sqlite"))   # carpeta inexistente
    at.run()
    next(r for r in at.radio if r.label.startswith("¿Heló esa noche")).set_value("Sí")
    at.button[0].click().run()
    assert not at.exception
    assert any(e.value == "No se pudo guardar tu respuesta. Inténtalo de nuevo en unos minutos." for e in at.error)

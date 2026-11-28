"""
Confirmaciones sí/no de helada en SQLite (T19 · HU-06).

Al día siguiente, la persona responde si heló en su zona. Se guarda solo:
    zona             clave de zones.yaml (carhuaz, huaraz, …)
    fecha            mañana de la noche confirmada, en hora de Lima (YYYY-MM-DD): la noche del 26 al 27/11
                     se guarda como 2026-11-27, que es también el día UTC de sus horas 00–12 (rules.py)
    helada           1 = sí heló, 0 = no heló
    actualizado_utc  cuándo se guardó la última respuesta
Zona y fecha forman la clave primaria: una segunda respuesta reemplaza a la anterior (vale la última) y
no se duplica. No se guarda ningún dato personal: ni nombre, ni teléfono, ni IP, ni texto libre.

La base es app/confirmaciones.sqlite (en .gitignore); HELADA_DB permite usar otra ruta.
En Streamlit Community Cloud el disco no es permanente: la base se pierde al reiniciar la app.
"""
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

RUTA_DB = Path(__file__).resolve().parent / "confirmaciones.sqlite"
COLUMNAS = ("zona", "fecha", "helada", "actualizado_utc")
LIMA = timezone(timedelta(hours=-5))


class ConfirmacionInvalida(ValueError):
    """La zona, la fecha o la respuesta no son válidas; el mensaje está en español."""


def ruta_db():
    return Path(os.environ.get("HELADA_DB", RUTA_DB))


def conectar(ruta=None):
    con = sqlite3.connect(ruta or ruta_db())
    con.execute("CREATE TABLE IF NOT EXISTS confirmaciones ("
                "zona TEXT NOT NULL, fecha TEXT NOT NULL, helada INTEGER NOT NULL CHECK (helada IN (0, 1)), "
                "actualizado_utc TEXT NOT NULL, PRIMARY KEY (zona, fecha))")
    return con


def guardar_confirmacion(zona, fecha, helada, zonas_validas, ruta=None, hoy=None):
    """
    Guarda la respuesta de `zona` para la noche que terminó la mañana de `fecha` (date o 'YYYY-MM-DD').
    Si ya había una, la reemplaza. Devuelve True si reemplazó una respuesta anterior.
    No acepta zonas fuera de `zonas_validas` ni fechas futuras (en hora de Lima).
    """
    if zona not in zonas_validas:
        raise ConfirmacionInvalida(f"La localidad '{zona}' no está en la lista.")
    try:
        f = fecha if isinstance(fecha, date) else date.fromisoformat(str(fecha))
    except ValueError:
        raise ConfirmacionInvalida("La fecha no es válida.") from None
    if f > (hoy or datetime.now(LIMA).date()):
        raise ConfirmacionInvalida("No se puede confirmar una noche que todavía no pasó.")
    if not isinstance(helada, bool):
        raise ConfirmacionInvalida("La respuesta debe ser sí o no.")
    con = conectar(ruta)
    try:
        habia = con.execute("SELECT 1 FROM confirmaciones WHERE zona = ? AND fecha = ?",
                            (zona, f.isoformat())).fetchone() is not None
        con.execute("INSERT INTO confirmaciones VALUES (?, ?, ?, ?) ON CONFLICT (zona, fecha) DO UPDATE SET "
                    "helada = excluded.helada, actualizado_utc = excluded.actualizado_utc",
                    (zona, f.isoformat(), int(helada), datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.commit()
    finally:
        con.close()
    return habia


def leer_confirmaciones(ruta=None):
    """Lista de dicts (zona, fecha, helada, actualizado_utc), ordenada por fecha y zona."""
    con = conectar(ruta)
    try:
        filas = con.execute(f"SELECT {', '.join(COLUMNAS)} FROM confirmaciones ORDER BY fecha, zona").fetchall()
    finally:
        con.close()
    return [dict(zip(COLUMNAS, f)) for f in filas]

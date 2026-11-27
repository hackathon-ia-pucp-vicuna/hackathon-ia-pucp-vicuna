"""
Pronóstico GFS para el producto, tolerante a fallos de red (T37 · HU-11).

Envuelve gfs_forecast.get_gfs_forecast() (Open-Meteo, con caché en disco) sin cambiar su interfaz:
- Si la caché tiene menos de `max_horas`, se usa sin llamar a la red.
- Si no, se descarga. Si Open-Meteo falla y hay caché, se usa la última descarga y se devuelve una
  advertencia con la fecha y hora (de Lima) en que se descargó.
- Si falla y no hay caché, se lanza SinPronostico con un mensaje en español.

Lo usan la app de Streamlit y el aviso por Telegram (app/aviso.py). Solo para el producto: nunca para
entrenar ni para la submission de Kaggle.
"""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import gfs_forecast

LIMA = timezone(timedelta(hours=-5))
HORAS_FUTURAS = 48
HORAS_PASADAS = 14
MSJ_SIN_PRONOSTICO = ("No se pudo obtener el pronóstico: revisa tu conexión a internet e inténtalo de nuevo "
                      "en unos minutos.")


class SinPronostico(RuntimeError):
    """Open-Meteo no respondió y no hay pronóstico guardado."""


def descargado_a_las(ruta, ahora=None):
    """(datetime de Lima en que se guardó la caché, horas transcurridas)."""
    cuando = datetime.fromtimestamp(os.path.getmtime(ruta), tz=timezone.utc).astimezone(LIMA)
    horas = ((ahora or datetime.now(timezone.utc)) - cuando).total_seconds() / 3600
    return cuando, horas


def advertencia_cache(ruta, ahora=None):
    cuando, horas = descargado_a_las(ruta, ahora)
    return (f"No se pudo conectar con el servicio de pronóstico. Se muestra el último pronóstico guardado, "
            f"descargado el {cuando:%d/%m} a las {cuando:%H:%M} (hora de Lima), hace {horas:.0f} h. "
            "Puede estar desactualizado.")


def obtener_pronostico(ruta_cache, max_horas=3):
    """(X, times, advertencia). advertencia es None si el pronóstico está al día."""
    ruta_cache = Path(ruta_cache)
    try:
        X, t = gfs_forecast.get_gfs_forecast(forecast_hours=HORAS_FUTURAS, past_hours=HORAS_PASADAS,
                                             cache_path=str(ruta_cache), cache_max_age_h=max_horas)
        return X, t, None
    except Exception as e:
        if ruta_cache.exists():
            try:
                d = np.load(ruta_cache)
                return d["X"], d["times"], advertencia_cache(ruta_cache)
            except Exception:
                pass                                   # caché dañada: se trata como si no hubiera
        raise SinPronostico(MSJ_SIN_PRONOSTICO) from e

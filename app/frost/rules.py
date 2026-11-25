"""
Reglas de helada (v0 · T44): zona → celda, noche objetivo, nivel de riesgo, inicio y duración.
T15 (@lalo-dev8) las completa y agrega pruebas. Sin red ni PyTorch: solo numpy y la biblioteca estándar.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

FROST = Path(__file__).resolve().parent
LIMA = timezone(timedelta(hours=-5))
NIVELES = ("sin riesgo", "vigilancia", "helada probable")
HORAS_NOCHE_UTC = range(0, 13)            # 00–12 UTC = 19:00–07:00 Lima


class ZonaFueraDeGrilla(ValueError):
    pass


def cargar_yaml(nombre):
    with open(FROST / nombre, encoding="utf-8") as f:
        return yaml.safe_load(f)


def celda_de_zona(lat, lon, wrf_lats, wrf_lons):
    """(fila, columna) de la celda de 3 km más cercana. wrf_lats con fila 0 = sur."""
    paso_lat = abs(wrf_lats[1] - wrf_lats[0])
    paso_lon = abs(wrf_lons[1] - wrf_lons[0])
    if not (min(wrf_lats) - paso_lat / 2 <= lat <= max(wrf_lats) + paso_lat / 2
            and min(wrf_lons) - paso_lon / 2 <= lon <= max(wrf_lons) + paso_lon / 2):
        raise ZonaFueraDeGrilla("La localidad está fuera del área que cubre el modelo.")
    return int(np.abs(wrf_lats - lat).argmin()), int(np.abs(wrf_lons - lon).argmin())


def noche_objetivo(ahora_utc=None):
    """
    Fecha UTC 'YYYYMMDD' de la noche a mostrar (horas 00–12 UTC de ese día).
    Antes de las 07:00 Lima: la noche en curso. Desde las 07:00 Lima: la noche siguiente.
    """
    ahora_lima = (ahora_utc or datetime.now(timezone.utc)).astimezone(LIMA)
    fecha = ahora_lima.date() if ahora_lima.hour < 7 else ahora_lima.date() + timedelta(days=1)
    return fecha.strftime("%Y%m%d")


def horas_de_la_noche(times, fecha_utc):
    """Máscara booleana de las horas 'YYYYMMDDHH' que caen en la noche `fecha_utc`."""
    times = np.asarray(times).astype(str)
    return np.array([t[:8] == fecha_utc and int(t[8:10]) in HORAS_NOCHE_UTC for t in times])


def a_hora_lima(t_utc):
    return datetime.strptime(str(t_utc), "%Y%m%d%H").replace(tzinfo=timezone.utc).astimezone(LIMA)


def evaluar_noche(t_celsius, times, umbral):
    """
    t_celsius : (H,) temperatura del modelo en la celda de la zona, horas de la noche en orden.
    times     : (H,) 'YYYYMMDDHH' UTC.
    umbral    : dict de thresholds.yaml (helada_c, vigilancia_c).
    """
    t = np.asarray(t_celsius, dtype=float)
    tmin = float(t.min())
    if tmin <= umbral["helada_c"]:
        nivel = NIVELES[2]
    elif tmin <= umbral["vigilancia_c"]:
        nivel = NIVELES[1]
    else:
        nivel = NIVELES[0]

    bajo = t <= umbral["helada_c"]
    inicio = a_hora_lima(np.asarray(times)[bajo.argmax()]) if bajo.any() else None
    return {
        "nivel": nivel,
        "tmin_c": tmin,
        "hora_tmin": a_hora_lima(np.asarray(times)[t.argmin()]),
        "inicio": inicio,                  # None = no se espera bajar del umbral
        "duracion_h": int(bajo.sum()),     # horas con T <= umbral (pronóstico horario)
    }

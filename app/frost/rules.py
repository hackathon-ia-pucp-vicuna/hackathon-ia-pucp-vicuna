"""
Reglas de helada: zona → celda, noche objetivo, nivel de riesgo, inicio y duración (T15 · HU-02).
Sin red ni PyTorch: solo numpy y la biblioteca estándar. Los umbrales se leen de thresholds.yaml.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

FROST = Path(__file__).resolve().parent
LIMA = timezone(timedelta(hours=-5))
NIVELES = ("sin riesgo", "vigilancia", "helada probable")
# 01–12 UTC = 20:00–07:00 Lima. Se excluye 00 UTC (19:00): el WRF se reinicia a esa hora y ese cuadro
# se parece al estado inicial (GFS), con un salto frente a la 01 UTC que el modelo reproduce.
HORAS_NOCHE_UTC = range(1, 13)


class ZonaFueraDeGrilla(ValueError):
    pass


class UmbralInvalido(ValueError):
    """thresholds.yaml no tiene el cultivo o sus valores no son coherentes."""


def cargar_yaml(nombre):
    with open(FROST / nombre, encoding="utf-8") as f:
        return yaml.safe_load(f)


CAMPOS_FUENTE = ("documento", "institucion", "anio", "enlace")


def _es_numero(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def umbral_de_cultivo(cultivo="papa", umbrales=None):
    """
    Umbral del cultivo leído de thresholds.yaml (o de `umbrales`, el dict ya cargado).

    Con validado: true y una fuente completa se usa helada_cultivo_c; si no, la referencia
    general (referencia_general_c, 0 °C) y `es_referencia_general` queda en True para rotularlo.
    Devuelve el dict del YAML más helada_c (el umbral que se aplica) y es_referencia_general.
    Exige helada_c y vigilancia_c numéricos, con helada_c <= vigilancia_c.
    """
    umbrales = cargar_yaml("thresholds.yaml") if umbrales is None else umbrales
    if cultivo not in umbrales:
        raise UmbralInvalido(f"No hay umbrales para el cultivo '{cultivo}' en thresholds.yaml.")
    u = dict(umbrales[cultivo])
    if "helada_c" not in u:                          # formato con fuente: elegir según validación
        if u.get("validado") is True:
            fuente = u.get("fuente") or {}
            faltan = [c for c in CAMPOS_FUENTE if not fuente.get(c)]
            if faltan:
                raise UmbralInvalido(f"'{cultivo}' está validado pero a su fuente le falta: {', '.join(faltan)}.")
            u["helada_c"], u["es_referencia_general"] = u.get("helada_cultivo_c"), False
        else:
            u["helada_c"], u["es_referencia_general"] = u.get("referencia_general_c", 0.0), True
    else:
        u.setdefault("es_referencia_general", not u.get("validado", False))
    for clave in ("helada_c", "vigilancia_c"):
        if not _es_numero(u.get(clave)):
            raise UmbralInvalido(f"'{cultivo}.{clave}' debe ser un número en thresholds.yaml.")
    if u["helada_c"] > u["vigilancia_c"]:
        raise UmbralInvalido(f"'{cultivo}': helada_c no puede ser mayor que vigilancia_c.")
    return u


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
    Fecha UTC 'YYYYMMDD' de la noche a mostrar (horas 01–12 UTC de ese día).
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


def nivel_de_riesgo(tmin_c, umbral):
    """sin riesgo > vigilancia_c · vigilancia en (helada_c, vigilancia_c] · helada probable <= helada_c."""
    if tmin_c <= umbral["helada_c"]:
        return NIVELES[2]
    if tmin_c <= umbral["vigilancia_c"]:
        return NIVELES[1]
    return NIVELES[0]


def _paso_horas(fechas):
    """Paso típico entre horas consecutivas (1 h en el pronóstico horario, 3 h si viene cada 3 h)."""
    if len(fechas) < 2:
        return 1.0
    difs = np.diff([f.timestamp() for f in fechas]) / 3600
    return float(np.median(difs))


def evaluar_noche(t_celsius, times, umbral):
    """
    t_celsius : (H,) temperatura del modelo en la celda de la zona, horas de la noche.
    times     : (H,) 'YYYYMMDDHH' UTC (se ordenan si vienen desordenadas).
    umbral    : dict de thresholds.yaml (helada_c, vigilancia_c); ver umbral_de_cultivo().

    Devuelve nivel, T mínima y su hora, inicio (primera hora con T <= helada_c, hora de Lima),
    fin (última hora bajo el umbral), duración en horas y los tramos bajo el umbral.
    Cada hora bajo el umbral cuenta como un paso de pronóstico (1 h si es horario).
    """
    t = np.asarray(t_celsius, dtype=float)
    tt = np.asarray(times).astype(str)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("Se necesita al menos una hora de temperatura para evaluar la noche.")
    if t.shape != tt.shape:
        raise ValueError("Temperaturas y horas deben tener el mismo largo.")
    if np.isnan(t).any():
        raise ValueError("Hay horas sin temperatura (NaN); no se puede evaluar la noche.")

    orden = np.argsort(tt, kind="stable")
    t, tt = t[orden], tt[orden]
    fechas = [a_hora_lima(x) for x in tt]
    paso = _paso_horas(fechas)

    bajo = t <= umbral["helada_c"]
    tramos = []
    i = 0
    while i < len(t):
        if bajo[i]:
            j = i
            while j + 1 < len(t) and bajo[j + 1]:
                j += 1
            tramos.append({"inicio": fechas[i], "fin": fechas[j], "horas": (j - i + 1) * paso})
            i = j + 1
        else:
            i += 1

    tmin = float(t.min())
    duracion = float(sum(tr["horas"] for tr in tramos))
    return {
        "nivel": nivel_de_riesgo(tmin, umbral),
        "tmin_c": tmin,
        "hora_tmin": fechas[int(t.argmin())],
        "inicio": tramos[0]["inicio"] if tramos else None,   # None = no se espera bajar del umbral
        "fin": tramos[-1]["fin"] if tramos else None,
        "duracion_h": int(duracion) if duracion.is_integer() else duracion,
        "tramos": tramos,
    }

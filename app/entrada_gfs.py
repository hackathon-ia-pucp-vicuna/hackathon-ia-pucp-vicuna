"""
Adaptador Open-Meteo → entrada del modelo (T35 · HU-01).

Convierte el pronóstico GFS de Open-Meteo (gfs_forecast.py) en las ventanas que espera predecir():
    Xw    : (N, 3, 17, 13, 4) = [t-2, t-1, t], RH (%), T (K), U (m/s), V (m/s); fila 0 = norte
    times : (N,) 'YYYYMMDDHH' UTC de la hora t
    horas : (N,) hora UTC (0–23) de cada ventana

Horas faltantes: gfs_forecast.py descarta las horas incompletas, así que puede haber huecos.
Una ventana solo se arma si t-2 y t-1 existen exactamente 2 y 1 horas antes; si no, esa hora
se descarta y se informa en `descartadas` (nunca se inventan datos).

Regla de la hackathon: Open-Meteo es solo para el producto. Este módulo no se usa para entrenar
ni para la submission de Kaggle.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from frost.rules import HORAS_NOCHE_UTC, a_hora_lima, horas_de_la_noche
from gfs_forecast import LATS, LONS, get_gfs_forecast

VARIABLES = ("RH", "T", "U", "V")
FORMA_GRILLA = (len(LATS), len(LONS), len(VARIABLES))    # (17, 13, 4)
RANGOS = {"RH": (0.0, 100.5), "T": (180.0, 340.0), "U": (-80.0, 80.0), "V": (-80.0, 80.0)}
CACHE_POR_DEFECTO = Path(__file__).resolve().parent / "gfs_cache.npz"   # en .gitignore


class EntradaInvalida(ValueError):
    """El pronóstico no tiene la forma, el orden o las unidades del entrenamiento."""


def validar_grilla(X, times):
    """Comprueba forma (H, 17, 13, 4), horas 'YYYYMMDDHH', valores finitos y unidades plausibles."""
    X = np.asarray(X)
    times = np.asarray(times).astype(str)
    if X.ndim != 4 or X.shape[1:] != FORMA_GRILLA:
        raise EntradaInvalida(f"Se esperaba (horas, 17, 13, 4) y llegó {X.shape}.")
    if len(times) != len(X):
        raise EntradaInvalida("El número de horas no coincide con el pronóstico.")
    if any(len(t) != 10 or not t.isdigit() for t in times):
        raise EntradaInvalida("Las horas deben tener el formato 'YYYYMMDDHH' (UTC).")
    if not np.isfinite(X).all():
        raise EntradaInvalida("El pronóstico tiene valores vacíos (NaN).")
    for c, v in enumerate(VARIABLES):
        lo, hi = RANGOS[v]
        if X[..., c].min() < lo or X[..., c].max() > hi:
            raise EntradaInvalida(
                f"{v} fuera de rango ({X[..., c].min():.1f} a {X[..., c].max():.1f}); "
                "revisa unidades (T en K) y orden RH, T, U, V.")
    return X.astype(np.float32), times


def ventanas_por_hora(X, times, k=3):
    """
    Ventanas [t-(k-1), …, t] con horas consecutivas exactas.
    Devuelve (Xw, times_ok, horas_utc, descartadas): las horas sin sus k-1 horas previas van a `descartadas`.
    """
    X, times = validar_grilla(X, times)
    orden = np.argsort(times, kind="stable")
    X, times = X[orden], times[orden]
    if len(np.unique(times)) != len(times):
        raise EntradaInvalida("Hay horas repetidas en el pronóstico.")

    fechas = [datetime.strptime(t, "%Y%m%d%H") for t in times]
    pos = {f: i for i, f in enumerate(fechas)}
    idx, ok, descartadas = [], [], []
    for i, f in enumerate(fechas):
        previas = [pos.get(f - timedelta(hours=lag)) for lag in range(k - 1, 0, -1)]
        if None in previas:
            descartadas.append(times[i])
            continue
        idx.append(previas + [i])
        ok.append(times[i])

    if not idx:
        vacio = np.empty((0, k) + FORMA_GRILLA, np.float32)
        return vacio, np.array([], dtype=str), np.array([], dtype=np.int64), descartadas
    Xw = X[np.array(idx)]                                  # (N, k, 17, 13, 4)
    times_ok = np.array(ok)
    horas = np.array([int(t[8:10]) for t in times_ok], dtype=np.int64)
    return Xw, times_ok, horas, descartadas


def ventanas_de_la_noche(X, times, noche):
    """
    Ventanas de las horas de la noche `noche` (YYYYMMDD UTC, 00–12 UTC), solo con horas consecutivas.
    Lo usan la app y el aviso de Telegram. Devuelve (Xw, times, X_t, faltan): X_t es el GFS de la hora t
    de cada ventana y `faltan` las horas de la noche que no se pueden predecir ('YYYYMMDDHH').
    """
    Xw, t_ok, _, _ = ventanas_por_hora(X, times)
    X, times = validar_grilla(X, times)
    m = horas_de_la_noche(t_ok, noche)
    t_noche = t_ok[m]
    faltan = [f"{noche}{h:02d}" for h in HORAS_NOCHE_UTC if f"{noche}{h:02d}" not in set(t_noche)]
    pos = {t: i for i, t in enumerate(times)}
    return Xw[m], t_noche, X[[pos[t] for t in t_noche]], faltan


def advertencia_horas_faltantes(faltan):
    """Texto en español para las horas de la noche sin predicción (hora de Lima), o None si no falta ninguna."""
    if not faltan:
        return None
    horas = ", ".join(f"{a_hora_lima(t):%H:%M}" for t in faltan)
    return (f"Faltan datos del pronóstico para estas horas de la noche (hora de Lima): {horas}. "
            "La mínima y la duración pueden estar subestimadas.")


def obtener_entrada(forecast_hours=48, past_hours=2, cache_path=CACHE_POR_DEFECTO, ttl_h=3.0,
                    descargar=get_gfs_forecast):
    """
    Descarga (o lee de caché si tiene menos de `ttl_h` horas) el GFS de Open-Meteo y arma las ventanas.
    past_hours >= 2 para que la primera hora del pronóstico tenga su t-2 y t-1.
    Devuelve dict con Xw, times, horas_utc y descartadas.
    """
    if past_hours < 2:
        raise ValueError("past_hours debe ser al menos 2 para armar la ventana t-2, t-1, t.")
    X, times = descargar(forecast_hours=forecast_hours, past_hours=past_hours,
                         cache_path=str(cache_path) if cache_path else None, cache_max_age_h=ttl_h)
    Xw, t_ok, horas, descartadas = ventanas_por_hora(X, times)
    return {"Xw": Xw, "times": t_ok, "horas_utc": horas, "descartadas": descartadas}

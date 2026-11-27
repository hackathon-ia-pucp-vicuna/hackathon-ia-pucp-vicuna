"""
Aviso de la noche por zona, sin interfaz (T18 · HU-05).

    Open-Meteo (GFS, con caché) → predecir() → reglas por cultivo → mensaje

Lo usa el envío por Telegram (app/notify/telegram.py). Es el mismo cálculo que la app de Streamlit:
mismo pronóstico (si falla Open-Meteo, el último guardado; app/pronostico.py), mismas ventanas, mismo
`predecir()` (mismos pesos que Kaggle), mismas reglas y mismos mensajes.
"""
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from frost.advice import mensaje  # noqa: E402
from frost.rules import (a_hora_lima, cargar_yaml, celda_de_zona, evaluar_noche,  # noqa: E402
                         horas_de_la_noche, noche_objetivo, umbral_de_cultivo)
import gfs_forecast  # noqa: E402
import predict  # noqa: E402
from pronostico import obtener_pronostico  # noqa: E402

CACHE_GFS = APP / "gfs_cache.npz"     # en .gitignore; la misma caché que la app
CACHE_HORAS = 3


class NocheNoCubierta(RuntimeError):
    """El pronóstico no tiene horas de la noche objetivo."""


def calcular_avisos(noche=None, cultivo="papa", claves=None):
    """
    Aviso de cada zona de zones.yaml (o solo de `claves`) para la noche `noche` (YYYYMMDD UTC; por
    defecto la de rules.noche_objetivo()). Devuelve una lista de dicts con clave, zona, noche,
    cultivo, umbral, el resultado de evaluar_noche(), el mensaje de advice.py y, si se usó el pronóstico
    guardado porque falló Open-Meteo, su advertencia.
    """
    noche = noche or noche_objetivo()
    zonas = cargar_yaml("zones.yaml")
    claves = list(zonas) if claves is None else list(claves)
    umbral = umbral_de_cultivo(cultivo)
    X, times, advertencia = obtener_pronostico(CACHE_GFS, CACHE_HORAS)
    Xw, _ = gfs_forecast.make_windows(X, times)
    m = horas_de_la_noche(times, noche)
    if not m.any():
        raise NocheNoCubierta("El pronóstico disponible no cubre la noche pedida.")
    t_noche = np.asarray(times)[m].astype(str)
    Y = predict.predecir(Xw[m], [int(t[8:10]) for t in t_noche])
    st = predict.cargar_static()
    avisos = []
    for clave in claves:
        zona = zonas[clave]
        fila, col = celda_de_zona(zona["lat"], zona["lon"], st["wrf_lats"], st["wrf_lons"])
        r = evaluar_noche(Y[:, fila, col, 1] - 273.15, t_noche, umbral)
        avisos.append({"clave": clave, "zona": zona["nombre"], "noche": noche, "cultivo": umbral.get("nombre", cultivo),
                       "umbral": umbral, "resultado": r, "mensaje": mensaje(r["nivel"], r, zona["nombre"]),
                       "inicio_noche": a_hora_lima(f"{noche}00"), "advertencia": advertencia})
    return avisos

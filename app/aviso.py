"""
Aviso de la noche por zona, sin interfaz (T18 · HU-05).

    Open-Meteo (GFS, con caché) → predecir() → reglas por cultivo → mensaje

Lo usa el envío por Telegram (app/notify/telegram.py). Es el mismo cálculo que la app de Streamlit:
mismas ventanas, mismo `predecir()` (mismos pesos que Kaggle), mismas reglas y mismos mensajes.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from frost.advice import mensaje  # noqa: E402
from frost.rules import (a_hora_lima, cargar_yaml, celda_de_zona, evaluar_noche,  # noqa: E402
                         noche_objetivo, umbral_de_cultivo)
from entrada_gfs import advertencia_horas_faltantes, ventanas_de_la_noche  # noqa: E402
import gfs_forecast  # noqa: E402
import predict  # noqa: E402

# La misma caché que la app (en .gitignore); HELADA_CACHE_GFS permite otra ruta, igual que en la app
CACHE_GFS = Path(os.environ.get("HELADA_CACHE_GFS", APP / "gfs_cache.npz"))
CACHE_HORAS = 3


class NocheNoCubierta(RuntimeError):
    """El pronóstico no tiene horas de la noche objetivo."""


class PedidoInvalido(ValueError):
    """Zona o noche pedidas que no existen. El mensaje está en español."""


def validar_pedido(noche, claves, zonas):
    """Revisa --noche (YYYYMMDD, fecha real) y --zonas (claves de zones.yaml) antes de descargar nada."""
    if noche is not None:
        try:
            if len(noche) != 8 or not noche.isdigit():
                raise ValueError
            datetime.strptime(noche, "%Y%m%d")
        except (TypeError, ValueError):
            raise PedidoInvalido(f"La noche «{noche}» no es válida: usa el formato AAAAMMDD, por ejemplo 20261127.") from None
    if claves is not None:
        desconocidas = [c for c in claves if c not in zonas]
        if desconocidas:
            raise PedidoInvalido(f"Zona desconocida: {', '.join(desconocidas)}. "
                                 f"Zonas disponibles: {', '.join(zonas)}.")


def calcular_avisos(noche=None, cultivo="papa", claves=None):
    """
    Aviso de cada zona de zones.yaml (o solo de `claves`) para la noche `noche` (YYYYMMDD UTC; por
    defecto la de rules.noche_objetivo()). Devuelve una lista de dicts con clave, zona, noche,
    cultivo, umbral, el resultado de evaluar_noche() y el mensaje de advice.py.
    """
    zonas = cargar_yaml("zones.yaml")
    validar_pedido(noche, claves, zonas)
    noche = noche or noche_objetivo()
    claves = list(zonas) if claves is None else list(claves)
    umbral = umbral_de_cultivo(cultivo)
    X, times = gfs_forecast.get_gfs_forecast(forecast_hours=48, past_hours=14,
                                             cache_path=str(CACHE_GFS), cache_max_age_h=CACHE_HORAS)
    Xw, t_noche, _, faltan = ventanas_de_la_noche(X, times, noche)
    if not len(t_noche):
        raise NocheNoCubierta("El pronóstico disponible no cubre la noche pedida.")
    Y = predict.predecir(Xw, [int(t[8:10]) for t in t_noche])
    aviso_horas = advertencia_horas_faltantes(faltan)
    st = predict.cargar_static()
    avisos = []
    for clave in claves:
        zona = zonas[clave]
        fila, col = celda_de_zona(zona["lat"], zona["lon"], st["wrf_lats"], st["wrf_lons"])
        r = evaluar_noche(Y[:, fila, col, 1] - 273.15, t_noche, umbral)
        avisos.append({"clave": clave, "zona": zona["nombre"], "noche": noche, "cultivo": umbral.get("nombre", cultivo),
                       "umbral": umbral, "resultado": r,
                       "mensaje": " ".join(filter(None, (mensaje(r["nivel"], r, zona["nombre"]), aviso_horas))),
                       "inicio_noche": a_hora_lima(f"{noche}00")})
    return avisos

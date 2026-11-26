"""
Helada Andina · App v0 de punta a punta (T44 · HU-02).

    zona → Open-Meteo (GFS, con caché) → predecir() → reglas → mensaje

Ejecutar desde la raíz del repo (Python 3.12):
    pip install -r requirements-app.txt
    streamlit run app/web/streamlit_app.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

from frost.advice import AVISO_LIMITES, mensaje  # noqa: E402
from frost.rules import (ZonaFueraDeGrilla, a_hora_lima, cargar_yaml, celda_de_zona,  # noqa: E402
                         evaluar_noche, horas_de_la_noche, noche_objetivo, umbral_de_cultivo)
from gfs_forecast import get_gfs_forecast, make_windows  # noqa: E402
from predict import ModeloNoDisponible, cargar_static, predecir  # noqa: E402

CACHE_GFS = APP / "gfs_cache.npz"     # en .gitignore
CACHE_HORAS = 3


def pronostico_gfs():
    """(X, times, aviso). Si Open-Meteo falla, usa la última caché con un aviso; si no hay, lanza el error."""
    try:
        X, t = get_gfs_forecast(forecast_hours=48, past_hours=14,
                                cache_path=str(CACHE_GFS), cache_max_age_h=CACHE_HORAS)
        return X, t, None
    except Exception:
        if CACHE_GFS.exists():
            d = np.load(CACHE_GFS)
            horas = (time.time() - os.path.getmtime(CACHE_GFS)) / 3600
            return d["X"], d["times"], (
                f"No se pudo conectar con el servicio de pronóstico. "
                f"Se muestra el último pronóstico guardado (de hace {horas:.0f} h)."
            )
        raise


st.set_page_config(page_title="Helada Andina", page_icon="❄️")
st.title("Helada Andina")
st.caption("Aviso de helada para la papa en el Callejón de Huaylas · versión 0")

zonas = cargar_yaml("zones.yaml")
umbral = umbral_de_cultivo("papa")   # 0 °C mientras la fuente no esté validada (T07)
bajo = f"bajo {umbral['helada_c']:g} °C"
clave = st.selectbox("Localidad", list(zonas), format_func=lambda k: zonas[k]["nombre"])
zona = zonas[clave]

try:
    static = cargar_static()
    fila, col = celda_de_zona(zona["lat"], zona["lon"], static["wrf_lats"], static["wrf_lons"])

    with st.spinner("Descargando el pronóstico…"):
        X, times, aviso = pronostico_gfs()
    if aviso:
        st.warning(aviso)

    noche = noche_objetivo()
    Xw, _ = make_windows(X, times)
    m = horas_de_la_noche(times, noche)
    if not m.any():
        st.error("El pronóstico disponible no cubre la noche de hoy. Intenta de nuevo más tarde.")
        st.stop()

    t_noche = np.asarray(times)[m].astype(str)
    with st.spinner("Calculando con el modelo…"):
        Y = predecir(Xw[m], [int(t[8:10]) for t in t_noche])
    t_celda = Y[:, fila, col, 1] - 273.15
    r = evaluar_noche(t_celda, t_noche, umbral)

except ModeloNoDisponible as e:
    st.error(f"El modelo no está disponible. {e}")
    st.stop()
except ZonaFueraDeGrilla as e:
    st.error(str(e))
    st.stop()
except Exception:
    st.error("No se pudo obtener el pronóstico: revisa tu conexión a internet e inténtalo de nuevo en unos minutos.")
    st.stop()

inicio_noche = a_hora_lima(f"{noche}00")
st.subheader(f"Noche del {inicio_noche:%d/%m} al {inicio_noche + pd.Timedelta(days=1):%d/%m} · {zona['nombre']}")

colores = {"sin riesgo": st.success, "vigilancia": st.warning, "helada probable": st.error}
colores[r["nivel"]](f"**{r['nivel'].capitalize()}** · {mensaje(r['nivel'])}")

c1, c2, c3 = st.columns(3)
c1.metric("Temperatura mínima", f"{r['tmin_c']:.1f} °C", help=f"Hacia las {r['hora_tmin']:%H:%M} (hora de Lima)")
c2.metric(f"Inicio {bajo}", f"{r['inicio']:%H:%M}" if r["inicio"] else "No se espera")
c3.metric(f"Duración {bajo}", f"{r['duracion_h']} h" if r["inicio"] else "—")

if umbral["es_referencia_general"]:
    st.caption(f"Umbral: referencia general de {umbral['helada_c']:g} °C, no específica para la papa "
               "(fuente pendiente de validar).")
else:
    st.caption(f"Umbral para la papa: {umbral['helada_c']:g} °C · Fuente: {umbral['fuente']['institucion'].split(';')[0]} "
               f"({umbral['fuente']['anio']}).")
st.caption(AVISO_LIMITES)

with st.expander("Temperatura por hora (modelo, hora de Lima)"):
    st.dataframe(pd.DataFrame({
        "Hora": [f"{a_hora_lima(t):%d/%m %H:%M}" for t in t_noche],
        "T (°C)": np.round(t_celda, 1),
    }), hide_index=True)

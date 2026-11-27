"""
Helada Andina · App de Streamlit (T44, T17, T19, T37 · HU-02, HU-04, HU-06, HU-07, HU-11).

    zona y cultivo → Open-Meteo (GFS, con caché) → predecir() → reglas → mensaje
                                                 ↘ GFS crudo interpolado (comparación)

Muestra el nivel de riesgo, la hora de inicio y la duración bajo el umbral del cultivo, la T mínima
del modelo y la del GFS crudo, y la tabla por hora (hora de Lima). Al final, un formulario para confirmar
si heló en una noche pasada (SQLite, sin datos personales; ver app/storage.py).

Si falla Open-Meteo se usa el último pronóstico guardado, con la hora en que se descargó y una
advertencia (app/pronostico.py); si no hay ninguno, un mensaje en español.

Caché: el pronóstico de Open-Meteo se guarda en app/gfs_cache.npz (3 h) y el cálculo de la noche
(modelo y GFS crudo en toda la grilla) en la caché de Streamlit por 3 h, así que cambiar de zona o de
cultivo no vuelve a correr el modelo.

Ejecutar desde la raíz del repo (Python 3.12):
    pip install -r requirements-app.txt
    streamlit run app/web/streamlit_app.py
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

from frost.advice import AVISO_LIMITES, mensaje  # noqa: E402
from frost.comparar import t_gfs_crudo_c  # noqa: E402
from frost.rules import (LIMA, UmbralInvalido, ZonaFueraDeGrilla, a_hora_lima, cargar_yaml,  # noqa: E402
                         celda_de_zona, evaluar_noche, horas_de_la_noche, noche_objetivo,
                         umbral_de_cultivo)
from gfs_forecast import make_windows  # noqa: E402
from predict import ModeloNoDisponible, cargar_static, predecir  # noqa: E402
from pronostico import SinPronostico, obtener_pronostico  # noqa: E402
from storage import ConfirmacionInvalida, guardar_confirmacion  # noqa: E402

CACHE_GFS = Path(os.environ.get("HELADA_CACHE_GFS", APP / "gfs_cache.npz"))   # en .gitignore
CACHE_HORAS = 3

MSJ_NOCHE_FUERA = "El pronóstico disponible no cubre la noche de hoy. Intenta de nuevo más tarde."
NOCHES_A_CONFIRMAR = 7
MSJ_INTERNO = ("Ocurrió un problema al preparar el aviso. Inténtalo de nuevo en unos minutos; "
               "si sigue pasando, avísale al equipo de Helada Andina.")


class NocheNoCubierta(RuntimeError):
    """El pronóstico no tiene horas de la noche objetivo."""


@st.cache_data(ttl=CACHE_HORAS * 3600, show_spinner=False, max_entries=4)
def calcular_noche(noche):
    """
    Modelo y GFS crudo en toda la grilla para las horas de la noche `noche` (YYYYMMDD, UTC).
    No depende de la zona ni del cultivo: se calcula una vez y queda en caché.
    Devuelve horas UTC, T del modelo y T del GFS crudo en °C (H,70,35) y el aviso de caché vieja.
    """
    X, times, aviso = obtener_pronostico(CACHE_GFS, CACHE_HORAS)
    Xw, _ = make_windows(X, times)
    m = horas_de_la_noche(times, noche)
    if not m.any():
        raise NocheNoCubierta(MSJ_NOCHE_FUERA)
    t_noche = np.asarray(times)[m].astype(str)
    Y = predecir(Xw[m], [int(t[8:10]) for t in t_noche])
    return {"times": t_noche, "t_modelo": Y[..., 1] - 273.15,
            "t_gfs": t_gfs_crudo_c(np.asarray(X)[m], cargar_static()), "aviso": aviso}


def tabla_por_hora(t_noche, t_modelo, t_gfs, umbral_c):
    return pd.DataFrame({
        "Hora (Lima)": [f"{a_hora_lima(t):%d/%m %H:%M}" for t in t_noche],
        "Modelo (°C)": np.round(t_modelo, 1),
        "GFS crudo (°C)": np.round(t_gfs, 1),
        "Diferencia (°C)": np.round(t_modelo - t_gfs, 1),
        f"Bajo {umbral_c:g} °C": ["sí" if x <= umbral_c else "" for x in t_modelo],
    })


def formulario_confirmacion(clave, zonas):
    """Confirmación sí/no de una noche pasada en la zona elegida (T19). Guarda solo zona, fecha y respuesta."""
    zona = zonas[clave]
    with st.expander(f"¿Heló en {zona['nombre']}? Confírmalo y ayúdanos a mejorar el aviso"):
        hoy = datetime.now(LIMA).date()
        mananas = [hoy - timedelta(days=i) for i in range(NOCHES_A_CONFIRMAR)]
        with st.form("confirmacion"):
            fecha = st.selectbox("Noche", mananas,
                                 format_func=lambda d: f"Del {d - timedelta(days=1):%d/%m} al {d:%d/%m}")
            respuesta = st.radio(f"¿Heló esa noche en {zona['nombre']}?", ["Sí", "No"], index=None, horizontal=True)
            enviado = st.form_submit_button("Guardar respuesta")
        if enviado:
            if respuesta is None:
                st.warning("Elige sí o no antes de guardar.")
            else:
                try:
                    reemplazo = guardar_confirmacion(clave, fecha, respuesta == "Sí", zonas)
                    st.success("Actualizamos tu respuesta para esa noche. ¡Gracias!" if reemplazo
                               else "Guardamos tu respuesta. ¡Gracias!")
                except ConfirmacionInvalida as e:
                    st.error(str(e))
                except Exception:
                    st.error("No se pudo guardar tu respuesta. Inténtalo de nuevo en unos minutos.")
        st.caption("Solo guardamos la localidad, la noche y tu respuesta. No pedimos ni guardamos datos personales.")


def mostrar_error(texto):
    st.error(texto)
    if "clave" in globals():                         # la confirmación no depende del pronóstico ni del modelo
        formulario_confirmacion(clave, zonas)
    st.stop()


st.set_page_config(page_title="Helada Andina", page_icon="❄️")
st.title("Helada Andina")
st.caption("Aviso de helada por cultivo en el Callejón de Huaylas")

try:
    zonas = cargar_yaml("zones.yaml")
    cultivos = cargar_yaml("thresholds.yaml")
except Exception:
    mostrar_error(MSJ_INTERNO)

c_zona, c_cultivo = st.columns(2)
clave = c_zona.selectbox("Localidad", list(zonas), format_func=lambda k: zonas[k]["nombre"])
cultivo = c_cultivo.selectbox("Cultivo", list(cultivos), format_func=lambda k: cultivos[k].get("nombre", k))
zona = zonas[clave]

try:
    umbral = umbral_de_cultivo(cultivo, cultivos)
    static = cargar_static()
    fila, col = celda_de_zona(zona["lat"], zona["lon"], static["wrf_lats"], static["wrf_lons"])
    noche = noche_objetivo()
    with st.spinner("Calculando el aviso…"):
        res = calcular_noche(noche)
    t_noche, t_modelo, t_gfs = res["times"], res["t_modelo"][:, fila, col], res["t_gfs"][:, fila, col]
    r = evaluar_noche(t_modelo, t_noche, umbral)
except (ModeloNoDisponible,) as e:
    mostrar_error(f"El modelo no está disponible. {e}")
except (ZonaFueraDeGrilla, UmbralInvalido, SinPronostico, NocheNoCubierta) as e:
    mostrar_error(str(e))
except Exception:
    mostrar_error(MSJ_INTERNO)

try:
    if res["aviso"]:
        st.warning(res["aviso"])
    nombre_cultivo = umbral.get("nombre", cultivo)
    inicio_noche = a_hora_lima(f"{noche}00")
    st.subheader(f"Noche del {inicio_noche:%d/%m} al {inicio_noche + pd.Timedelta(days=1):%d/%m} · "
                 f"{zona['nombre']} · {nombre_cultivo}")

    colores = {"sin riesgo": st.success, "vigilancia": st.warning, "helada probable": st.error}
    colores[r["nivel"]](f"**{r['nivel'].capitalize()}** · {mensaje(r['nivel'], r, zona['nombre'])}")

    bajo = f"bajo {umbral['helada_c']:g} °C"
    c1, c2, c3 = st.columns(3)
    c1.metric("Nivel de riesgo", r["nivel"].capitalize())
    c2.metric(f"Inicio {bajo}", f"{r['inicio']:%H:%M}" if r["inicio"] else "No se espera")
    c3.metric(f"Duración {bajo}", f"{r['duracion_h']:g} h" if r["inicio"] else "—")

    st.markdown("**T mínima de la noche: nuestro modelo frente al GFS crudo**")
    tmin_gfs = float(np.min(t_gfs))
    d1, d2, d3 = st.columns(3)
    d1.metric("Modelo (3 km)", f"{r['tmin_c']:.1f} °C", help=f"Hacia las {r['hora_tmin']:%H:%M} (hora de Lima)")
    d2.metric("GFS crudo (27 km)", f"{tmin_gfs:.1f} °C",
              help="El pronóstico global interpolado a la celda, sin nuestro modelo")
    d3.metric("Diferencia", f"{r['tmin_c'] - tmin_gfs:+.1f} °C", help="Modelo menos GFS crudo")

    st.markdown("**Temperatura por hora**")
    st.dataframe(tabla_por_hora(t_noche, t_modelo, t_gfs, umbral["helada_c"]), hide_index=True)

    if umbral["es_referencia_general"]:
        st.caption(f"Umbral: referencia general de {umbral['helada_c']:g} °C, no específica para "
                   f"{nombre_cultivo.lower()} (fuente pendiente de validar).")
    else:
        st.caption(f"Umbral para {nombre_cultivo.lower()}: {umbral['helada_c']:g} °C · Fuente: "
                   f"{umbral['fuente']['institucion'].split(';')[0]} ({umbral['fuente']['anio']}).")
    st.caption(f"Celda de 3 km: fila {fila}, columna {col} · elevación media {zona.get('elevacion_dem_m', '—')} m. "
               "El GFS crudo es el pronóstico global (unos 27 km) interpolado a esa celda.")
    st.caption(AVISO_LIMITES)
except Exception:
    mostrar_error(MSJ_INTERNO)

formulario_confirmacion(clave, zonas)

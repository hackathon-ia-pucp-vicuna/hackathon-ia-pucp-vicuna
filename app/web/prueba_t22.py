"""
T22 · Prueba de despliegue gratuito con PyTorch en CPU (HU-11).

App mínima para medir, en el hosting gratuito, el tiempo de arranque y la memoria
de una red del mismo tamaño que la del starter (`Net`, train/starter_notebook.ipynb).

- Pesos ALEATORIOS con semilla fija: no son el modelo de Kaggle ni predicen nada real.
- Entrada SINTÉTICA de 24 horas: no usa la data oficial (no va al hosting) ni Open-Meteo.

Ejecutar en local:
    pip install -r app/web/requirements.txt
    streamlit run app/web/streamlit_app.py
"""
import os
import platform
import sys
import time

import numpy as np
import streamlit as st

T_SCRIPT = time.time()


def proceso_inicio():
    """Hora de inicio del proceso (Linux, vía /proc); None si no se puede leer."""
    try:
        return os.stat(f"/proc/{os.getpid()}").st_ctime
    except OSError:
        return None


def memoria_mb():
    """(RSS actual, RSS máximo) en MB. En Linux ru_maxrss viene en KB."""
    actual = None
    try:
        with open("/proc/self/status") as f:
            for linea in f:
                if linea.startswith("VmRSS:"):
                    actual = int(linea.split()[1]) / 1024
    except OSError:
        pass
    try:
        import resource                                        # solo existe en Linux/macOS
        maximo = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        maximo = None
    return actual, maximo


@st.cache_resource
def cargar_y_medir():
    """Importa torch, crea la red y corre 24 horas de inferencia. Se ejecuta una vez por proceso."""
    t0 = time.time()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    t_import = time.time() - t0

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # Grilla WRF (70x35) dentro de la grilla GFS (17x13) y DEM sintéticos, con las formas reales.
    gy, gx = torch.meshgrid(torch.linspace(-0.9, 0.9, 70), torch.linspace(-0.9, 0.9, 35), indexing="ij")
    grid = torch.stack([gx, gy], -1)[None]                     # (1, 70, 35, 2)
    dem = torch.randn(1, 1, 70, 35)

    class Net(nn.Module):
        """Copia de la arquitectura del starter; grilla y DEM como buffers en vez de globales."""

        def __init__(self, w=64):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(14, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU())
            self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
            self.dec = nn.Sequential(
                nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
                nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
                nn.Conv2d(w, 4, 1))
            self.register_buffer("grid", grid)
            self.register_buffer("dem", dem)

        def forward(self, x, h):                               # x: (B,3,17,13,4), h: (B,) hora UTC
            B = x.shape[0]
            x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
            ang = 2 * np.pi * h / 24
            hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
            f = self.enc(torch.cat([x, hh], 1))
            f = F.grid_sample(f, self.grid.expand(B, -1, -1, -1), align_corners=True)
            f = torch.cat([f, self.dem.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
            return self.dec(f).permute(0, 2, 3, 1)             # (B, 70, 35, 4)

    t1 = time.time()
    modelo = Net().eval()
    t_modelo = time.time() - t1
    n_params = sum(p.numel() for p in modelo.parameters())

    x = torch.randn(24, 3, 17, 13, 4)
    h = torch.arange(24, dtype=torch.float32)
    t2 = time.time()
    with torch.no_grad():
        y = modelo(x, h)
    t_inferencia = time.time() - t2

    with torch.no_grad():                                       # segunda pasada: ya "en caliente"
        t3 = time.time()
        modelo(x, h)
        t_inferencia_2 = time.time() - t3

    return {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "hilos": torch.get_num_threads(),
        "params": n_params,
        "pesos_mb": n_params * 4 / 1024**2,
        "salida": tuple(y.shape),
        "t_import": t_import,
        "t_modelo": t_modelo,
        "t_inferencia": t_inferencia,
        "t_inferencia_2": t_inferencia_2,
        "t_total_carga": time.time() - t0,
        "proceso_inicio": proceso_inicio(),
        "listo_en": time.time(),
    }


st.set_page_config(page_title="Helada Andina · prueba CPU", page_icon="❄️")
st.title("Helada Andina · prueba de despliegue (T22)")
st.caption("Red del tamaño del starter con pesos aleatorios y entrada sintética. No es un pronóstico.")

with st.spinner("Importando PyTorch y corriendo 24 horas de inferencia en CPU…"):
    m = cargar_y_medir()

rss, rss_max = memoria_mb()
arranque = m["listo_en"] - m["proceso_inicio"] if m["proceso_inicio"] else None

c1, c2, c3 = st.columns(3)
c1.metric("Arranque del proceso → modelo listo", f"{arranque:.1f} s" if arranque else "n/d")
c2.metric("Memoria RSS actual", f"{rss:.0f} MB" if rss else "n/d")
c3.metric("Memoria RSS máxima", f"{rss_max:.0f} MB" if rss_max else "n/d")

st.subheader("Detalle")
st.table({
    "Medida": [
        "Importar torch", "Crear la red", "Inferencia 24 h (1.ª)", "Inferencia 24 h (2.ª)",
        "Carga total (import + red + 2 inferencias)", "Parámetros", "Tamaño de pesos (float32)",
        "Forma de salida", "torch", "CUDA disponible", "Hilos de CPU", "Python", "Plataforma",
    ],
    "Valor": [
        f"{m['t_import']:.2f} s", f"{m['t_modelo']:.3f} s", f"{m['t_inferencia']:.3f} s",
        f"{m['t_inferencia_2']:.3f} s", f"{m['t_total_carga']:.2f} s", f"{m['params']:,}",
        f"{m['pesos_mb']:.2f} MB", str(m["salida"]), m["torch"], str(m["cuda"]), str(m["hilos"]),
        sys.version.split()[0], platform.platform(),
    ],
})
st.caption(
    "Las medidas de carga se toman una vez por proceso (caché). Para medir el arranque en frío, "
    "abre la app después de que el hosting la haya dormido o tras un *Reboot app*."
)

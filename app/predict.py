"""
Único punto de entrada al modelo para el producto: `predecir()`.

Usa la MISMA red y los MISMOS pesos que la submission de Kaggle (copia de `Net` del starter).
`tests/test_paridad.py` comprueba que esta copia da la misma salida que la red del notebook (T12).

Archivos que necesita (ninguno se genera desde aquí):
  models/modelo.pt          state_dict de la red (T03)
  models/estadisticas.npz   mu_x, sd_x, clim (24,70,35,4), anom_sd del entrenamiento; no va al repo,
                            se genera con `python train/generar_estadisticas.py` (T03)
  data/static.npz           DEM y grillas de la competencia (se baja de Kaggle; no va al repo)

Medir el tiempo en CPU para 24 horas (con los archivos reales):
  python app/predict.py
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_PESOS = RAIZ / "models" / "modelo.pt"
RUTA_ESTADISTICAS = RAIZ / "models" / "estadisticas.npz"
RUTA_STATIC = RAIZ / "data" / "static.npz"


class ModeloNoDisponible(RuntimeError):
    """Falta un archivo del modelo; el mensaje está en español para mostrarlo al usuario."""


def _exigir(ruta, que, como=""):
    if not Path(ruta).exists():
        raise ModeloNoDisponible(f"No se encontró {que} ({Path(ruta).name}). {como}".strip())


@lru_cache(maxsize=1)
def cargar_static():
    _exigir(RUTA_STATIC, "el archivo de relieve y grillas de la competencia")
    st = np.load(RUTA_STATIC)
    return {k: st[k] for k in st.files}


def construir_red(st):
    """
    Red `Net` del starter con la grilla y el DEM de `st` (dict con gfs_lats, gfs_lons, wrf_lats,
    wrf_lons, dem). Pesos sin cargar. Separada de _cargar() para poder probar la paridad.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Igual que en el starter: grilla WRF dentro de la grilla GFS (para grid_sample) y DEM normalizado.
    glat, glon = st["gfs_lats"], st["gfs_lons"]          # GFS: fila 0 = norte
    wlat, wlon = st["wrf_lats"], st["wrf_lons"]          # WRF: fila 0 = sur
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    grid = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32)
    dem = st["dem"]
    dem_n = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32)[None, None]

    class Net(nn.Module):
        """Copia de `Net` del starter; GRID y DEM pasan de globales a buffers no persistentes."""

        def __init__(self, w=64):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(14, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU())
            self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
            self.dec = nn.Sequential(
                nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
                nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
                nn.Conv2d(w, 4, 1))
            self.register_buffer("grid", grid, persistent=False)
            self.register_buffer("dem", dem_n, persistent=False)

        def forward(self, x, h):                     # x: (B,3,17,13,4), h: (B,) hora UTC
            B = x.shape[0]
            x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
            ang = 2 * np.pi * h / 24
            hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
            f = self.enc(torch.cat([x, hh], 1))
            f = F.grid_sample(f, self.grid.expand(B, -1, -1, -1), align_corners=True)
            f = torch.cat([f, self.dem.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
            return self.dec(f).permute(0, 2, 3, 1)  # (B,70,35,4) anomalía normalizada

    return Net()


@lru_cache(maxsize=1)
def _cargar():
    import torch

    _exigir(RUTA_PESOS, "los pesos del modelo")
    _exigir(RUTA_ESTADISTICAS, "las estadísticas de normalización",
            "Genéralas con: python train/generar_estadisticas.py")
    est = np.load(RUTA_ESTADISTICAS)
    modelo = construir_red(cargar_static())
    modelo.load_state_dict(torch.load(RUTA_PESOS, map_location="cpu", weights_only=True))
    modelo.eval()
    stats = {k: est[k].astype(np.float32) for k in ("mu_x", "sd_x", "clim", "anom_sd")}
    return torch, modelo, stats


def predecir(Xw, horas_utc, lote=64):
    """
    Xw        : (N, 3, 17, 13, 4) ventanas [t-2, t-1, t] del GFS en unidades físicas
                (RH %, T K, U m/s, V m/s); fila 0 = norte.
    horas_utc : (N,) hora UTC (0–23) de cada ventana.
    Devuelve  : (N, 70, 35, 4) en unidades físicas; fila 0 = sur (como Y en Kaggle).
    """
    torch, modelo, s = _cargar()
    Xw = np.asarray(Xw, dtype=np.float32)
    h_all = np.asarray(horas_utc, dtype=np.int64)
    if Xw.ndim != 5 or Xw.shape[1:] != (3, 17, 13, 4):
        raise ValueError(f"Se esperaban ventanas (N, 3, 17, 13, 4) y llegó {Xw.shape}.")
    if len(h_all) != len(Xw):
        raise ValueError("Debe haber una hora UTC por ventana.")
    salida = []
    with torch.no_grad():
        for i in range(0, len(Xw), lote):
            x = torch.from_numpy((Xw[i:i + lote] - s["mu_x"]) / s["sd_x"])
            h = h_all[i:i + lote]
            anom = modelo(x, torch.from_numpy(h).float()).numpy()
            salida.append(s["clim"][h] + anom * s["anom_sd"])
    if not salida:
        return np.empty((0, 70, 35, 4), np.float32)
    return np.concatenate(salida)


def medir_24h(repeticiones=3):
    """Segundos (mejor de `repeticiones`) para predecir 24 horas en CPU, con entrada típica."""
    import time

    _, _, s = _cargar()
    rng = np.random.default_rng(0)
    Xw = (s["mu_x"] + s["sd_x"] * rng.standard_normal((24, 3, 17, 13, 4))).astype(np.float32)
    horas = np.arange(24)
    predecir(Xw[:1], horas[:1])                      # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        predecir(Xw, horas)
        tiempos.append(time.perf_counter() - t0)
    return min(tiempos)


if __name__ == "__main__":
    print(f"predecir() en CPU · 24 horas: {medir_24h():.3f} s")

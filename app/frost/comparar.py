"""
GFS crudo en la grilla de 3 km, para compararlo con el modelo (T17 · HU-04).

Es la misma interpolación bilineal que usa T36 (`scripts/evaluar_tmin_nocturna.py`), donde se midió
el error de la T mínima nocturna del GFS crudo y del modelo frente al WRF; `tests/test_comparar.py`
comprueba que ambas copias dan lo mismo. No usa el modelo: es la referencia "sin reducción de escala".
"""
import numpy as np


def interpolar_gfs(X, st):
    """Bilineal (N,17,13) GFS (fila 0 = norte) → (N,70,35) WRF (fila 0 = sur)."""
    glat, glon, wlat, wlon = st["gfs_lats"], st["gfs_lons"], st["wrf_lats"], st["wrf_lons"]
    fy = (wlat - glat[0]) / (glat[1] - glat[0])          # índice fraccional de fila GFS
    fx = (wlon - glon[0]) / (glon[1] - glon[0])
    y0 = np.clip(np.floor(fy).astype(int), 0, len(glat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(glon) - 2)
    wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
    Y0, X0 = y0[:, None], x0[None, :]
    return ((1 - wy) * (1 - wx) * X[:, Y0, X0] + (1 - wy) * wx * X[:, Y0, X0 + 1]
            + wy * (1 - wx) * X[:, Y0 + 1, X0] + wy * wx * X[:, Y0 + 1, X0 + 1])


def t_gfs_crudo_c(X, st):
    """T del GFS crudo (N,17,13,4 en unidades físicas) interpolada a 3 km, en °C: (N,70,35)."""
    return interpolar_gfs(np.asarray(X, dtype=np.float64)[..., 1], st) - 273.15

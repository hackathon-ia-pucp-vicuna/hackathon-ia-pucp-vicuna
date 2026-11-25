"""
Zonas del Callejón de Huaylas → celda de la grilla de 3 km y elevación del DEM (T14 · HU-07).

Orientación (igual que en Kaggle):
  GFS (X, 17×13): fila 0 = norte · WRF (Y, 70×35) y DEM: fila 0 = sur · columna 0 = oeste.

Completar zones.yaml con fila, columna y elevación del DEM (necesita data/static.npz):
  python app/frost/zonas.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):                       # ejecución directa: python app/frost/zonas.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from frost.rules import FROST, ZonaFueraDeGrilla, celda_de_zona  # noqa: F401
else:
    from .rules import FROST, ZonaFueraDeGrilla, celda_de_zona  # noqa: F401

RUTA_ZONAS = FROST / "zones.yaml"


def comprobar_orientacion(st):
    """Falla si las grillas no tienen la orientación esperada (GFS norte→sur, WRF sur→norte)."""
    if not np.all(np.diff(st["gfs_lats"]) < 0):
        raise ValueError("gfs_lats debe ir de norte a sur (fila 0 = norte).")
    if not np.all(np.diff(st["wrf_lats"]) > 0):
        raise ValueError("wrf_lats debe ir de sur a norte (fila 0 = sur).")
    if not (np.all(np.diff(st["gfs_lons"]) > 0) and np.all(np.diff(st["wrf_lons"]) > 0)):
        raise ValueError("Las longitudes deben ir de oeste a este (columna 0 = oeste).")
    if st["dem"].shape != (len(st["wrf_lats"]), len(st["wrf_lons"])):
        raise ValueError("El DEM debe tener la forma de la grilla WRF (70×35).")


def celda_y_elevacion(lat, lon, st):
    """(fila, columna, elevación en m) en la grilla WRF/DEM. Lanza ZonaFueraDeGrilla si cae fuera."""
    comprobar_orientacion(st)
    fila, col = celda_de_zona(lat, lon, st["wrf_lats"], st["wrf_lons"])
    return fila, col, float(st["dem"][fila, col])


def completar_zonas(zonas, st):
    """Copia de `zonas` con fila, columna y elevacion_dem_m calculadas a partir de `st`."""
    salida = {}
    for clave, z in zonas.items():
        fila, col, elev = celda_y_elevacion(z["lat"], z["lon"], st)
        salida[clave] = {**z, "fila": fila, "columna": col, "elevacion_dem_m": round(elev)}
    return salida


def _guardar(zonas, ruta=RUTA_ZONAS):
    texto = ruta.read_text(encoding="utf-8")
    cabecera = "".join(l for l in texto.splitlines(keepends=True) if l.startswith("#"))
    cuerpo = yaml.safe_dump(zonas, allow_unicode=True, sort_keys=False)
    ruta.write_text(cabecera + "\n" + cuerpo, encoding="utf-8")


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[2]
    st_npz = np.load(raiz / "data" / "static.npz")
    st = {k: st_npz[k] for k in st_npz.files}
    zonas = completar_zonas(yaml.safe_load(RUTA_ZONAS.read_text(encoding="utf-8")), st)
    _guardar(zonas)
    for z in zonas.values():
        print(f"{z['nombre']:<10} fila {z['fila']:>2}  columna {z['columna']:>2}  DEM {z['elevacion_dem_m']} m")

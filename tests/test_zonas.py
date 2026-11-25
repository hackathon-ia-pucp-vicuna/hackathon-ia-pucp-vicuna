"""Pruebas de zonas, orientación de grillas y extracción del DEM (T14 · HU-07). Sin red ni PyTorch."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))

from frost.zonas import (RUTA_ZONAS, ZonaFueraDeGrilla, celda_y_elevacion,  # noqa: E402
                         completar_zonas, comprobar_orientacion)
from gfs_forecast import LATS  # noqa: E402

RUTA_STATIC = RAIZ / "data" / "static.npz"
ZONAS = yaml.safe_load(RUTA_ZONAS.read_text(encoding="utf-8"))


def static_sintetico():
    """Geometría como la de la competencia; el DEM crece hacia el norte y el este para rastrear la orientación."""
    wrf_lats = np.linspace(-10.4, -8.3, 70)           # fila 0 = sur
    wrf_lons = np.linspace(-78.3, -77.2, 35)          # columna 0 = oeste
    dem = 1000 + 10 * np.arange(70)[:, None] + np.arange(35)[None, :]
    return {"gfs_lats": LATS.astype(float), "gfs_lons": np.round(np.arange(-79.0, -76.0 + 1e-9, 0.25), 2),
            "wrf_lats": wrf_lats, "wrf_lons": wrf_lons, "dem": dem.astype(float)}


def test_hay_al_menos_tres_localidades_con_coordenadas_en_el_callejon():
    assert len(ZONAS) >= 3
    for z in ZONAS.values():
        assert {"nombre", "lat", "lon", "fila", "columna", "elevacion_dem_m"} <= set(z)
        assert -9.9 < z["lat"] < -8.8 and -78.0 < z["lon"] < -77.3   # Callejón de Huaylas


def test_gfs_de_open_meteo_tiene_fila_cero_al_norte():
    assert LATS[0] > LATS[-1] and LATS[0] == -7.25


def test_fila_cero_al_sur_en_wrf_y_dem():
    st = static_sintetico()
    norte = celda_y_elevacion(-8.31, -77.6, st)
    sur = celda_y_elevacion(-10.39, -77.6, st)
    assert sur[0] == 0 and norte[0] == 69             # fila 0 = sur, última fila = norte
    assert sur[2] < norte[2]                          # el DEM sintético crece hacia el norte


def test_columna_cero_al_oeste():
    st = static_sintetico()
    oeste = celda_y_elevacion(-9.3, -78.29, st)
    este = celda_y_elevacion(-9.3, -77.21, st)
    assert oeste[1] == 0 and este[1] == 34


def test_celda_y_elevacion_coinciden_con_el_dem():
    st = static_sintetico()
    fila, col, elev = celda_y_elevacion(ZONAS["huaraz"]["lat"], ZONAS["huaraz"]["lon"], st)
    assert elev == st["dem"][fila, col]
    assert abs(st["wrf_lats"][fila] - ZONAS["huaraz"]["lat"]) <= np.diff(st["wrf_lats"])[0] / 2 + 1e-9


def test_huaraz_esta_al_sur_de_yungay():
    st = static_sintetico()
    z = completar_zonas(ZONAS, st)
    assert z["huaraz"]["fila"] < z["carhuaz"]["fila"] < z["yungay"]["fila"]
    assert all(isinstance(v["elevacion_dem_m"], int) for v in z.values())


@pytest.mark.parametrize("lat,lon", [(-12.0, -77.5), (-9.3, -76.0), (-6.0, -77.5), (-9.3, -79.5)])
def test_zona_fuera_de_la_grilla_da_error_claro(lat, lon):
    with pytest.raises(ZonaFueraDeGrilla, match="fuera del área"):
        celda_y_elevacion(lat, lon, static_sintetico())


def test_orientacion_invertida_se_detecta():
    st = static_sintetico()
    with pytest.raises(ValueError, match="sur a norte"):
        comprobar_orientacion({**st, "wrf_lats": st["wrf_lats"][::-1]})
    with pytest.raises(ValueError, match="norte a sur"):
        comprobar_orientacion({**st, "gfs_lats": st["gfs_lats"][::-1]})
    with pytest.raises(ValueError, match="70×35"):
        comprobar_orientacion({**st, "dem": st["dem"].T})


@pytest.mark.skipif(not RUTA_STATIC.exists(), reason="falta data/static.npz (se baja de Kaggle)")
def test_zones_yaml_coincide_con_el_static_real():
    d = np.load(RUTA_STATIC)
    st = {k: d[k] for k in d.files}
    comprobar_orientacion(st)
    for clave, z in completar_zonas(ZONAS, st).items():
        assert (ZONAS[clave]["fila"], ZONAS[clave]["columna"]) == (z["fila"], z["columna"]), \
            "zones.yaml desactualizado: ejecuta python app/frost/zonas.py"
        assert ZONAS[clave]["elevacion_dem_m"] == z["elevacion_dem_m"]
        assert 1500 < z["elevacion_dem_m"] < 4500    # los pueblos del Callejón están entre ~2,200 y ~3,100 m

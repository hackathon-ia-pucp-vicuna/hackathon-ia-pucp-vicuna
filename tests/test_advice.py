"""Pruebas de app/frost/advice.py (T16 · HU-03)."""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from frost.advice import AVISO_LIMITES, PLANTILLAS, campos, mensaje  # noqa: E402

LIMA = timezone(timedelta(hours=-5))
NIVELES = ("sin riesgo", "vigilancia", "helada probable")
RESULTADO = {
    "tmin_c": -2.3,
    "hora_tmin": datetime(2024, 6, 16, 4, tzinfo=LIMA),
    "inicio": datetime(2024, 6, 15, 23, tzinfo=LIMA),
    "duracion_h": 5,
}


def oraciones(texto):
    return [o for o in re.split(r"(?<=[.!?])\s+", texto.strip()) if o]


@pytest.mark.parametrize("nivel", NIVELES)
@pytest.mark.parametrize("resultado,zona", [(None, None), (RESULTADO, "Carhuaz"), ({"tmin_c": 1.0}, "Yungay")])
def test_sin_campos_sin_rellenar(nivel, resultado, zona):
    m = mensaje(nivel, resultado, zona)
    assert "{" not in m and "}" not in m
    assert "None" not in m and "nan" not in m.lower()


@pytest.mark.parametrize("nivel", NIVELES)
def test_maximo_tres_oraciones_con_recomendacion_e_incertidumbre(nivel):
    for m in (mensaje(nivel), mensaje(nivel, RESULTADO, "Huaraz")):
        partes = oraciones(m)
        assert 1 <= len(partes) <= 3, partes
        assert PLANTILLAS[nivel]["recomendacion"] in m
        assert "pronóstico" in partes[-1] and "°C" in partes[-1]


def test_hay_un_mensaje_por_nivel_sin_textos_vacios():
    assert set(PLANTILLAS) == set(NIVELES)
    for p in PLANTILLAS.values():
        assert all(isinstance(v, str) and v.strip() for v in p.values())
        assert not campos(p["general"]) and not campos(p["recomendacion"])


def test_helada_con_datos_muestra_hora_duracion_y_minima():
    m = mensaje("helada probable", RESULTADO, "Carhuaz")
    assert "Carhuaz" in m and "23:00" in m and "5 h" in m and "-2 °C" in m


def test_minima_cercana_a_cero_no_muestra_menos_cero():
    m = mensaje("vigilancia", {"tmin_c": -0.3, "hora_tmin": RESULTADO["hora_tmin"]}, "Huaraz")
    assert "-0" not in m and " 0 °C" in m


def test_duracion_decimal_con_coma():
    m = mensaje("helada probable", {**RESULTADO, "duracion_h": 1.5}, "Carhuaz")
    assert "1,5 h" in m


def test_aviso_declara_limites_del_modelo():
    for palabra in ("lluvia", "nubosidad", "radiación", "ráfagas"):
        assert palabra in AVISO_LIMITES


def test_nivel_desconocido_falla():
    with pytest.raises(KeyError):
        mensaje("muy alto")

"""Pruebas de app/frost/rules.py (T15 · HU-02). Sin red ni PyTorch."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from frost.rules import (NIVELES, UmbralInvalido, a_hora_lima, evaluar_noche,  # noqa: E402
                         horas_de_la_noche, nivel_de_riesgo, noche_objetivo, umbral_de_cultivo)

UMBRAL = {"helada_c": 0.0, "vigilancia_c": 2.0}
# Noche del 15/06/2024 en hora de Lima: 00–12 UTC del 16/06 (13 horas)
HORAS = [f"20240616{h:02d}" for h in range(13)]


def noche(*temps):
    """Temperaturas por hora desde las 19:00 Lima; completa el resto con 8 °C."""
    t = list(temps) + [8.0] * (len(HORAS) - len(temps))
    return np.array(t, dtype=float)


def test_sin_riesgo_noche_templada():
    r = evaluar_noche(noche(), HORAS, UMBRAL)
    assert r["nivel"] == "sin riesgo"
    assert r["inicio"] is None and r["fin"] is None
    assert r["duracion_h"] == 0 and r["tramos"] == []


def test_vigilancia_sin_cruce_del_umbral():
    r = evaluar_noche(noche(6, 4, 2.0, 1.5, 3), HORAS, UMBRAL)
    assert r["nivel"] == "vigilancia"
    assert r["inicio"] is None and r["duracion_h"] == 0
    assert r["tmin_c"] == 1.5


def test_borde_exacto_en_cero_es_helada():
    r = evaluar_noche(noche(5, 3, 1, 0.0, 1), HORAS, UMBRAL)
    assert r["nivel"] == "helada probable"
    assert r["duracion_h"] == 1
    assert r["inicio"] == a_hora_lima("2024061603")   # 22:00 Lima


def test_borde_justo_encima_de_cero_no_es_helada():
    r = evaluar_noche(noche(5, 3, 1, 0.01, 1), HORAS, UMBRAL)
    assert r["nivel"] == "vigilancia"
    assert r["inicio"] is None


def test_borde_exacto_en_vigilancia():
    assert nivel_de_riesgo(2.0, UMBRAL) == "vigilancia"
    assert nivel_de_riesgo(2.0001, UMBRAL) == "sin riesgo"


def test_un_cruce_inicio_fin_y_duracion_en_hora_de_lima():
    t = noche(6, 4, 2, 1, -0.5, -1.5, -2, -1, 0.5, 2)
    r = evaluar_noche(t, HORAS, UMBRAL)
    assert r["nivel"] == "helada probable"
    assert r["inicio"].hour == 23 and r["inicio"].utcoffset().total_seconds() == -5 * 3600
    assert r["fin"].hour == 2                       # 02:00 Lima del día siguiente
    assert r["duracion_h"] == 4
    assert len(r["tramos"]) == 1
    assert r["tmin_c"] == -2.0 and r["hora_tmin"].hour == 1


def test_varios_cruces_suman_duracion_y_conservan_tramos():
    t = noche(3, -1, -0.5, 1, 2, -0.2, 1, -1, -2, -3, 0.5)
    r = evaluar_noche(t, HORAS, UMBRAL)
    assert [tr["horas"] for tr in r["tramos"]] == [2, 1, 3]
    assert r["duracion_h"] == 6
    assert r["inicio"].hour == 20                   # primer cruce
    assert r["fin"].hour == 4                       # fin del último tramo


def test_toda_la_noche_bajo_cero():
    r = evaluar_noche(np.full(13, -1.0), HORAS, UMBRAL)
    assert r["duracion_h"] == 13
    assert r["inicio"].hour == 19 and r["fin"].hour == 7


def test_horas_desordenadas_se_ordenan():
    t = noche(5, 3, -1, -2, 4)
    orden = np.random.default_rng(0).permutation(13)
    r1 = evaluar_noche(t, HORAS, UMBRAL)
    r2 = evaluar_noche(t[orden], np.array(HORAS)[orden], UMBRAL)
    assert r1["inicio"] == r2["inicio"] and r1["duracion_h"] == r2["duracion_h"] == 2


def test_pronostico_cada_tres_horas():
    horas = ["2024061600", "2024061603", "2024061606", "2024061609", "2024061612"]
    r = evaluar_noche([4, -1, -2, 1, 3], horas, UMBRAL)
    assert r["duracion_h"] == 6                     # 2 pasos de 3 h
    assert r["inicio"].hour == 22


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        evaluar_noche([], [], UMBRAL)
    with pytest.raises(ValueError):
        evaluar_noche([1, 2], HORAS[:3], UMBRAL)
    with pytest.raises(ValueError):
        evaluar_noche([1, np.nan], HORAS[:2], UMBRAL)


def test_umbral_se_lee_del_yaml():
    u = umbral_de_cultivo("papa")
    assert {"helada_c", "vigilancia_c", "fuente", "validado"} <= set(u)
    assert u["helada_c"] <= u["vigilancia_c"]


def test_umbral_distinto_cambia_el_nivel():
    t = noche(5, 3, 1.5, 2)
    assert evaluar_noche(t, HORAS, UMBRAL)["nivel"] == "vigilancia"
    assert evaluar_noche(t, HORAS, {"helada_c": 2.0, "vigilancia_c": 4.0})["nivel"] == NIVELES[2]


def test_umbral_invalido():
    with pytest.raises(UmbralInvalido):
        umbral_de_cultivo("quinua", {"papa": UMBRAL})
    with pytest.raises(UmbralInvalido):
        umbral_de_cultivo("papa", {"papa": {"helada_c": 3.0, "vigilancia_c": 1.0}})
    with pytest.raises(UmbralInvalido):
        umbral_de_cultivo("papa", {"papa": {"helada_c": None, "vigilancia_c": 1.0}})


def test_noche_objetivo_y_horas_de_la_noche():
    # 06:59 Lima → noche en curso; 07:00 Lima → noche siguiente
    assert noche_objetivo(datetime(2024, 6, 16, 11, 59, tzinfo=timezone.utc)) == "20240616"
    assert noche_objetivo(datetime(2024, 6, 16, 12, 0, tzinfo=timezone.utc)) == "20240617"
    m = horas_de_la_noche(["2024061523", "2024061600", "2024061612", "2024061613"], "20240616")
    assert m.tolist() == [False, True, True, False]

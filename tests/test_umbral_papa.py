"""Umbral de helada para la papa con fuente y validación (T07 · HU-01). Sin red."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from frost.rules import CAMPOS_FUENTE, UmbralInvalido, cargar_yaml, evaluar_noche, umbral_de_cultivo  # noqa: E402

PAPA = cargar_yaml("thresholds.yaml")["papa"]
HORAS = [f"20240616{h:02d}" for h in range(4)]


def test_yaml_tiene_fuente_completa_y_campo_validado():
    assert isinstance(PAPA["validado"], bool)
    for campo in CAMPOS_FUENTE + ("cita",):
        assert PAPA["fuente"].get(campo), f"falta fuente.{campo}"
    assert PAPA["fuente"]["enlace"].startswith("https://")
    assert isinstance(PAPA["fuente"]["anio"], int)
    assert PAPA["referencia_general_c"] == 0.0
    assert PAPA["helada_cultivo_c"] <= PAPA["vigilancia_c"]


def test_sin_validar_la_app_solo_usa_cero_grados():
    u = umbral_de_cultivo("papa", {"papa": {**PAPA, "validado": False}})
    assert u["helada_c"] == 0.0 and u["es_referencia_general"] is True


def test_validado_usa_el_umbral_del_cultivo():
    u = umbral_de_cultivo("papa", {"papa": {**PAPA, "validado": True}})
    assert u["helada_c"] == PAPA["helada_cultivo_c"] and u["es_referencia_general"] is False


def test_el_umbral_validado_cambia_el_nivel_de_la_noche():
    temps = [3.0, 1.0, -0.5, 0.5]                    # mínima -0.5 °C: helada a 0 °C, vigilancia a -1 °C
    sin_validar = umbral_de_cultivo("papa", {"papa": {**PAPA, "validado": False}})
    validado = umbral_de_cultivo("papa", {"papa": {**PAPA, "validado": True}})
    assert evaluar_noche(temps, HORAS, sin_validar)["nivel"] == "helada probable"
    assert evaluar_noche(temps, HORAS, validado)["nivel"] == "vigilancia"


@pytest.mark.parametrize("campo", CAMPOS_FUENTE)
def test_validado_sin_fuente_completa_falla(campo):
    fuente = {**PAPA["fuente"], campo: None}
    with pytest.raises(UmbralInvalido, match=campo):
        umbral_de_cultivo("papa", {"papa": {**PAPA, "fuente": fuente, "validado": True}})


def test_validado_sin_fuente_falla():
    with pytest.raises(UmbralInvalido):
        umbral_de_cultivo("papa", {"papa": {**PAPA, "fuente": None, "validado": True}})


def test_umbral_actual_del_repo_es_coherente_con_su_validacion():
    u = umbral_de_cultivo("papa")
    esperado = PAPA["helada_cultivo_c"] if PAPA["validado"] else PAPA["referencia_general_c"]
    assert u["helada_c"] == esperado

"""Ventanas sin huecos de horas (T11 · HU-09). Sin red ni PyTorch."""
import ast
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "train"))

import ventanas as v  # noqa: E402

NOTEBOOK = RAIZ / "train" / "t11_ventanas.ipynb"
STARTER = RAIZ / "train" / "starter_notebook.ipynb"
RUTA_TRAIN = RAIZ / "data" / "train.npz"
RUTA_TEST = RAIZ / "data" / "test.npz"


def marcas(*desfases_h, inicio="2024061600"):
    t0 = datetime.strptime(inicio, "%Y%m%d%H")
    return np.array([(t0 + timedelta(hours=h)).strftime("%Y%m%d%H") for h in desfases_h])


# Huecos: 3→6 (faltan 4 y 5) y 9→30
CON_HUECOS = (0, 1, 2, 3, 6, 7, 8, 9, 30)


def grilla(n):
    """X sintético donde cada hora tiene un valor distinto, para rastrear qué hora entró en cada ventana."""
    X = np.zeros((n, 2, 2, 4), np.float32)
    X[...] = np.arange(n, dtype=np.float32)[:, None, None, None]
    return X


def assert_completas_son_consecutivas(times, idx, completa):
    fechas = [datetime.strptime(str(t), "%Y%m%d%H") for t in times]
    for i in np.flatnonzero(completa):
        assert [fechas[j] for j in idx[i]] == [fechas[i] - timedelta(hours=h) for h in (2, 1, 0)]


def test_ventanas_que_cruzan_un_hueco_quedan_marcadas():
    W, completa = v.armar_ventanas(grilla(9), marcas(*CON_HUECOS))
    assert W.shape == (9, 3, 2, 2, 4) and W.dtype == np.float32
    assert completa.tolist() == [False, False, True, True, False, False, True, True, False]


def test_hueco_de_una_sola_hora():
    _, completa = v.indices_ventanas(marcas(0, 1, 2, 4, 5, 6))      # falta la hora 3
    assert completa.tolist() == [False, False, True, False, False, True]


def test_la_hora_que_falta_se_rellena_con_t_y_no_con_la_vecina_del_hueco():
    W, _ = v.armar_ventanas(grilla(9), marcas(*CON_HUECOS))
    assert W[4, :, 0, 0, 0].tolist() == [4, 4, 4]       # primera hora tras el hueco: nada de la hora 3
    assert W[5, :, 0, 0, 0].tolist() == [5, 4, 5]       # falta t-2; t-1 sí existe


def test_ninguna_ventana_completa_cruza_un_hueco_con_datos_desordenados():
    rng = np.random.default_rng(0)
    desfases = rng.permutation(rng.choice(400, size=250, replace=False))
    t = marcas(*desfases.tolist())
    idx, completa = v.indices_ventanas(t)
    assert completa.any() and (~completa).any()
    assert_completas_son_consecutivas(t, idx, completa)
    existentes = set(desfases.tolist())
    assert completa.tolist() == [(d - 1 in existentes) and (d - 2 in existentes) for d in desfases.tolist()]


def test_entradas_invalidas_fallan():
    with pytest.raises(ValueError, match="repetidas"):
        v.indices_ventanas(marcas(0, 1, 1))
    with pytest.raises(ValueError, match="horas"):
        v.armar_ventanas(grilla(3), marcas(0, 1))


def celdas(ruta):
    return ["".join(c["source"]) for c in json.loads(ruta.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def test_las_ventanas_son_identicas_a_las_del_starter():
    pd = pytest.importorskip("pandas")
    codigo = next(c for c in celdas(STARTER) if "def windows(" in c)
    funcion = next(n for n in ast.parse(codigo).body if isinstance(n, ast.FunctionDef) and n.name == "windows")
    ns = {"np": np, "pd": pd}
    exec(ast.get_source_segment(codigo, funcion), ns)
    t = marcas(*CON_HUECOS)
    np.testing.assert_array_equal(v.armar_ventanas(grilla(9), t)[0], ns["windows"](grilla(9), t))


@pytest.mark.parametrize("modulo", ["validacion.py", "ventanas.py"])
def test_el_notebook_lleva_la_copia_exacta_del_modulo(modulo):
    copias = [c for c in celdas(NOTEBOOK) if c.startswith(f"# COPIA EXACTA de train/{modulo}")]
    assert len(copias) == 1
    assert copias[0].split("\n", 1)[1] == (RAIZ / "train" / modulo).read_text(encoding="utf-8").rstrip("\n")


def test_el_notebook_usa_la_red_del_experimento_t08():
    def red(ruta):
        return [c for c in celdas(ruta) if "class Net" in c]

    assert red(NOTEBOOK) == red(RAIZ / "train" / "t08_experimentos.ipynb")
    assert len(red(NOTEBOOK)) == 1


def test_el_notebook_excluye_las_incompletas_solo_del_entrenamiento():
    codigo = "\n".join(celdas(NOTEBOOK))
    assert "ramas = {'todas': trn, 'completas': trn & comp_tr}" in codigo
    assert "D = datos_de(trn, usadas)" in codigo           # climatología y normalización iguales en las dos ramas
    assert "predecir_lote(entrenar(D, EPOCAS, PERDIDA, semilla), D, Wtr[val], ttr[val])" in codigo


def test_el_notebook_es_sintaxis_valida_y_no_importa_app():
    for c in celdas(NOTEBOOK):
        for nodo in ast.walk(ast.parse(c)):
            if isinstance(nodo, ast.Import):
                assert not any(a.name.split(".")[0] == "app" for a in nodo.names)
            if isinstance(nodo, ast.ImportFrom):
                assert (nodo.module or "").split(".")[0] != "app"


@pytest.mark.skipif(not (RUTA_TRAIN.exists() and RUTA_TEST.exists()), reason="sin data del concurso (no va al repo)")
def test_conteo_con_la_data_real():
    t = np.load(RUTA_TRAIN)["times"].astype(str)
    idx, completa = v.indices_ventanas(t)
    val = t >= "2024061600"
    assert (len(t), int(completa.sum())) == (2935, 2908)
    assert (int((~val).sum()), int((~val & completa).sum())) == (2599, 2574)
    assert (int(val.sum()), int((val & completa).sum())) == (336, 334)
    assert_completas_son_consecutivas(t, idx, completa)

    te = np.load(RUTA_TEST)["times"].astype(str)
    _, completa_te = v.indices_ventanas(te)
    assert (len(te), int(completa_te.sum())) == (792, 788)

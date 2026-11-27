"""Validación temporal y métrica oficial (T02 · HU-09). Sin red ni PyTorch."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "train"))

import validacion as v  # noqa: E402

RUTA_TRAIN = RAIZ / "data" / "train.npz"
NOTEBOOK = RAIZ / "train" / "t02_linea_base.ipynb"
NOTEBOOK_T08 = RAIZ / "train" / "t08_experimentos.ipynb"
NOTEBOOK_T09 = RAIZ / "train" / "t09_arquitectura.ipynb"
NOTEBOOK_T09B = RAIZ / "train" / "t09b_capacidad.ipynb"
NOTEBOOK_T09D = RAIZ / "train" / "t09d_lead.ipynb"
NOTEBOOK_T09E = RAIZ / "train" / "t09e_analisis.ipynb"
NOTEBOOK_T09F = RAIZ / "train" / "t09f_gfs00.ipynb"
NOTEBOOK_T09G = RAIZ / "train" / "t09g_gfs00_dif.ipynb"
NOTEBOOK_T09H = RAIZ / "train" / "t09h_diversidad.ipynb"
STARTER = RAIZ / "train" / "starter_notebook.ipynb"


def marcas(dia_ini, dia_fin):
    """Marcas horarias de junio de 2024 entre dos días, ambos incluidos."""
    return np.array([f"202406{d:02d}{h:02d}" for d in range(dia_ini, dia_fin + 1) for h in range(24)])


def datos(semilla=0):
    t = marcas(10, 20)
    rng = np.random.default_rng(semilla)
    return rng.normal(size=(len(t), 3, 2, 4)), t


def test_el_corte_es_temporal_y_no_mezcla_fechas():
    t = marcas(1, 30)
    desorden = np.random.default_rng(0).permutation(t)
    trn, val = v.corte_temporal(desorden)
    assert max(desorden[trn]) == "2024061523"
    assert min(desorden[val]) == "2024061600"
    assert not (trn & val).any() and (trn | val).all()
    assert val.sum() == 15 * 24


def test_corte_vacio_falla():
    with pytest.raises(ValueError, match="vacío"):
        v.corte_temporal(marcas(1, 5))


def test_rmse_y_scores_con_casos_calculados_a_mano():
    real = np.zeros((2, 1, 1, 4))
    pred = np.array([[[[3.0, 1.0, 0.0, 2.0]]], [[[4.0, 1.0, 0.0, 2.0]]]])
    rmse = v.rmse_por_variable(pred, real)
    np.testing.assert_allclose(rmse, [np.sqrt(12.5), 1.0, 0.0, 2.0])
    assert v.score(v.ESCALAS_OFICIALES, v.ESCALAS_OFICIALES) == pytest.approx(0.0)
    assert v.score(v.ESCALAS_OFICIALES / 2, v.ESCALAS_OFICIALES) == pytest.approx(0.5)
    assert v.score(np.zeros(4), v.ESCALAS_OFICIALES) == 1.0


def test_constantes_oficiales_y_horas_evaluadas():
    np.testing.assert_array_equal(v.ESCALAS_OFICIALES, [25.360, 2.0652, 2.6776, 2.0876])
    assert v.HORAS_EVALUADAS == (0, 3, 6, 9, 12, 15, 18, 21)


def test_climatologia_da_score_local_cero():
    Y, t = datos()
    trn, val = v.corte_temporal(t)
    pred = v.climatologia(Y[trn], t[trn])[v.horas(t[val])]
    assert v.validar(pred, Y, t)["score_local"] == pytest.approx(0.0, abs=1e-12)


def test_la_climatologia_no_usa_la_validacion():
    Y, t = datos()
    trn, val = v.corte_temporal(t)
    pred = v.climatologia(Y[trn], t[trn])[v.horas(t[val])]
    cambiado = Y.copy()
    cambiado[val] += 100.0
    clim_cambiada = v.climatologia(cambiado[trn], t[trn])[v.horas(t[val])]
    np.testing.assert_array_equal(pred, clim_cambiada)
    assert v.escala_climatologia(cambiado, t).min() > 90


def test_score_oficial_solo_usa_las_horas_evaluadas():
    Y, t = datos()
    _, val = v.corte_temporal(t)
    pred = Y[val].copy()
    pred[~np.isin(v.horas(t[val]), v.HORAS_EVALUADAS)] += 5.0
    r = v.validar(pred, Y, t)
    assert r["score_oficial"] == 1.0
    assert r["score_local"] < 1.0
    assert (r["horas_val"], r["horas_evaluadas"]) == (120, 40)
    assert (r["primera"], r["ultima"]) == ("2024061600", "2024062023")


def test_entradas_invalidas_fallan():
    Y, t = datos()
    _, val = v.corte_temporal(t)
    with pytest.raises(ValueError, match="Formas distintas"):
        v.validar(Y[val][:1], Y, t)
    con_nan = Y[val].copy()
    con_nan[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        v.validar(con_nan, Y, t)
    with pytest.raises(ValueError, match="horas UTC"):
        v.climatologia(Y[:5], t[:5])


def celdas(ruta):
    return ["".join(c["source"]) for c in json.loads(ruta.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("notebook", [NOTEBOOK, NOTEBOOK_T08, NOTEBOOK_T09, NOTEBOOK_T09B, NOTEBOOK_T09D, NOTEBOOK_T09E, NOTEBOOK_T09F, NOTEBOOK_T09G, NOTEBOOK_T09H], ids=lambda r: r.stem)
def test_el_notebook_lleva_la_copia_exacta_del_modulo(notebook):
    copias = [c for c in celdas(notebook) if c.startswith("# COPIA EXACTA de train/validacion.py")]
    assert len(copias) == 1
    assert copias[0].split("\n", 1)[1] == (RAIZ / "train" / "validacion.py").read_text(encoding="utf-8").rstrip("\n")


def test_la_linea_base_se_mide_con_validar():
    assert any("validar(predict(model, Wtr[val], ttr[val]), Ytr, ttr)" in c for c in celdas(NOTEBOOK))


def test_el_notebook_no_cambia_modelo_ni_hiperparametros_del_starter():
    def entrenamiento(ruta):
        return [c for c in celdas(ruta) if "class Net" in c]

    assert entrenamiento(NOTEBOOK) == entrenamiento(STARTER)
    assert len(entrenamiento(STARTER)) == 1


@pytest.mark.parametrize("notebook", [NOTEBOOK, NOTEBOOK_T08, NOTEBOOK_T09, NOTEBOOK_T09B, NOTEBOOK_T09D, NOTEBOOK_T09E, NOTEBOOK_T09F, NOTEBOOK_T09G, NOTEBOOK_T09H], ids=lambda r: r.stem)
def test_el_notebook_es_sintaxis_valida_y_no_importa_app(notebook):
    for c in celdas(notebook):
        for nodo in ast.walk(ast.parse(c)):
            if isinstance(nodo, ast.Import):
                assert not any(a.name.split(".")[0] == "app" for a in nodo.names)
            if isinstance(nodo, ast.ImportFrom):
                assert (nodo.module or "").split(".")[0] != "app"


@pytest.mark.skipif(not RUTA_TRAIN.exists(), reason="sin data/train.npz (no va al repo)")
def test_con_la_data_real_reproduce_la_climatologia_del_log_de_kaggle():
    """Referencia: corrida del starter en Kaggle (CPU), RMSE de climatología 17.177, 2.01, 2.261, 1.946."""
    tr = np.load(RUTA_TRAIN)
    t = tr["times"].astype(str)
    trn, val = v.corte_temporal(t)
    assert min(t[val]) >= "2024061600" > max(t[trn])
    assert max(t[val]) == "2024063023"
    escala = v.escala_climatologia(tr["Y"], t)
    np.testing.assert_allclose(escala, [17.177, 2.01, 2.261, 1.946], atol=0.001)

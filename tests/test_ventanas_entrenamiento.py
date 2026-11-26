"""
Ventanas del notebook de entrenamiento sin huecos de horas (T11 · HU-09).

Extrae `windows()` del notebook (el notebook es autocontenido y no importa nada del repo) y comprueba
que marca como incompleta toda ventana que cruza un hueco, y que el entrenamiento solo usa las completas.
"""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "train" / "starter_notebook.ipynb"
TRAIN = RAIZ / "data" / "train.npz"


def celdas():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def windows_del_notebook():
    codigo = next(c for c in celdas() if "def windows" in c)
    funcion = re.search(r"^def windows\(.*?(?=^\S)", codigo, re.S | re.M).group(0)
    ns = {"np": np, "pd": pd}
    exec(funcion, ns)
    return ns["windows"]


def horas(*desfases_h, inicio="2024061600"):
    t0 = datetime.strptime(inicio, "%Y%m%d%H")
    return np.array([(t0 + timedelta(hours=h)).strftime("%Y%m%d%H") for h in desfases_h])


def grilla(n):
    """X sintético donde cada hora tiene un valor distinto, para rastrear qué hora entró en cada ventana."""
    X = np.zeros((n, 17, 13, 4), np.float32)
    X[...] = np.arange(n, dtype=np.float32)[:, None, None, None]
    return X


def test_ventanas_que_cruzan_un_hueco_quedan_marcadas():
    windows = windows_del_notebook()
    t = horas(0, 1, 2, 3, 6, 7, 8, 9, 30)          # huecos: 3→6 (falta 4, 5) y 9→30
    W, ok = windows(grilla(len(t)), t)
    assert W.shape == (9, 3, 17, 13, 4)
    assert ok.tolist() == [False, False, True, True, False, False, True, True, False]


def test_ventanas_completas_usan_exactamente_t2_t1_t():
    windows = windows_del_notebook()
    t = horas(0, 1, 2, 3, 6, 7, 8, 9, 30)
    W, ok = windows(grilla(len(t)), t)
    ts = pd.to_datetime(t, format="%Y%m%d%H")
    for i in np.where(ok)[0]:
        usadas = W[i, :, 0, 0, 0].astype(int)       # índice de la hora que entró en cada paso
        assert [ts[j] for j in usadas] == [ts[i] - pd.Timedelta(hours=h) for h in (2, 1, 0)]


def test_hueco_de_una_sola_hora():
    windows = windows_del_notebook()
    t = horas(0, 1, 2, 4, 5, 6)                    # falta la hora 3
    _, ok = windows(grilla(len(t)), t)
    assert ok.tolist() == [False, False, True, False, False, True]


def test_entrenamiento_y_validacion_solo_usan_ventanas_completas():
    entrenamiento = next(c for c in celdas() if "for ep in range(EPOCHS)" in c)
    assert "itr = np.where(trn & ok_tr)[0]" in entrenamiento
    assert "vok = val & ok_tr" in entrenamiento and "Wtr[vok]" in entrenamiento
    assert "(trn & ok_tr).sum()" in entrenamiento
    assert "np.where(trn)[0]" not in entrenamiento and "Wtr[val]" not in entrenamiento


@pytest.mark.skipif(not TRAIN.exists(), reason="falta data/train.npz (se baja de Kaggle)")
def test_conteo_con_la_data_real():
    windows = windows_del_notebook()
    d = np.load(TRAIN)
    t = d["times"].astype(str)
    _, ok = windows(d["X"], t)
    val = t >= "2024061600"
    assert len(ok) == 2935 and ok.sum() == 2908
    assert (~ok & ~val).sum() == 25 and (~ok & val).sum() == 2
    # Ninguna ventana completa cruza un hueco: sus 3 horas son consecutivas en la data real
    ts = pd.to_datetime(t, format="%Y%m%d%H")
    existentes = set(ts)
    for i in np.where(ok)[0]:
        assert all(ts[i] - pd.Timedelta(hours=h) in existentes for h in (1, 2))

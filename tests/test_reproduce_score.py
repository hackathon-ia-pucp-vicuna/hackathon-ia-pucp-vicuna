"""
Los pesos de models/ reproducen el score de validación en un proceso nuevo (T03 · HU-09).

Genera las estadísticas con train/generar_estadisticas.py en una carpeta temporal, carga
models/modelo.pt con app/predict.predecir() (CPU) a través de un manifiesto que apunta a la línea base
(T49: con ese manifiesto los scores de referencia no cambian), arma las ventanas con la función `windows` del
notebook de entrenamiento y evalúa con train/validacion.py.
Los valores de referencia son los de predecir() con la HR recortada a 0–100 (T10), medidos en Kaggle;
los del entrenamiento, sin recorte, están en models/FICHA.md.

La reproducción se omite si faltan PyTorch, pandas, los pesos o la data oficial.

Se ejecuta con:  pytest -q -s tests/test_reproduce_score.py
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(RAIZ / "train"))

import generar_estadisticas as generador  # noqa: E402
import predict  # noqa: E402
import validacion  # noqa: E402

NOTEBOOK = RAIZ / "train" / "t02_linea_base.ipynb"
FICHA = RAIZ / "models" / "FICHA.md"
RUTA_TRAIN = RAIZ / "data" / "train.npz"
TOLERANCIA = 1e-4
REFERENCIA = {"score_local": 0.3377929925918579, "score_oficial": 0.44098055507774636}


def sha_en_ficha(nombre):
    fila = next(f for f in FICHA.read_text(encoding="utf-8").splitlines() if f.startswith(f"| `{nombre}`"))
    return re.search(r"[0-9a-f]{64}", fila).group(0)


def test_los_pesos_son_los_de_la_ficha():
    assert hashlib.sha256(predict.RUTA_PESOS.read_bytes()).hexdigest() == sha_en_ficha("modelo.pt")


def data_sintetica(dias=4):
    """Horas completas antes y después del corte, con la geometría de la competencia."""
    rng = np.random.default_rng(0)
    t = [f"202406{d:02d}{h:02d}" for d in range(16 - dias, 16 + dias) for h in range(24)]
    X = rng.normal(10, 2, (len(t), 17, 13, 4)).astype(np.float32)
    Y = rng.normal(10, 2, (len(t), 70, 35, 4)).astype(np.float32)
    return X, Y, np.array(t)


def test_generar_usa_solo_entrenamiento_y_compara_con_lo_publicado(tmp_path):
    X, Y, t = data_sintetica()
    ref = {k: v.tolist() for k, v in generador.calcular(X, Y, t).items() if k != "clim"}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    np.savez(tmp_path / "train.npz", X=X, Y=Y, times=t)
    salida = generador.generar(tmp_path / "train.npz", tmp_path / "estadisticas.npz", tmp_path / "ref.json")
    est = np.load(salida)
    assert est["clim"].shape == (24, 70, 35, 4)
    trn = t < validacion.INICIO_VAL
    np.testing.assert_allclose(est["clim"][0], Y[trn & (validacion.horas(t) == 0)].mean(0), rtol=1e-6)

    ref["mu_x"][0] += 1
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="mu_x"):
        generador.generar(tmp_path / "train.npz", tmp_path / "otra.npz", tmp_path / "ref.json")
    assert not (tmp_path / "otra.npz").exists()


def test_normalizacion_publicada_tiene_solo_los_doce_numeros():
    ref = json.loads(generador.RUTA_NORMALIZACION.read_text(encoding="utf-8"))
    assert set(ref) == {"mu_x", "sd_x", "anom_sd"}
    assert all(len(v) == 4 and np.isfinite(v).all() for v in ref.values())


def ventanas_del_notebook():
    """La función `windows` tal cual está en el notebook, para no mantener otra copia."""
    pd = pytest.importorskip("pandas")
    for celda in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        codigo = "".join(celda["source"])
        if celda["cell_type"] == "code" and "def windows(" in codigo:
            funcion = next(n for n in ast.parse(codigo).body
                           if isinstance(n, ast.FunctionDef) and n.name == "windows")
            ns = {"np": np, "pd": pd}
            exec(ast.get_source_segment(codigo, funcion), ns)
            return ns["windows"]
    raise AssertionError("El notebook no define windows().")


def test_los_archivos_reproducen_el_score(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    faltan = [r.name for r in (predict.RUTA_PESOS, predict.RUTA_STATIC, RUTA_TRAIN) if not r.exists()]
    if faltan:
        pytest.skip(f"faltan {', '.join(faltan)}")
    # T49: predecir() carga lo que dice el manifiesto; aquí uno que apunta a la línea base (variante starter)
    salida = generador.generar(ruta_salida=tmp_path / "estadisticas.npz")
    manifiesto = {"variante": "starter", "estadisticas": str(salida),
                  "normalizacion": json.loads(generador.RUTA_NORMALIZACION.read_text(encoding="utf-8")),
                  "miembros": [{"archivo": str(predict.RUTA_PESOS),
                                "sha256": hashlib.sha256(predict.RUTA_PESOS.read_bytes()).hexdigest()}]}
    (tmp_path / "modelo.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    monkeypatch.setattr(predict, "RUTA_MANIFIESTO", tmp_path / "modelo.json")
    predict._cargar.cache_clear()

    datos = np.load(RUTA_TRAIN)
    X, Y, t = datos["X"], datos["Y"], datos["times"].astype(str)
    _, val = validacion.corte_temporal(t)
    ventanas = ventanas_del_notebook()(X, t)[val]
    res = validacion.validar(predict.predecir(ventanas, validacion.horas(t[val])), Y, t)

    print()
    for clave, ref in REFERENCIA.items():
        print(f"T03 · {clave}: {res[clave]:.8f} · referencia {ref:.8f} · diferencia {res[clave] - ref:+.2e}")
    print("T03 · RMSE:", res["rmse"])
    print("T03 · RMSE horas evaluadas:", res["rmse_horas_evaluadas"])
    predict._cargar.cache_clear()
    for clave, ref in REFERENCIA.items():
        assert abs(res[clave] - ref) < TOLERANCIA, clave

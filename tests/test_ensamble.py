"""
Pesos del ensamble del producto en models/ (T50 · HU-09): mezcla de 30, envío del 27 nov (público 0.50332).

- El manifiesto models/modelo.json lista 30 miembros y sus SHA-256 coinciden con los archivos.
- `generar_estadisticas.py --ensamble` usa todas las horas y se compara con la normalización del manifiesto.
- Con la data oficial: la inferencia del propio notebook de entrenamiento (train/t09h_diversidad.ipynb, que
  también construye r1, r2 y lead), con los 30 pesos en CPU, reproduce una muestra fija de 2,400 filas del CSV enviado (24 horas de test).

La app (app/predict.predecir()) se prueba aparte: paridad en tests/test_paridad.py y la misma muestra en
tests/test_predecir_ensamble.py (T49).

Se ejecuta con:  pytest -q -s tests/test_ensamble.py
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "train"))

import generar_estadisticas as generador  # noqa: E402
import validacion  # noqa: E402

MANIFIESTO = RAIZ / "models" / "modelo.json"
NOTEBOOK = RAIZ / "train" / "t09h_diversidad.ipynb"
MUESTRA = RAIZ / "tests" / "fixtures" / "envio_mezcla30_muestra.json"
DATA = RAIZ / "data"
TOLERANCIA_ENVIO = 0.002            # criterio de T50, por variable


def manifiesto():
    return json.loads(MANIFIESTO.read_text(encoding="utf-8"))


def test_el_manifiesto_lista_treinta_miembros_con_su_sha():
    m = manifiesto()
    assert m["envio"]["publico"] == 0.50332
    assert m["variante"] == "residuo_crudo_profundo"
    assert len(m["miembros"]) == 30
    assert len({x["archivo"] for x in m["miembros"]}) == 30
    for x in m["miembros"]:
        ruta = RAIZ / "models" / x["archivo"]
        assert hashlib.sha256(ruta.read_bytes()).hexdigest() == x["sha256"], x["archivo"]
        assert x["configuracion"] in m["configuraciones"]
    # r1 y r2 solo difieren en el dropout, que no actúa al predecir: misma red
    c1, c2, c3 = (dict(m["configuraciones"][k]) for k in ("r1", "r2", "lead"))
    assert {k for k in c1 if c1[k] != c2[k]} == {"dropout"}
    # lead = r2 + la hora desde el inicio del WRF (un canal más de entrada)
    assert {k for k in c3 if c2.get(k) != c3[k]} == {"lead"} and c3["lead"] == "lineal"
    # miembros de diversidad: cada uno cambia una sola cosa respecto de lead
    for nombre, cambio in (("ancho96", {"w": 96}), ("ep40", {"epocas": 40}), ("unet", {"arq": "unet", "w": 48})):
        assert m["configuraciones"][nombre] == {**c3, **cambio}, nombre


def test_la_normalizacion_del_manifiesto_tiene_doce_numeros():
    norm = manifiesto()["normalizacion"]
    assert norm["horas"] == "todas"
    assert all(len(norm[k]) == 4 and np.isfinite(norm[k]).all() for k in ("mu_x", "sd_x", "anom_sd"))


def data_sintetica():
    rng = np.random.default_rng(0)
    t = np.array([f"202406{d:02d}{h:02d}" for d in range(12, 20) for h in range(24)])
    return (rng.normal(10, 2, (len(t), 17, 13, 4)).astype(np.float32),
            rng.normal(10, 2, (len(t), 70, 35, 4)).astype(np.float32), t)


def test_generar_ensamble_usa_todas_las_horas(tmp_path):
    X, Y, t = data_sintetica()
    todas = generador.calcular(X, Y, t, todas=True)
    np.testing.assert_allclose(todas["mu_x"], X.mean((0, 1, 2)), rtol=1e-6)
    assert not np.allclose(todas["mu_x"], generador.calcular(X, Y, t)["mu_x"], rtol=1e-6)

    norm = {"horas": "todas", **{k: np.asarray(v).tolist() for k, v in todas.items() if k != "clim"}}
    (tmp_path / "modelo.json").write_text(json.dumps({"normalizacion": norm}))
    np.savez(tmp_path / "train.npz", X=X, Y=Y, times=t)
    salida = generador.generar_ensamble(tmp_path / "train.npz", tmp_path / "e.npz", tmp_path / "modelo.json")
    np.testing.assert_allclose(np.load(salida)["clim"][0], Y[validacion.horas(t) == 0].mean(0), rtol=1e-6)

    norm["sd_x"][1] *= 1.001
    (tmp_path / "modelo.json").write_text(json.dumps({"normalizacion": norm}))
    with pytest.raises(ValueError, match="sd_x"):
        generador.generar_ensamble(tmp_path / "train.npz", tmp_path / "otra.npz", tmp_path / "modelo.json")
    assert not (tmp_path / "otra.npz").exists()


def faltan_archivos():
    return [n for n in ("train.npz", "test.npz", "static.npz") if not (DATA / n).exists()]


@pytest.mark.skipif(bool(faltan_archivos()), reason="sin la data oficial (no va al repo)")
def test_con_la_data_real_la_normalizacion_coincide(tmp_path):
    est = np.load(generador.generar_ensamble(ruta_salida=tmp_path / "e.npz"))
    assert est["clim"].shape == (24, 70, 35, 4)


def inferencia_del_notebook():
    """Celdas de la red y de la inferencia de train/t09h_diversidad.ipynb, ejecutadas tal cual en CPU."""
    torch = pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    import torch.nn as nn
    import torch.nn.functional as F

    celdas = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
              if c["cell_type"] == "code"]
    st = np.load(DATA / "static.npz")
    ns = {"np": np, "pd": pd, "torch": torch, "nn": nn, "F": F, "st": st, "dem": st["dem"],
          "climatologia": validacion.climatologia, "horas": validacion.horas,
          "to_t": lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)}
    codigo = next(c for c in celdas if "def windows(" in c)       # la celda también arma WTR: solo la función
    funcion = next(n for n in ast.parse(codigo).body if isinstance(n, ast.FunctionDef) and n.name == "windows")
    exec(ast.get_source_segment(codigo, funcion), ns)
    for marca in ("class Net(", "def predecir_lote("):
        exec(next(c for c in celdas if marca in c), ns)
    return torch, ns


@pytest.mark.skipif(bool(faltan_archivos()), reason="sin la data oficial (no va al repo)")
def test_los_treinta_pesos_reproducen_la_muestra_del_envio(tmp_path):
    torch, ns = inferencia_del_notebook()
    m = manifiesto()
    est = np.load(generador.generar_ensamble(ruta_salida=tmp_path / "e.npz"))
    stats = {k: est[k] for k in est.files}
    D = {"cfg": m["configuraciones"]["r1"], "stats": stats,
         "CLIM": ns["to_t"](stats["clim"]), "ASD": ns["to_t"](stats["anom_sd"])}

    filas = json.loads(MUESTRA.read_text(encoding="utf-8"))["filas"]
    ids = [f[0] for f in filas]
    esperado = np.array([f[1:] for f in filas], np.float64)
    te = np.load(DATA / "test.npz")
    tte = te["times"].astype(str)
    W = ns["windows"](te["X"], tte)
    horas_muestra = sorted({i[:10] for i in ids})
    pos = {t: k for k, t in enumerate(tte)}
    sel = np.array([pos[h] for h in horas_muestra])

    claves = ("skip", "terreno", "profundo", "w", "bloques", "enc_prof", "dropout", "lags")
    suma = 0
    for x in m["miembros"]:
        cfg = m["configuraciones"][x["configuracion"]]
        if cfg.get("arq") == "unet":
            red = ns["UNet"](w=cfg["w"], dropout=cfg["dropout"], lags=cfg["lags"])
        else:
            red = ns["Net"](**{k: cfg[k] for k in claves}, lead=cfg.get("lead", "ciclica"))
        red.load_state_dict(torch.load(RAIZ / "models" / x["archivo"], map_location="cpu", weights_only=True))
        suma = suma + ns["predecir_lote"](red, D, W[sel], tte[sel])
    P = suma / len(m["miembros"])

    k = {h: j for j, h in enumerate(horas_muestra)}
    obtenido = np.array([P[k[i[:10]], int(i.split("_")[1]), int(i.split("_")[2])] for i in ids])
    dif = np.abs(obtenido - esperado).max(0)
    print(f"\nT50 · {len(ids)} filas de {len(horas_muestra)} horas · dif. máx RH, T, U, V: {np.round(dif, 5).tolist()}")
    assert (dif <= TOLERANCIA_ENVIO).all()

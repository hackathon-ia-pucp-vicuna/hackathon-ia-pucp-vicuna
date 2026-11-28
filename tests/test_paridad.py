"""
Paridad entre la red de cada notebook de entrenamiento y su copia en app/predict.py (T12, T49 · HU-01).

Variantes:
- starter (línea base): `Net` de train/starter_notebook.ipynb contra predict.construir_red().
- residuo_crudo_profundo (T09/T20): `Net` (la de t09b más la opción lead) y `UNet` de train/t09h_diversidad.ipynb contra
  predict.construir_red_residuo() (lead cíclica y lineal) y predict.construir_red_unet().

Para cada variante:
- Sintética (siempre): misma grilla, DEM y pesos aleatorios en las dos redes; 5 muestras, diferencia < 1e-5.
- Real (si existen los pesos y data/static.npz): lo mismo con los pesos y el relieve reales
  (models/modelo.pt para la línea base; los miembros de models/modelo.json para el ensamble).

Se ejecuta con:  pytest -q tests/test_paridad.py -s   (-s muestra el tiempo de 24 h en CPU)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))

import predict  # noqa: E402

NOTEBOOK = RAIZ / "train" / "starter_notebook.ipynb"
NOTEBOOK_RESIDUO = RAIZ / "train" / "t09h_diversidad.ipynb"   # red de t09b con la opción lead, y la U-Net
TOLERANCIA = 1e-5
CLAVES_NET = ("skip", "terreno", "profundo", "w", "bloques", "enc_prof", "dropout", "lags")
CFG_RESIDUO = {"w": 64, "bloques": [1, 2, 4, 1], "enc_prof": False, "dropout": 0.1, "wd": 0.0001, "lags": 3,
               "skip": False, "residuo": "crudo", "terreno": False, "profundo": True, "peso_mj": 1, "epocas": 25}


def static_sintetico(semilla=0):
    """Grillas con la misma geometría que la competencia: GFS 17×13 (fila 0 = norte), WRF 70×35 (fila 0 = sur)."""
    rng = np.random.default_rng(semilla)
    return {
        "gfs_lats": np.round(np.arange(-7.25, -11.25 - 1e-9, -0.25), 2),
        "gfs_lons": np.round(np.arange(-79.0, -76.0 + 1e-9, 0.25), 2),
        "wrf_lats": np.linspace(-10.4, -8.3, 70),
        "wrf_lons": np.linspace(-78.3, -77.2, 35),
        "dem": rng.uniform(1500, 6700, (70, 35)).astype(np.float32),
    }


def red_del_notebook(st):
    """Ejecuta la definición de `Net` tal cual está en el notebook, con GRID y DEM como en su celda."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "class Net" in "".join(c["source"]))
    clase = re.search(r"^class Net\(nn\.Module\):.*?(?=^\S)", codigo, re.S | re.M).group(0)

    glat, glon, wlat, wlon, dem = (st[k] for k in ("gfs_lats", "gfs_lons", "wrf_lats", "wrf_lons", "dem"))
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    ns = {
        "np": np, "torch": torch, "nn": nn, "F": F,
        "GRID": torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32),
        "DEM": torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32)[None, None],
    }
    exec(clase, ns)
    return ns["Net"]()


def celda_red_residuo(st):
    """Ejecuta tal cual la celda de la red de t09h_diversidad (grilla, DEM, interpolar, Bloque y Net) con `st`."""
    celdas = ["".join(c["source"]) for c in json.loads(NOTEBOOK_RESIDUO.read_text(encoding="utf-8"))["cells"]
              if c["cell_type"] == "code"]
    ns = {"np": np, "torch": torch, "nn": nn, "F": F, "st": st, "dem": st["dem"],
          "to_t": lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)}
    exec(next(c for c in celdas if "class Net(" in c), ns)
    return ns


def red_residuo_del_notebook(st, cfg):
    ns = celda_red_residuo(st)
    if cfg.get("arq") == "unet":
        return ns["UNet"](w=cfg["w"], dropout=cfg["dropout"], lags=cfg["lags"])
    return ns["Net"](**{k: cfg[k] for k in CLAVES_NET}, lead=cfg.get("lead", "ciclica"))


def muestras(n=5, semilla=1):
    rng = np.random.default_rng(semilla)
    x = torch.tensor(rng.standard_normal((n, 3, 17, 13, 4)), dtype=torch.float32)
    h = torch.tensor(rng.integers(0, 24, n), dtype=torch.float32)
    return x, h


def comparar_redes(red_nb, red_app, estado=None):
    if estado is None:
        torch.manual_seed(0)
        with torch.no_grad():                         # pesos aleatorios, incluido el parámetro `static`
            for p in red_nb.parameters():
                p.normal_(0, 0.1)
        estado = red_nb.state_dict()
    else:
        red_nb.load_state_dict(estado)
    red_app.load_state_dict(estado)                  # mismo state_dict: mismas claves en ambas redes
    red_nb.eval(), red_app.eval()
    x, h = muestras()
    with torch.no_grad():
        a, b = red_nb(x, h).numpy(), red_app(x, h).numpy()
    return float(np.abs(a - b).max()), a.shape


def comparar(st, estado=None):
    return comparar_redes(red_del_notebook(st), predict.construir_red(st), estado)


def test_paridad_sintetica():
    dif, forma = comparar(static_sintetico())
    assert forma == (5, 70, 35, 4)
    assert dif < TOLERANCIA, f"diferencia máxima {dif:.2e}"


@pytest.mark.skipif(not (predict.RUTA_PESOS.exists() and predict.RUTA_STATIC.exists()),
                    reason="faltan models/modelo.pt o data/static.npz (T03)")
def test_paridad_con_pesos_reales():
    estado = torch.load(predict.RUTA_PESOS, map_location="cpu", weights_only=True)
    dif, _ = comparar(predict.cargar_static(), estado)
    assert dif < TOLERANCIA, f"diferencia máxima {dif:.2e}"


def test_paridad_sintetica_residuo():
    st = static_sintetico()
    dif, forma = comparar_redes(red_residuo_del_notebook(st, CFG_RESIDUO), predict.construir_red_residuo(st, CFG_RESIDUO))
    assert forma == (5, 70, 35, 4)
    assert dif < TOLERANCIA, f"diferencia máxima {dif:.2e}"


def test_paridad_sintetica_residuo_con_lead_lineal():
    st = static_sintetico()
    cfg = {**CFG_RESIDUO, "lead": "lineal"}
    dif, forma = comparar_redes(red_residuo_del_notebook(st, cfg), predict.construir_red_residuo(st, cfg))
    assert forma == (5, 70, 35, 4)
    assert dif < TOLERANCIA, f"diferencia máxima {dif:.2e}"


def test_paridad_sintetica_unet():
    st = static_sintetico()
    cfg = {**CFG_RESIDUO, "w": 48, "arq": "unet", "lead": "lineal"}
    dif, forma = comparar_redes(red_residuo_del_notebook(st, cfg), predict.construir_red_miembro(st, cfg))
    assert forma == (5, 70, 35, 4)
    assert dif < TOLERANCIA, f"diferencia máxima {dif:.2e}"


def test_arquitectura_desconocida_da_error_claro():
    with pytest.raises(predict.ModeloNoDisponible, match="arq"):
        predict.construir_red_miembro(static_sintetico(), {**CFG_RESIDUO, "arq": "otra"})


def test_lead_desconocido_da_error_claro():
    with pytest.raises(predict.ModeloNoDisponible, match="lead"):
        predict.construir_red_residuo(static_sintetico(), {**CFG_RESIDUO, "lead": "onehot"})


def test_interpolacion_de_la_app_igual_a_la_del_notebook():
    st = static_sintetico()
    ns = celda_red_residuo(st)
    gfs = torch.tensor(np.random.default_rng(3).normal(0, 1, (5, 4, 17, 13)), dtype=torch.float32)
    a = ns["interpolar"](gfs).numpy()
    b = predict.construir_red_residuo(st, CFG_RESIDUO).interpolar(gfs).numpy()
    assert np.abs(a - b).max() < TOLERANCIA


def test_variante_no_soportada_da_error_claro():
    with pytest.raises(predict.ModeloNoDisponible, match="no es la variante"):
        predict.construir_red_residuo(static_sintetico(), {**CFG_RESIDUO, "skip": True})


@pytest.mark.skipif(not (predict.RUTA_MANIFIESTO.exists() and predict.RUTA_STATIC.exists()),
                    reason="faltan models/modelo.json o data/static.npz (T50)")
def test_paridad_residuo_con_todos_los_pesos_reales():
    m = predict.leer_manifiesto()
    assert m["variante"] == "residuo_crudo_profundo"
    st = predict.cargar_static()
    difs = []
    for x in m["miembros"]:
        cfg = m["configuraciones"][x["configuracion"]]
        estado = torch.load(RAIZ / "models" / x["archivo"], map_location="cpu", weights_only=True)
        difs.append(comparar_redes(red_residuo_del_notebook(st, cfg), predict.construir_red_miembro(st, cfg), estado)[0])
    print(f"\nT49 · paridad con {len(difs)} miembros reales · diferencia máxima {max(difs):.2e}")
    assert max(difs) < TOLERANCIA


def escribir_manifiesto(carpeta, variante, archivos, est, cfg=None):
    """Manifiesto en `carpeta`, que hace de models/ (las rutas van relativas a ella), con la normalización
    de `est` y la huella de su `clim`. Quien lo usa apunta predict.RUTA_MODELOS a `carpeta`."""
    import hashlib

    norm = {k: np.asarray(est[k]).tolist() for k in ("mu_x", "sd_x", "anom_sd")}
    norm["clim_media"] = predict.media_clim(est["clim"]).tolist()
    m = {"variante": variante, "estadisticas": "estadisticas.npz", "normalizacion": norm,
         "miembros": [{"archivo": Path(a).relative_to(carpeta).as_posix(),
                       "sha256": hashlib.sha256(Path(a).read_bytes()).hexdigest(),
                       "configuracion": "c"} for a in archivos]}
    if cfg is not None:
        m["configuraciones"] = {"c": cfg}
    ruta = carpeta / "modelo.json"
    ruta.write_text(json.dumps(m), encoding="utf-8")
    return ruta


def estadisticas_plausibles():
    clim = np.zeros((24, 70, 35, 4), np.float32)
    clim[..., :] = [70.0, 280.0, 1.0, -1.0]
    return {"mu_x": np.array([70.0, 285.0, 0.5, -0.5], np.float32), "sd_x": np.array([20.0, 6.0, 3.0, 3.0], np.float32),
            "clim": clim, "anom_sd": np.array([15.0, 2.0, 2.5, 2.0], np.float32)}


def preparar_modelo(tmp_path, monkeypatch, variante="starter", n=1):
    """Archivos del modelo con pesos aleatorios y estadísticas plausibles, para probar predecir() de punta a punta."""
    st = static_sintetico()
    np.savez(tmp_path / "static.npz", **st)
    archivos = []
    for i in range(n):
        torch.manual_seed(i)
        red = predict.construir_red(st) if variante == "starter" else predict.construir_red_residuo(st, CFG_RESIDUO)
        with torch.no_grad():
            red.static.normal_(0, 0.5)                # que los miembros den salidas distintas
        archivos.append(tmp_path / f"m{i}.pt")
        torch.save(red.state_dict(), archivos[-1])
    est = estadisticas_plausibles()
    np.savez(tmp_path / "estadisticas.npz", **est)
    ruta = escribir_manifiesto(tmp_path, variante, archivos, est, None if variante == "starter" else CFG_RESIDUO)
    monkeypatch.setattr(predict, "RUTA_STATIC", tmp_path / "static.npz")
    monkeypatch.setattr(predict, "RUTA_MODELOS", tmp_path)
    monkeypatch.setattr(predict, "RUTA_MANIFIESTO", ruta)
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()
    return st, archivos, est


@pytest.fixture
def modelo_temporal(tmp_path, monkeypatch):
    preparar_modelo(tmp_path, monkeypatch)
    yield
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()


def entrada_tipica(n=24, semilla=2):
    rng = np.random.default_rng(semilla)
    Xw = np.empty((n, 3, 17, 13, 4), np.float32)
    Xw[...] = [70.0, 285.0, 0.5, -0.5]
    return Xw + rng.normal(0, 1, Xw.shape).astype(np.float32)


def test_predecir_devuelve_70x35x4_en_unidades_fisicas(modelo_temporal):
    Xw = entrada_tipica()
    Y = predict.predecir(Xw, np.arange(24))
    assert Y.shape == (24, 70, 35, 4)
    assert np.isfinite(Y).all()
    assert 200 < Y[..., 1].mean() < 330                # T en K, no normalizada
    assert predict.predecir(Xw[:0], []).shape == (0, 70, 35, 4)


def test_predecir_valida_la_entrada(modelo_temporal):
    with pytest.raises(ValueError):
        predict.predecir(np.zeros((2, 17, 13, 4)), [0, 1])
    with pytest.raises(ValueError):
        predict.predecir(np.zeros((2, 3, 17, 13, 4)), [0])


def test_tiempo_24_horas_en_cpu(modelo_temporal):
    seg = predict.medir_24h()
    print(f"\npredecir() en CPU · 24 horas: {seg:.3f} s (pesos aleatorios, misma arquitectura)")
    assert seg < 10


def test_falta_archivo_da_error_claro(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "RUTA_MANIFIESTO", tmp_path / "no_existe.json")
    monkeypatch.setattr(predict, "RUTA_STATIC", tmp_path / "no_existe.npz")
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()
    with pytest.raises(predict.ModeloNoDisponible, match="No se encontró"):
        predict.predecir(np.zeros((1, 3, 17, 13, 4)), [0])
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()

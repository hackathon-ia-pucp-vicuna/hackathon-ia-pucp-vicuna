"""
predecir() con ensamble de miembros declarado en models/modelo.json (T49 · HU-09).

- El manifiesto manda: carga exactamente los miembros listados y falla con un error claro en español si
  falta uno, si su SHA-256 no coincide o si las estadísticas no coinciden con la normalización de referencia.
- La salida es la media de los miembros, cada uno con la HR recortada a 0–100 (1 y 5 miembros sintéticos,
  en las dos variantes).
- Con la data oficial: los 10 pesos del envío 58945688 reproducen, con predecir(), la muestra fija del CSV
  enviado (2,400 filas, 24 horas de test) y se mide el tiempo de 24 horas en CPU.

Se ejecuta con:  pytest -q -s tests/test_predecir_ensamble.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(RAIZ / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import predict  # noqa: E402
from test_paridad import CFG_RESIDUO, celda_red_residuo, entrada_tipica, escribir_manifiesto, preparar_modelo  # noqa: E402

DATA = RAIZ / "data"
MUESTRA = RAIZ / "tests" / "fixtures" / "envio_58945688_muestra.json"
TOLERANCIA = 1e-5
TOLERANCIA_ENVIO = 1e-3          # la GPU no es determinista y el CSV está redondeado a 3 decimales


@pytest.fixture(autouse=True)
def limpiar_cache():
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()
    yield
    predict.cargar_static.cache_clear()
    predict._cargar.cache_clear()


def media_esperada(st, archivos, est, variante, Xw, h):
    """Media de los miembros calculada aparte: cada red sobre la entrada normalizada, más el término base
    (interpolación del notebook) en la variante residuo, a unidades físicas y HR recortada por miembro."""
    x = torch.tensor((Xw - est["mu_x"]) / est["sd_x"], dtype=torch.float32)
    clim, asd = est["clim"][h], est["anom_sd"]
    base = 0
    if variante == "residuo_crudo_profundo":
        gfs = torch.tensor(Xw[:, -1], dtype=torch.float32).permute(0, 3, 1, 2)
        base = (celda_red_residuo(st)["interpolar"](gfs).permute(0, 2, 3, 1).numpy() - clim) / asd
    salidas, crudas = [], []
    for a in archivos:
        red = predict.construir_red(st) if variante == "starter" else predict.construir_red_residuo(st, CFG_RESIDUO)
        red.load_state_dict(torch.load(a, map_location="cpu", weights_only=True))
        red.eval()
        with torch.no_grad():
            Y = clim + (red(x, torch.tensor(h, dtype=torch.float32)).numpy() + base) * asd
        crudas.append(Y[..., 0].max())
        Y[..., 0] = np.clip(Y[..., 0], 0, 100)
        salidas.append(Y)
    return np.mean(salidas, 0), max(crudas)


@pytest.mark.parametrize("variante", ["starter", "residuo_crudo_profundo"])
@pytest.mark.parametrize("n", [1, 5])
def test_la_salida_es_la_media_de_los_miembros_con_hr_recortada(tmp_path, monkeypatch, variante, n):
    st, archivos, est = preparar_modelo(tmp_path, monkeypatch, variante, n)
    # HR cerca de 100 y la salida de RH de cada red subida en +0.4 desviaciones (6 %), distinta por miembro:
    # unas celdas pasan de 100 % antes del recorte, así que recortar por miembro importa
    est["clim"][..., 0] = 97.0
    np.savez(tmp_path / "estadisticas.npz", **est)
    for i, a in enumerate(archivos):
        estado = torch.load(a, map_location="cpu", weights_only=True)
        ultima = [k for k in estado if k.startswith("dec.") and k.endswith(".bias")][-1]
        estado[ultima][0] += 0.4 + 0.05 * i
        torch.save(estado, a)
    escribir_manifiesto(tmp_path, variante, archivos, est, None if variante == "starter" else CFG_RESIDUO)
    Xw, h = entrada_tipica(), np.arange(24)
    Xw[..., 0] = 97.0                                # en la variante residuo, el GFS de la hora t entra como base
    esperado, maximo_crudo = media_esperada(st, archivos, est, variante, Xw, h)
    Y = predict.predecir(Xw, h)
    assert maximo_crudo > 100                        # sin recorte alguna celda pasaría de 100 %
    assert Y[..., 0].max() <= 100 and Y[..., 0].min() >= 0
    dif = np.abs(Y - esperado).max()
    print(f"\nT49 · media de {n} miembro(s), {variante}: diferencia máxima {dif:.2e}")
    assert dif < TOLERANCIA


def test_carga_exactamente_los_miembros_del_manifiesto(tmp_path, monkeypatch):
    preparar_modelo(tmp_path, monkeypatch, "residuo_crudo_profundo", 5)
    _, variante, modelos, _ = predict._cargar()
    assert variante == "residuo_crudo_profundo" and len(modelos) == 5


def test_falta_un_miembro_da_error_claro(tmp_path, monkeypatch):
    _, archivos, _ = preparar_modelo(tmp_path, monkeypatch, "starter", 3)
    archivos[1].unlink()
    with pytest.raises(predict.ModeloNoDisponible, match="Falta el miembro .*m1.pt"):
        predict.predecir(entrada_tipica(1), [0])


def test_sha_distinto_da_error_claro(tmp_path, monkeypatch):
    _, archivos, _ = preparar_modelo(tmp_path, monkeypatch, "starter", 2)
    archivos[0].write_bytes(archivos[0].read_bytes() + b"x")
    with pytest.raises(predict.ModeloNoDisponible, match="no coincide con su SHA-256"):
        predict.predecir(entrada_tipica(1), [0])


@pytest.mark.parametrize("clave", ["mu_x", "sd_x", "anom_sd"])
def test_estadisticas_que_no_coinciden_dan_error_claro(tmp_path, monkeypatch, clave):
    _, _, est = preparar_modelo(tmp_path, monkeypatch, "residuo_crudo_profundo", 1)
    est[clave] = est[clave] * (1 + 1e-4)              # fuera de la tolerancia relativa 1e-5
    np.savez(tmp_path / "estadisticas.npz", **est)
    with pytest.raises(predict.ModeloNoDisponible, match=f"no coinciden con la normalización del manifiesto en {clave}"):
        predict.predecir(entrada_tipica(1), [0])


def test_estadisticas_dentro_de_la_tolerancia_se_aceptan(tmp_path, monkeypatch):
    _, _, est = preparar_modelo(tmp_path, monkeypatch, "starter", 1)
    est["mu_x"] = est["mu_x"] * (1 + 1e-6)
    np.savez(tmp_path / "estadisticas.npz", **est)
    assert predict.predecir(entrada_tipica(1), [0]).shape == (1, 70, 35, 4)


def test_variante_desconocida_da_error_claro(tmp_path, monkeypatch):
    preparar_modelo(tmp_path, monkeypatch, "starter", 1)
    m = json.loads(predict.RUTA_MANIFIESTO.read_text(encoding="utf-8"))
    m["variante"] = "unet"
    predict.RUTA_MANIFIESTO.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(predict.ModeloNoDisponible, match="Variante de red desconocida"):
        predict.predecir(entrada_tipica(1), [0])


@pytest.mark.parametrize("nombre", ["/tmp/m0.pt", "../m0.pt", "ensamble/../../m0.pt"])
def test_rutas_fuera_de_models_dan_error_claro(tmp_path, monkeypatch, nombre):
    preparar_modelo(tmp_path, monkeypatch, "starter", 1)
    m = json.loads(predict.RUTA_MANIFIESTO.read_text(encoding="utf-8"))
    m["miembros"][0]["archivo"] = nombre
    predict.RUTA_MANIFIESTO.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(predict.ModeloNoDisponible, match="fuera de models/"):
        predict.predecir(entrada_tipica(1), [0])


def test_clim_distinta_da_error_claro(tmp_path, monkeypatch):
    _, _, est = preparar_modelo(tmp_path, monkeypatch, "residuo_crudo_profundo", 1)
    est["clim"][..., 1] += 0.5                       # misma mu_x, sd_x y anom_sd, otra climatología de T
    np.savez(tmp_path / "estadisticas.npz", **est)
    with pytest.raises(predict.ModeloNoDisponible, match="en clim_media"):
        predict.predecir(entrada_tipica(1), [0])


def test_estadisticas_incompletas_dan_error_claro(tmp_path, monkeypatch):
    _, _, est = preparar_modelo(tmp_path, monkeypatch, "starter", 1)
    np.savez(tmp_path / "estadisticas.npz", **{k: v for k, v in est.items() if k != "clim"})
    with pytest.raises(predict.ModeloNoDisponible, match="les faltan: clim"):
        predict.predecir(entrada_tipica(1), [0])


def test_configuracion_incompleta_da_error_claro(tmp_path, monkeypatch):
    preparar_modelo(tmp_path, monkeypatch, "residuo_crudo_profundo", 1)
    m = json.loads(predict.RUTA_MANIFIESTO.read_text(encoding="utf-8"))
    del m["configuraciones"]["c"]["bloques"]
    predict.RUTA_MANIFIESTO.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(predict.ModeloNoDisponible, match="le faltan campos: bloques"):
        predict.predecir(entrada_tipica(1), [0])


def faltan_archivos():
    faltan = [n for n in ("train.npz", "test.npz", "static.npz") if not (DATA / n).exists()]
    if not predict.RUTA_MANIFIESTO.exists():
        faltan.append("models/modelo.json")
    return faltan


@pytest.mark.skipif(bool(faltan_archivos()), reason="sin la data oficial o sin models/modelo.json")
def test_predecir_reproduce_la_muestra_del_envio(tmp_path, monkeypatch):
    import generar_estadisticas as generador
    from ventanas import armar_ventanas

    m = predict.leer_manifiesto()
    assert m["envio_kaggle"] == 58945688
    # La carpeta temporal hace de models/: el manifiesto real sin cambios, copia de los 10 pesos y
    # estadísticas generadas con --ensamble (su clim también se comprueba, con clim_media)
    assert len(m["normalizacion"].get("clim_media", [])) == 4
    shutil.copytree(RAIZ / "models" / "ensamble", tmp_path / "ensamble")
    shutil.copy(predict.RUTA_MANIFIESTO, tmp_path / "modelo.json")
    generador.generar_ensamble(ruta_salida=tmp_path / m["estadisticas"])
    monkeypatch.setattr(predict, "RUTA_MODELOS", tmp_path)
    monkeypatch.setattr(predict, "RUTA_MANIFIESTO", tmp_path / "modelo.json")

    filas = json.loads(MUESTRA.read_text(encoding="utf-8"))["filas"]
    ids = [f[0] for f in filas]
    esperado = np.array([f[1:] for f in filas], np.float64)
    te = np.load(DATA / "test.npz")
    tte = te["times"].astype(str)
    W, _ = armar_ventanas(te["X"], tte)              # la hora que falta se rellena con t, como en el notebook
    horas_muestra = sorted({i[:10] for i in ids})
    pos = {t: k for k, t in enumerate(tte)}
    sel = np.array([pos[h] for h in horas_muestra])
    P = predict.predecir(W[sel], np.array([int(t[-2:]) for t in tte[sel]]))

    k = {h: j for j, h in enumerate(horas_muestra)}
    obtenido = np.array([P[k[i[:10]], int(i.split("_")[1]), int(i.split("_")[2])] for i in ids])
    dif = np.abs(obtenido - esperado).max(0)
    seg = predict.medir_24h()
    print(f"\nT49 · predecir() con {len(m['miembros'])} miembros · {len(ids)} filas de {len(horas_muestra)} horas · "
          f"dif. máx RH, T, U, V: {np.round(dif, 5).tolist()} · 24 h en CPU: {seg:.2f} s")
    assert (dif <= TOLERANCIA_ENVIO).all()

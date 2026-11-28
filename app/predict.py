"""
Único punto de entrada al modelo para el producto: `predecir()` (T12, T49).

Usa la MISMA red y los MISMOS pesos que la submission de Kaggle. Qué pesos se cargan lo decide
`models/modelo.json` (el manifiesto): variante de red, miembros con su SHA-256, normalización de
referencia y archivo de estadísticas. La salida es la media de los miembros, cada uno con la HR
recortada a 0–100.

Variantes que se pueden construir (copias de las redes de los notebooks; `tests/test_paridad.py`
comprueba que dan la misma salida que la red del notebook):
  starter                  `Net` del starter (línea base, T03). Predice la anomalía sobre la climatología.
  residuo_crudo_profundo   red de T09/T20 (train/t09b_capacidad.ipynb): decodificador con bloques
                           residuales dilatados; corrige el GFS de la hora t interpolado a 3 km.
                           No usa `clim_gfs`. Un miembro con cfg["arq"] = "unet" es la U-Net de
                           train/t09h_diversidad.ipynb, con la misma entrada, salida y término base.

Archivos que necesita (ninguno se genera desde aquí):
  models/modelo.json        manifiesto (T49). Las rutas van relativas a models/; la normalización de
                            referencia trae mu_x, sd_x, anom_sd y, opcional, clim_media (media de clim por variable)
  models/<miembros>.pt      state_dict de cada miembro (T03, T50)
  models/<estadisticas>.npz mu_x, sd_x, clim (24,70,35,4), anom_sd; no va al repo, se genera con
                            `python train/generar_estadisticas.py` (línea base) o con `--ensamble` (T50)
  data/static.npz           DEM y grillas de la competencia (se baja de Kaggle; no va al repo)

Medir el tiempo en CPU para 24 horas (con los archivos reales):
  python app/predict.py
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELOS = RAIZ / "models"
RUTA_MANIFIESTO = RUTA_MODELOS / "modelo.json"
RUTA_PESOS = RUTA_MODELOS / "modelo.pt"                 # pesos de la línea base (T03)
RUTA_ESTADISTICAS = RUTA_MODELOS / "estadisticas.npz"   # estadísticas de la línea base (T03)
RUTA_STATIC = RAIZ / "data" / "static.npz"

VARIANTES = ("starter", "residuo_crudo_profundo")
ESTADISTICAS_POR_DEFECTO = {"starter": "estadisticas.npz", "residuo_crudo_profundo": "estadisticas_ensamble.npz"}
TOLERANCIA_NORMALIZACION = 1e-5                          # relativa
COMO_GENERAR = {"starter": "python train/generar_estadisticas.py",
                "residuo_crudo_profundo": "python train/generar_estadisticas.py --ensamble"}


class ModeloNoDisponible(RuntimeError):
    """Falta un archivo del modelo o no coincide con el manifiesto; el mensaje está en español."""


def _exigir(ruta, que, como=""):
    if not Path(ruta).exists():
        raise ModeloNoDisponible(f"No se encontró {que} ({Path(ruta).name}). {como}".strip())


def _ruta_en_models(nombre):
    """Ruta de un archivo del manifiesto, siempre dentro de models/: no se aceptan rutas absolutas ni con '..'."""
    rel = Path(nombre)
    if rel.is_absolute() or rel.anchor or ".." in rel.parts:  # anchor: en Windows "/tmp/x" no es absoluta
        raise ModeloNoDisponible(f"El manifiesto del modelo nombra un archivo fuera de models/: {nombre}. "
                                 "Usa una ruta relativa a models/.")
    return RUTA_MODELOS / rel


@lru_cache(maxsize=1)
def cargar_static():
    _exigir(RUTA_STATIC, "el archivo de relieve y grillas de la competencia")
    st = np.load(RUTA_STATIC)
    return {k: st[k] for k in st.files}


def _grilla_y_dem(st):
    """Grilla WRF dentro de la grilla GFS (para grid_sample) y DEM normalizado, igual que en los notebooks."""
    import torch

    glat, glon = st["gfs_lats"], st["gfs_lons"]          # GFS: fila 0 = norte
    wlat, wlon = st["wrf_lats"], st["wrf_lons"]          # WRF: fila 0 = sur
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    grid = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32)
    dem = st["dem"]
    dem_n = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32)[None, None]
    return grid, dem_n


def construir_red(st):
    """
    Red `Net` del starter con la grilla y el DEM de `st` (dict con gfs_lats, gfs_lons, wrf_lats,
    wrf_lons, dem). Pesos sin cargar. Separada de _cargar() para poder probar la paridad.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    grid, dem_n = _grilla_y_dem(st)

    class Net(nn.Module):
        """Copia de `Net` del starter; GRID y DEM pasan de globales a buffers no persistentes."""

        def __init__(self, w=64):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(14, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU())
            self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
            self.dec = nn.Sequential(
                nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
                nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
                nn.Conv2d(w, 4, 1))
            self.register_buffer("grid", grid, persistent=False)
            self.register_buffer("dem", dem_n, persistent=False)

        def forward(self, x, h):                     # x: (B,3,17,13,4), h: (B,) hora UTC
            B = x.shape[0]
            x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
            ang = 2 * np.pi * h / 24
            hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
            f = self.enc(torch.cat([x, hh], 1))
            f = F.grid_sample(f, self.grid.expand(B, -1, -1, -1), align_corners=True)
            f = torch.cat([f, self.dem.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
            return self.dec(f).permute(0, 2, 3, 1)  # (B,70,35,4) anomalía normalizada

    return Net()


def construir_red_residuo(st, cfg):
    """
    Red de T09/T20 (`Net` de train/t09b_capacidad.ipynb con profundo=True, sin skip, terreno ni
    codificador profundo) con la grilla y el DEM de `st`. Con cfg["lead"] = "lineal", la de
    train/t09d_lead.ipynb: un canal más de entrada con la hora desde el inicio del WRF (h/23). Pesos sin cargar. Mismas claves de
    state_dict que la red del notebook. El dropout no actúa al predecir.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    soportada = {"skip": False, "terreno": False, "enc_prof": False, "profundo": True, "lags": 3, "residuo": "crudo"}
    faltan = [k for k in (*soportada, "w", "bloques") if k not in cfg]
    if faltan:
        raise ModeloNoDisponible(f"A la configuración del manifiesto le faltan campos: {', '.join(faltan)}.")
    distintas = {k: cfg.get(k) for k, v in soportada.items() if cfg.get(k) != v}
    if distintas:
        raise ModeloNoDisponible(f"La configuración del manifiesto no es la variante residuo_crudo_profundo: {distintas}.")
    lead = cfg.get("lead", "ciclica")           # T20: hora desde el inicio del WRF (train/t09d_lead.ipynb)
    if lead not in ("ciclica", "lineal"):
        raise ModeloNoDisponible(f"La configuración del manifiesto no es la variante residuo_crudo_profundo: lead={lead!r}.")
    nh = 2 if lead == "ciclica" else 3
    w, bloques, p = int(cfg["w"]), [int(d) for d in cfg["bloques"]], float(cfg.get("dropout", 0.0))
    grid, dem_n = _grilla_y_dem(st)

    class Bloque(nn.Module):
        def __init__(self, w, d, p=0.0):
            super().__init__()
            self.p = p
            self.f = nn.Sequential(nn.Conv2d(w, w, 3, padding=d, dilation=d), nn.GELU(),
                                   nn.Conv2d(w, w, 3, padding=d, dilation=d))

        def forward(self, x):
            return F.gelu(x + F.dropout2d(self.f(x), self.p, self.training))

    class NetResiduo(nn.Module):
        """Copia de `Net` del notebook (profundo=True); GRID y DEM pasan a buffers no persistentes."""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(4 * 3 + nh, w, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(w, w, 3, padding=1), nn.GELU())
            self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
            self.dec = nn.Sequential(nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
                                     *[Bloque(w, d, p) for d in bloques], nn.Conv2d(w, 4, 1))
            self.register_buffer("grid", grid, persistent=False)
            self.register_buffer("dem", dem_n, persistent=False)

        def interpolar(self, a):
            """(B,C,17,13) GFS -> (B,C,70,35) WRF, bilineal."""
            return F.grid_sample(a, self.grid.expand(len(a), -1, -1, -1), align_corners=True)

        def forward(self, x, h):                     # x: (B,3,17,13,4) normalizada, h: (B,) hora UTC
            B = x.shape[0]
            x = x.permute(0, 1, 4, 2, 3).reshape(B, -1, 17, 13)
            ang = 2 * np.pi * h / 24
            hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)
            if lead == "lineal":                     # el WRF se reinicia a las 00 UTC: h/23 separa 23 de 00
                hh = torch.cat([hh, (h / 23 * 2 - 1)[:, None]], 1)
            hh = hh[..., None, None].expand(B, nh, 17, 13)
            f = self.interpolar(self.enc(torch.cat([x, hh], 1)))
            f = torch.cat([f, self.dem.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
            return self.dec(f).permute(0, 2, 3, 1)  # (B,70,35,4) anomalía normalizada (sin el término base)

    return NetResiduo()


def construir_red_unet(st, cfg):
    """
    U-Net a 3 km (`UNet` de train/t09h_diversidad.ipynb): GFS interpolado de las 3 horas, hora, DEM y
    mapa estático. Misma entrada, salida y término base que la red residual. Pesos sin cargar.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    soportada = {"lags": 3, "residuo": "crudo"}
    if "w" not in cfg or any(cfg.get(k) != v for k, v in soportada.items()):
        raise ModeloNoDisponible("La configuración del manifiesto no es la U-Net de la variante residuo_crudo_profundo.")
    w, p = int(cfg["w"]), float(cfg.get("dropout", 0.0))
    grid, dem_n = _grilla_y_dem(st)

    def par(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Dropout2d(p),
                             nn.Conv2d(b, b, 3, padding=1), nn.GELU())

    def sube(a, ref):
        return F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    class UNet(nn.Module):
        """Copia de `UNet` del notebook; GRID y DEM pasan a buffers no persistentes."""

        def __init__(self):
            super().__init__()
            self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
            self.d1, self.d2, self.d3 = par(4 * 3 + 3 + 9, w), par(w, 2 * w), par(2 * w, 4 * w)
            self.u2, self.u1 = par(4 * w + 2 * w, 2 * w), par(2 * w + w, w)
            self.fin = nn.Conv2d(w, 4, 1)
            self.register_buffer("grid", grid, persistent=False)
            self.register_buffer("dem", dem_n, persistent=False)

        def interpolar(self, a):
            """(B,C,17,13) GFS -> (B,C,70,35) WRF, bilineal."""
            return F.grid_sample(a, self.grid.expand(len(a), -1, -1, -1), align_corners=True)

        def forward(self, x, h):                     # x: (B,3,17,13,4) normalizada, h: (B,) hora UTC
            B = x.shape[0]
            g = self.interpolar(x.permute(0, 1, 4, 2, 3).reshape(B, -1, 17, 13))
            ang = 2 * np.pi * h / 24
            hh = torch.stack([torch.sin(ang), torch.cos(ang), h / 23 * 2 - 1], 1)[..., None, None].expand(B, 3, 70, 35)
            e1 = self.d1(torch.cat([g, hh, self.dem.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1))
            e2 = self.d2(F.avg_pool2d(e1, 2, ceil_mode=True))
            e3 = self.d3(F.avg_pool2d(e2, 2, ceil_mode=True))
            d = self.u2(torch.cat([sube(e3, e2), e2], 1))
            d = self.u1(torch.cat([sube(d, e1), e1], 1))
            return self.fin(d).permute(0, 2, 3, 1)  # (B,70,35,4) anomalía normalizada (sin el término base)

    return UNet()


def construir_red_miembro(st, cfg):
    """Red de un miembro de la variante residuo_crudo_profundo según cfg["arq"] (sin campo: la red residual)."""
    arq = cfg.get("arq", "residual")
    if arq == "unet":
        return construir_red_unet(st, cfg)
    if arq == "residual":
        return construir_red_residuo(st, cfg)
    raise ModeloNoDisponible(f"Arquitectura desconocida en el manifiesto del modelo: arq={arq!r}.")


def leer_manifiesto(ruta=None):
    """Lee y valida models/modelo.json. Devuelve el dict con la ruta de estadísticas resuelta."""
    ruta = Path(ruta or RUTA_MANIFIESTO)
    _exigir(ruta, "el manifiesto del modelo", "Debe listar la variante de red y los miembros (T49).")
    try:
        m = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ModeloNoDisponible(f"El manifiesto {ruta.name} no es un JSON válido: {e}.") from e
    variante = m.get("variante")
    if variante not in VARIANTES:
        raise ModeloNoDisponible(f"Variante de red desconocida en {ruta.name}: {variante!r}. Se admiten: {', '.join(VARIANTES)}.")
    if not m.get("miembros"):
        raise ModeloNoDisponible(f"El manifiesto {ruta.name} no lista ningún miembro.")
    norm = m.get("normalizacion", {})
    faltan = [k for k in ("mu_x", "sd_x", "anom_sd") if len(norm.get(k, [])) != 4]
    if "clim_media" in norm and len(norm["clim_media"]) != 4:
        faltan.append("clim_media")
    if faltan:
        raise ModeloNoDisponible(f"El manifiesto {ruta.name} no trae la normalización de referencia: {', '.join(faltan)}.")
    for x in m["miembros"]:
        if not x.get("archivo") or len(x.get("sha256", "")) != 64:
            raise ModeloNoDisponible(f"Un miembro de {ruta.name} no tiene archivo o SHA-256: {x}.")
        if variante == "residuo_crudo_profundo" and x.get("configuracion") not in m.get("configuraciones", {}):
            raise ModeloNoDisponible(f"El miembro {x['archivo']} no tiene su configuración en {ruta.name}.")
    m["_estadisticas"] = _ruta_en_models(m.get("estadisticas") or ESTADISTICAS_POR_DEFECTO[variante])
    return m


def _comprobar_miembro(x):
    ruta = _ruta_en_models(x["archivo"])
    if not ruta.exists():
        raise ModeloNoDisponible(f"Falta el miembro {x['archivo']} que lista el manifiesto del modelo.")
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
    if sha != x["sha256"]:
        raise ModeloNoDisponible(f"El miembro {x['archivo']} no coincide con su SHA-256 del manifiesto "
                                 "(el archivo cambió o está dañado).")
    return ruta


def media_clim(clim):
    """Media de la climatología por variable (4 números): huella de `clim` que guarda el manifiesto."""
    return np.asarray(clim, np.float64).reshape(-1, 4).mean(0)


def _comprobar_normalizacion(est, m):
    """mu_x, sd_x y anom_sd contra el manifiesto; `clim` por su media por variable si el manifiesto la trae."""
    norm = m["normalizacion"]
    valores = {k: est[k] for k in ("mu_x", "sd_x", "anom_sd")}
    if "clim_media" in norm:
        valores["clim_media"] = media_clim(est["clim"])
    for k, v in valores.items():
        ref = np.asarray(norm[k], np.float64)
        if not np.allclose(v, ref, rtol=TOLERANCIA_NORMALIZACION, atol=0):
            raise ModeloNoDisponible(
                f"Las estadísticas ({m['_estadisticas'].name}) no coinciden con la normalización del manifiesto en "
                f"{k} (tolerancia relativa {TOLERANCIA_NORMALIZACION:g}). Genéralas de nuevo con: {COMO_GENERAR[m['variante']]}")


@lru_cache(maxsize=1)
def _cargar():
    import torch

    m = leer_manifiesto()
    rutas = [_comprobar_miembro(x) for x in m["miembros"]]
    _exigir(m["_estadisticas"], "las estadísticas de normalización", f"Genéralas con: {COMO_GENERAR[m['variante']]}")
    est = np.load(m["_estadisticas"])
    faltan = [k for k in ("mu_x", "sd_x", "clim", "anom_sd") if k not in est.files]
    if faltan:
        raise ModeloNoDisponible(f"A las estadísticas ({m['_estadisticas'].name}) les faltan: {', '.join(faltan)}. "
                                 f"Genéralas de nuevo con: {COMO_GENERAR[m['variante']]}")
    stats = {k: est[k].astype(np.float32) for k in ("mu_x", "sd_x", "clim", "anom_sd")}
    _comprobar_normalizacion(stats, m)
    st = cargar_static()
    modelos = []
    for x, ruta in zip(m["miembros"], rutas):
        if m["variante"] == "starter":
            red = construir_red(st)
        else:
            red = construir_red_miembro(st, m["configuraciones"][x["configuracion"]])
        red.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
        red.eval()
        modelos.append(red)
    return torch, m["variante"], modelos, stats


def predecir(Xw, horas_utc, lote=64):
    """
    Xw        : (N, 3, 17, 13, 4) ventanas [t-2, t-1, t] del GFS en unidades físicas
                (RH %, T K, U m/s, V m/s); fila 0 = norte.
    horas_utc : (N,) hora UTC (0–23) de cada ventana.
    Devuelve  : (N, 70, 35, 4) en unidades físicas; fila 0 = sur (como Y en Kaggle).
                Media de los miembros del manifiesto, cada uno con la HR recortada a 0–100 %.
    """
    torch, variante, modelos, s = _cargar()
    Xw = np.asarray(Xw, dtype=np.float32)
    h_all = np.asarray(horas_utc, dtype=np.int64)
    if Xw.ndim != 5 or Xw.shape[1:] != (3, 17, 13, 4):
        raise ValueError(f"Se esperaban ventanas (N, 3, 17, 13, 4) y llegó {Xw.shape}.")
    if len(h_all) != len(Xw):
        raise ValueError("Debe haber una hora UTC por ventana.")
    if len(Xw) == 0:
        return np.empty((0, 70, 35, 4), np.float32)
    clim, asd = torch.from_numpy(s["clim"]), torch.from_numpy(s["anom_sd"])
    salida = []
    with torch.no_grad():
        for i in range(0, len(Xw), lote):
            x = torch.from_numpy((Xw[i:i + lote] - s["mu_x"]) / s["sd_x"])
            h = torch.from_numpy(h_all[i:i + lote])
            base = None
            if variante == "residuo_crudo_profundo":   # GFS de la hora t interpolado, en anomalía normalizada
                gfs = torch.from_numpy(Xw[i:i + lote, -1]).permute(0, 3, 1, 2)
                base = (modelos[0].interpolar(gfs).permute(0, 2, 3, 1) - clim[h]) / asd
            suma = 0
            for red in modelos:
                anom = red(x, h.float())
                if base is not None:
                    anom = anom + base
                Y = clim[h] + anom * asd
                Y[..., 0] = Y[..., 0].clamp(0, 100)    # HR en %, por miembro (como en Kaggle)
                suma = suma + Y
            salida.append((suma / len(modelos)).numpy())
    return np.concatenate(salida)


def medir_24h(repeticiones=3):
    """Segundos (mejor de `repeticiones`) para predecir 24 horas en CPU, con entrada típica."""
    import time

    _, _, _, s = _cargar()
    rng = np.random.default_rng(0)
    Xw = (s["mu_x"] + s["sd_x"] * rng.standard_normal((24, 3, 17, 13, 4))).astype(np.float32)
    horas = np.arange(24)
    predecir(Xw[:1], horas[:1])                      # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        predecir(Xw, horas)
        tiempos.append(time.perf_counter() - t0)
    return min(tiempos)


if __name__ == "__main__":
    _, variante, modelos, _ = _cargar()
    print(f"predecir() en CPU · {variante}, {len(modelos)} miembro(s) · 24 horas: {medir_24h():.3f} s")

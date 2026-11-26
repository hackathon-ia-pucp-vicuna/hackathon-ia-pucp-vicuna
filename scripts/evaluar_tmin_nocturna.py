"""
T36 · T mínima nocturna en la validación temporal: modelo vs GFS crudo interpolado, contra el WRF.

Noche = 19:00–07:00 hora de Lima = 00–12 UTC del mismo día UTC. Solo cuentan las noches con las 13 horas.
Para cada noche y celda se toma la T mínima (horaria y solo en las horas 00, 03, 06, 09, 12 UTC, que son
las que evalúa Kaggle y usa el aviso) y se compara con la del WRF.

GFS interpolado: bilineal de la grilla 17×13 a la de 3 km con las coordenadas reales (lo mismo que hace
`grid_sample` en la red), sin corrección por altura.

Uso (desde la raíz, con data/train.npz, data/static.npz, models/modelo.pt y models/estadisticas.npz):
    python scripts/evaluar_tmin_nocturna.py
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(RAIZ / "train"))

from predict import cargar_static, predecir  # noqa: E402
from validacion import INICIO_VAL  # noqa: E402
from ventanas import armar_ventanas  # noqa: E402

HORAS_NOCHE = tuple(range(0, 13))          # 00–12 UTC = 19:00–07:00 Lima
HORAS_KAGGLE = (0, 3, 6, 9, 12)
K = 273.15
COTA_CULTIVO = 4000                        # m; tope aproximado de la papa en los Andes
LOCALIDADES = {"Carhuaz": (36, 14), "Huaraz": (27, 18), "Yungay": (42, 11)}   # app/frost/zones.yaml


def interpolar_gfs(X, st):
    """Bilineal (N,17,13) GFS (fila 0 = norte) → (N,70,35) WRF (fila 0 = sur)."""
    glat, glon, wlat, wlon = st["gfs_lats"], st["gfs_lons"], st["wrf_lats"], st["wrf_lons"]
    fy = (wlat - glat[0]) / (glat[1] - glat[0])          # índice fraccional de fila GFS
    fx = (wlon - glon[0]) / (glon[1] - glon[0])
    y0 = np.clip(np.floor(fy).astype(int), 0, len(glat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(glon) - 2)
    wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
    Y0, X0 = y0[:, None], x0[None, :]
    return ((1 - wy) * (1 - wx) * X[:, Y0, X0] + (1 - wy) * wx * X[:, Y0, X0 + 1]
            + wy * (1 - wx) * X[:, Y0 + 1, X0] + wy * wx * X[:, Y0 + 1, X0 + 1])


def noches_completas(t_val):
    dias = sorted({t[:8] for t in t_val})
    completas = [d for d in dias if all(f"{d}{h:02d}" in t_val for h in HORAS_NOCHE)]
    return completas, [d for d in dias if d not in completas]


def tmin(T, t_val, noches, horas):
    """(noches, 70, 35): mínimo sobre las horas dadas de cada noche."""
    pos = {t: i for i, t in enumerate(t_val)}
    return np.stack([T[[pos[f"{d}{h:02d}"] for h in horas]].min(0) for d in noches])


def metricas(pred, real):
    e = pred - real
    return np.sqrt((e ** 2).mean()), np.abs(e).mean(), e.mean()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = np.load(RAIZ / "data" / "train.npz")
    X, Y, t = d["X"], d["Y"], d["times"].astype(str)
    st = cargar_static()

    Xw, _ = armar_ventanas(X, t)
    val = t >= INICIO_VAL
    t_val = list(t[val])
    h_val = np.array([int(s[-2:]) for s in t_val])

    T_wrf = Y[val][..., 1]
    T_mod = predecir(Xw[val], h_val)[..., 1]
    T_gfs = interpolar_gfs(X[val][..., 1], st)

    noches, incompletas = noches_completas(set(t_val))
    sin_datos = sorted({f"202406{x:02d}" for x in range(16, 31)} - {x[:8] for x in t_val})
    print(f"Validación {t_val[0]}–{t_val[-1]} · {len(noches)} noches completas · "
          f"incompletas: {incompletas or 'ninguna'} · días sin datos: {sin_datos or 'ninguno'}", end="\n\n")

    dem = st["dem"]
    ambitos = {
        "70×35 celdas": np.ones(dem.shape, bool),
        f"celdas ≤ {COTA_CULTIVO} m": dem <= COTA_CULTIVO,
        "3 localidades": np.isin(np.arange(dem.size), [f * dem.shape[1] + c for f, c in LOCALIDADES.values()]
                                 ).reshape(dem.shape),
    }
    print("Celdas por ámbito: " + " · ".join(f"{k}: {int(v.sum())}" for k, v in ambitos.items()), end="\n\n")

    for etiqueta, horas in (("horaria (00–12 UTC)", HORAS_NOCHE), ("horas Kaggle (00,03,06,09,12 UTC)", HORAS_KAGGLE)):
        w, m, g = (tmin(T, t_val, noches, horas) for T in (T_wrf, T_mod, T_gfs))
        print(f"### T mínima {etiqueta}", end="\n\n")
        print("| Ámbito | Fuente | RMSE (°C) | MAE (°C) | Sesgo (°C) |")
        print("|---|---|---|---|---|")
        for ambito, mask in ambitos.items():
            for nombre, p in (("Modelo", m), ("GFS interpolado", g)):
                print("| {} | {} | {:.2f} | {:.2f} | {:+.2f} |".format(ambito, nombre, *metricas(p[:, mask], w[:, mask])))
        print()

        for ambito in ("70×35 celdas", f"celdas ≤ {COTA_CULTIVO} m"):
            mask = ambitos[ambito]
            wh, mh, gh = ((a[:, mask] - K) <= 0 for a in (w, m, g))
            print(f"{ambito} · WRF: noches con alguna celda ≤ 0 °C: {int(wh.any(1).sum())}/{len(noches)} · "
                  f"celdas-noche ≤ 0 °C: {int(wh.sum())} ({wh.mean():.1%})")
            for nombre, p in (("Modelo", mh), ("GFS interpolado", gh)):
                det = (p & wh).sum() / wh.sum() if wh.any() else float("nan")
                fa = (p & ~wh).sum() / p.sum() if p.any() else float("nan")
                print(f"  {nombre}: avisa {int(p.sum())} · detecta {det:.1%} de las reales · falsas alarmas {fa:.1%}")
        for loc, (f, c) in LOCALIDADES.items():
            print(f"  {loc} ({int(dem[f, c])} m): T mín. más baja de las noches · WRF {(w[:, f, c] - K).min():.1f} °C · "
                  f"modelo {(m[:, f, c] - K).min():.1f} °C · GFS {(g[:, f, c] - K).min():.1f} °C")
        print()

if __name__ == "__main__":
    main()

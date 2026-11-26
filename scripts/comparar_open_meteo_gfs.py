"""
T13 · Compara el GFS de Open-Meteo (archivo histórico) con el GFS del concurso, hora a hora y celda a celda.

Uso (desde la raíz del repo, con data/train.npz y data/test.npz):
    python scripts/comparar_open_meteo_gfs.py 2024-07-02 2024-07-15
    python scripts/comparar_open_meteo_gfs.py 2024-06-16 2024-06-22

Regla de la hackathon: Open-Meteo es solo para el producto. Este script solo VALIDA que la entrada del
producto sea igual a la del entrenamiento; no guarda ni usa estos datos para entrenar ni para Kaggle.
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))

from gfs_forecast import get_gfs_forecast  # noqa: E402

VARS, UNID = ["RH", "T", "U", "V"], ["%", "K", "m/s", "m/s"]


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def comparar(desde, hasta):
    Xo, to = get_gfs_forecast(start_date=desde, end_date=hasta)
    for nombre in ("train", "test"):
        d = np.load(RAIZ / "data" / f"{nombre}.npz")
        tc = d["times"].astype(str)
        comunes = np.intersect1d(to, tc)
        if len(comunes) == 0:
            continue
        A = Xo[np.searchsorted(to, comunes)].astype(np.float64)
        B = d["X"][np.searchsorted(tc, comunes)].astype(np.float64)
        print(f"\n{nombre}.npz · {len(comunes)} horas comunes ({comunes[0]} a {comunes[-1]}) · "
              f"{len(np.setdiff1d(to, tc))} horas de Open-Meteo que no están en {nombre}.npz")
        print(f"{'var':<4}{'media OM':>10}{'media conc':>11}{'sd OM':>8}{'sd conc':>8}"
              f"{'sesgo':>8}{'RMSE':>8}{'dif máx':>9}{'r':>8}{'r N-S invertido':>17}")
        for c, v in enumerate(VARS):
            a, b = A[..., c], B[..., c]
            dif = a - b
            print(f"{v:<4}{a.mean():>10.3f}{b.mean():>11.3f}{a.std():>8.3f}{b.std():>8.3f}"
                  f"{dif.mean():>+8.3f}{np.sqrt((dif ** 2).mean()):>8.3f}{np.abs(dif).max():>9.4f}"
                  f"{corr(a, b):>8.4f}{corr(a, b[:, ::-1]):>17.4f}")
        ta, tb = A[..., 1].mean((1, 2)), B[..., 1].mean((1, 2))
        lag = {k: round(corr(ta[max(k, 0):len(ta) + min(k, 0)], tb[max(-k, 0):len(tb) - max(k, 0)]), 3)
               for k in (-2, -1, 0, 1, 2)}
        print(f"correlación de T media según desfase en horas: {lag}")


if __name__ == "__main__":
    comparar(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("2024-07-02", "2024-07-15")))

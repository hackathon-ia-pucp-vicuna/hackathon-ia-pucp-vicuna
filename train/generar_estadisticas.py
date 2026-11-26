"""
Genera models/estadisticas.npz a partir de la data oficial (T03 · HU-09). Solo NumPy.

El archivo no va al repo: `clim` (promedio del WRF de entrenamiento por hora UTC y celda) deriva de
la data de la competencia. En el repo solo están los 12 números de models/normalizacion.json, con los
que este script comprueba que la data y el corte temporal son los del entrenamiento.

Desde la raíz del repo, con data/train.npz descargado de Kaggle:
  python train/generar_estadisticas.py
"""
import json
from pathlib import Path

import numpy as np

import validacion as v

RAIZ = Path(__file__).resolve().parents[1]
RUTA_TRAIN = RAIZ / "data" / "train.npz"
RUTA_NORMALIZACION = RAIZ / "models" / "normalizacion.json"
RUTA_SALIDA = RAIZ / "models" / "estadisticas.npz"
TOLERANCIA = 1e-5                                    # relativa


def calcular(X, Y, times, inicio_val=v.INICIO_VAL):
    """mu_x, sd_x, clim y anom_sd calculados SOLO con las horas de entrenamiento."""
    t = np.asarray(times).astype(str)
    trn, _ = v.corte_temporal(t, inicio_val)
    clim = v.climatologia(Y[trn], t[trn])
    return {"mu_x": X[trn].mean((0, 1, 2)), "sd_x": X[trn].std((0, 1, 2)), "clim": clim,
            "anom_sd": (Y[trn] - clim[v.horas(t[trn])]).std((0, 1, 2))}


def comprobar(stats, referencia):
    """Falla si algún número calculado se aparta del publicado."""
    for clave, esperado in referencia.items():
        if not np.allclose(stats[clave], esperado, rtol=TOLERANCIA, atol=0):
            raise ValueError(f"{clave} calculado {np.asarray(stats[clave]).tolist()} no coincide con "
                             f"el publicado {esperado}: revisa data/train.npz.")


def generar(ruta_train=RUTA_TRAIN, ruta_salida=RUTA_SALIDA, ruta_normalizacion=RUTA_NORMALIZACION):
    """Escribe las estadísticas en `ruta_salida`, solo si coinciden con las publicadas."""
    if not Path(ruta_train).exists():
        raise FileNotFoundError(f"Falta {ruta_train}: descarga la data de la competencia (ver data/README.md).")
    datos = np.load(ruta_train)
    stats = calcular(datos["X"], datos["Y"], datos["times"])
    comprobar(stats, json.loads(Path(ruta_normalizacion).read_text(encoding="utf-8")))
    np.savez(ruta_salida, **stats)
    return Path(ruta_salida)


if __name__ == "__main__":
    print(f"Estadísticas escritas en {generar()}")

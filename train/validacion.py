"""
Validación temporal con la métrica del concurso (T02 · HU-09). Solo NumPy.

Corte temporal fijo: validación = últimas dos semanas de junio (desde 2024-06-16 00 UTC);
entrenamiento = todo lo anterior. Nunca se hace split aleatorio.

Hay dos scores y no son comparables entre sí:
  score_local    el del starter: cada RMSE se divide entre el RMSE de la climatología de train
                 medido en la propia validación (climatología = 0). Todas las horas.
  score_oficial  fórmula del leaderboard, con sus constantes fijas, sobre las horas que Kaggle
                 evalúa (00, 03, ..., 21 UTC). Es el que se usa para comparar experimentos.

El notebook lleva una copia exacta de este archivo en una celda para correr solo en Kaggle;
tests/test_validacion.py comprueba que ambas sean iguales.
"""
import numpy as np

VARS = ("RH", "T", "U", "V")
INICIO_VAL = "2024061600"
ESCALAS_OFICIALES = np.array([25.360, 2.0652, 2.6776, 2.0876])
HORAS_EVALUADAS = tuple(range(0, 24, 3))


def horas(times):
    """Hora UTC (0-23) de cada marca 'YYYYMMDDHH'."""
    return np.array([int(str(t)[-2:]) for t in times])


def corte_temporal(times, inicio_val=INICIO_VAL):
    """Máscaras booleanas (train, val): val son las marcas desde `inicio_val` en adelante."""
    val = np.asarray(times).astype(str) >= inicio_val
    trn = ~val
    if not trn.any() or not val.any():
        raise ValueError(f"El corte {inicio_val} deja vacío el entrenamiento o la validación.")
    return trn, val


def climatologia(Y, times):
    """Media por hora UTC: (24, filas, columnas, variables). Exige datos en las 24 horas."""
    h = horas(times)
    faltan = sorted(set(range(24)) - set(h.tolist()))
    if faltan:
        raise ValueError(f"No hay datos para las horas UTC {faltan}.")
    return np.stack([Y[h == k].mean(0) for k in range(24)])


def rmse_por_variable(pred, real):
    """RMSE de cada variable sobre todas las horas y celdas. Entradas (N, filas, columnas, 4)."""
    if pred.shape != real.shape:
        raise ValueError(f"Formas distintas: pred {pred.shape} y real {real.shape}.")
    rmse = np.sqrt(((pred - real) ** 2).mean(axis=(0, 1, 2)))
    if not np.isfinite(rmse).all():
        raise ValueError("Hay NaN o infinitos en la predicción o en el valor real.")
    return rmse


def score(rmse, escalas):
    """1 - media de rmse/escala; las cuatro variables pesan igual."""
    return float(1 - (rmse / escalas).mean())


def escala_climatologia(Y, times, inicio_val=INICIO_VAL):
    """RMSE en validación de la climatología calculada solo con train (denominador de score_local)."""
    t = np.asarray(times).astype(str)
    trn, val = corte_temporal(t, inicio_val)
    clim = climatologia(Y[trn], t[trn])
    return rmse_por_variable(clim[horas(t[val])], Y[val])


def validar(pred_val, Y, times, inicio_val=INICIO_VAL):
    """
    Evalúa una predicción de la validación.

    pred_val : (N_val, 70, 35, 4) en unidades físicas, en el orden de las horas de validación.
    Y, times : target y marcas de TODO train.npz (de ahí salen el corte y la climatología).
    """
    t = np.asarray(times).astype(str)
    _, val = corte_temporal(t, inicio_val)
    real, t_val = Y[val], t[val]
    rmse = rmse_por_variable(pred_val, real)
    evaluadas = np.isin(horas(t_val), HORAS_EVALUADAS)
    rmse_3h = rmse_por_variable(pred_val[evaluadas], real[evaluadas])
    return {
        "score_local": score(rmse, escala_climatologia(Y, t, inicio_val)),
        "rmse": {v: float(r) for v, r in zip(VARS, rmse)},
        "score_oficial": score(rmse_3h, ESCALAS_OFICIALES),
        "rmse_horas_evaluadas": {v: float(r) for v, r in zip(VARS, rmse_3h)},
        "horas_val": int(val.sum()),
        "horas_evaluadas": int(evaluadas.sum()),
        "primera": str(min(t_val)),
        "ultima": str(max(t_val)),
    }

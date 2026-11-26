"""
Ventanas (t-2, t-1, t) del GFS sin cruzar huecos de horas (T11 · HU-09). Solo NumPy.

La data del concurso tiene huecos de horas y de días. Una ventana está completa si existen las k horas
consecutivas que terminan en t. En las incompletas, la hora que falta se rellena con la hora t (igual que
el starter): sirven para predecir horas que Kaggle exige, pero se excluyen del entrenamiento.

Las horas previas se buscan por fecha, no por posición: nunca entra una hora del otro lado de un hueco.

El notebook lleva una copia exacta de este archivo en una celda para correr solo en Kaggle;
tests/test_ventanas.py comprueba que ambas sean iguales.
"""
from datetime import datetime, timedelta

import numpy as np


def indices_ventanas(times, k=3):
    """
    Índices de las horas (t-(k-1), …, t) de cada marca 'YYYYMMDDHH'.

    Devuelve (idx, completa): idx es (N, k), con el índice de t donde falta una hora previa;
    completa es una máscara (N,), True si las k horas existen.
    """
    fechas = [datetime.strptime(str(t), "%Y%m%d%H") for t in times]
    pos = {f: i for i, f in enumerate(fechas)}
    if len(pos) != len(fechas):
        raise ValueError("Hay horas repetidas en times.")
    idx = np.empty((len(fechas), k), np.int64)
    completa = np.ones(len(fechas), bool)
    for i, f in enumerate(fechas):
        for j, atras in enumerate(range(k - 1, -1, -1)):
            previa = pos.get(f - timedelta(hours=atras))
            if previa is None:
                completa[i] = False
                previa = i
            idx[i, j] = previa
    return idx, completa


def armar_ventanas(X, times, k=3):
    """Ventanas (N, k, ...) en float32 y la máscara de las completas."""
    if len(X) != len(times):
        raise ValueError(f"X tiene {len(X)} horas y times {len(times)}.")
    idx, completa = indices_ventanas(times, k)
    return np.asarray(X, np.float32)[idx], completa

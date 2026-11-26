# Ventanas con huecos de horas (T11 · HU-09)

El modelo recibe las horas t-2, t-1 y t del GFS. La data del concurso tiene huecos (horas o días sin datos), así que algunas ventanas no tienen sus 3 horas consecutivas. Las horas previas se buscan por fecha, de modo que nunca entra una hora del otro lado de un hueco; la que falta se rellenaba con la hora t, y el entrenamiento usaba esas ventanas como las demás.

## Tratamiento
| Uso | Ventanas incompletas | Motivo |
|---|---|---|
| Entrenamiento | **Se excluyen** | No tienen la entrada que el modelo espera |
| Validación (16–30 jun) | Se mantienen, rellenadas | El score sigue siendo comparable con T02 y T08; aparte se reporta el de las completas |
| Test | Se mantienen, rellenadas | Kaggle exige predecir las 792 horas |
| Producto (`app/entrada_gfs.py`, T35) | Se descartan | Sin cambios en esta tarea |

El criterio «ninguna ventana cruza un hueco» se lee así: ninguna ventana toma horas del otro lado de un hueco (en ningún conjunto) y ninguna ventana incompleta entrena el modelo.

La lógica está en `train/ventanas.py`; `train/t11_ventanas.ipynb` lleva una copia exacta. Los notebooks del starter, T02 y T08 no cambian, porque son el registro de corridas anteriores. El entrenamiento final (T24) debe usar este módulo.

## Antes y después
| Conjunto | Ventanas | Completas | Incompletas | Incompletas en horas evaluadas (00, 03, …, 21 UTC) |
|---|---|---|---|---|
| Entrenamiento (hasta el 15 jun) | 2,599 | 2,574 | 25 | 11 |
| Validación (16–30 jun) | 336 | 334 | 2 | 1 |
| **Todo `train.npz`** (entrenamiento final) | **2,935** | **2,908** | **27** | **12** |
| Test | 792 | 788 | 4 | 2 |

Las ventanas incompletas son las primeras horas de cada archivo y las que siguen a cada hueco: 13 huecos en train (281 horas) y 1 en test (24 horas). `tests/test_ventanas.py` verifica los conteos con la data oficial.

## Huecos en la data
| Archivo | Última hora antes del hueco (UTC) | Primera hora después (UTC) | Horas faltantes |
|---|---|---|---|
| train | 04/03 20:00 | 05/03 00:00 | 3 |
| train | 05/03 20:00 | 06/03 00:00 | 3 |
| train | 20/03 22:00 | 26/03 00:00 | 121 |
| train | 31/03 21:00 | 01/04 00:00 | 2 |
| train | 15/04 04:00 | 15/04 06:00 | 1 |
| train | 17/04 22:00 | 18/04 00:00 | 1 |
| train | 23/04 23:00 | 28/04 00:00 | 96 |
| train | 28/04 12:00 | 28/04 14:00 | 1 |
| train | 28/04 14:00 | 28/04 17:00 | 2 |
| train | 29/04 21:00 | 30/04 00:00 | 2 |
| train | 17/05 05:00 | 17/05 07:00 | 1 |
| train | 30/05 23:00 | 01/06 00:00 | 24 |
| train | 21/06 23:00 | 23/06 00:00 | 24 |
| test | 30/07 23:00 | 01/08 00:00 | 24 |

## Experimento
Configuración del ensamble de T08 (pérdida `rmse`, 25 épocas, HR recortada, 5 semillas), entrenando con las 2,599 horas frente a las 2,574 completas, con la misma climatología y normalización. Regla fijada antes de correr: se adopta la exclusión salvo que empeore el `score_oficial` en las 5 semillas, o que el ensamble caiga más que la desviación entre semillas de la rama con todas las horas.

Corrida: Kaggle privado `johancg04/vicuna-t11-ventanas` v1 (Tesla T4, Python 3.13.16, NumPy 2.1.3, torch 2.11.0), sin envío a la competencia. El notebook imprimió los conteos de la tabla de arriba y `pytest tests/` dio 94 passed.

Ensamble de 5 semillas, validación del 16–30 jun, horas evaluadas:

| Entrenamiento | score_oficial | RH (%) | T (K) | U (m/s) | V (m/s) | score solo en ventanas completas |
|---|---|---|---|---|---|---|
| Todas (2,599) | 0.45329 | 9.329 | 0.947 | 1.648 | 1.556 | 0.45155 |
| Completas (2,574) | 0.45223 | 9.391 | 0.951 | 1.648 | 1.555 | 0.45068 |

| Semilla | Todas | Completas | Diferencia |
|---|---|---|---|
| 0 | 0.44279 | 0.43711 | −0.00568 |
| 1 | 0.43911 | 0.44159 | +0.00248 |
| 2 | 0.43515 | 0.43615 | +0.00100 |
| 3 | 0.43417 | 0.43428 | +0.00011 |
| 4 | 0.43762 | 0.43445 | −0.00317 |

**Decisión: se adopta la exclusión**, según la regla fijada antes de correr. No mejora el score: la diferencia media (−0.00105) y la del ensamble (−0.00106) son menores que la desviación entre semillas (0.00342).

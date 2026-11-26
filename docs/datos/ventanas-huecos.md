# Ventanas con huecos de horas (T11 · HU-09)

El modelo recibe las horas t-2, t-1 y t del GFS. La data del concurso tiene huecos (horas o días sin datos), así que algunas ventanas no tienen sus 3 horas consecutivas. Antes, el notebook las rellenaba repitiendo la hora t y entrenaba con ellas como si fueran datos reales.

## Cambio en `train/starter_notebook.ipynb`
- `windows()` devuelve también `ok`: `False` si falta t-2 o t-1 (la ventana cruza un hueco).
- Entrenamiento: `itr = np.where(trn & ok_tr)[0]`. Validación: `vok = val & ok_tr`. Ninguna ventana que cruza un hueco se usa para entrenar ni para medir el score local.
- Test: no se descarta ninguna hora, porque Kaggle exige predecir las 792. Las 4 ventanas incompletas se predicen repitiendo la hora t en el lugar de la que falta (comportamiento anterior, ahora documentado).
- El producto hace lo mismo que el entrenamiento: `app/entrada_gfs.py` (T35) descarta las horas sin t-2/t-1.

## Antes y después
| Conjunto | Ventanas antes | Cruzan un hueco | Ventanas que quedan |
|---|---|---|---|
| Entrenamiento (hasta 15 jun) | 2,599 | 25 | **2,574** |
| Validación (16–30 jun) | 336 | 2 | **334** |
| **Train total** | **2,935** | **27** | **2,908** |
| Test (se predicen todas) | 792 | 4 | 792 (4 con relleno) |

Las ventanas incompletas son las 2 primeras horas tras cada hueco y las 2 primeras horas de cada archivo: 13 huecos en train y 1 en test. Los conteos los imprime el notebook en la celda de ventanas, y `tests/test_ventanas_entrenamiento.py` los verifica con `data/train.npz`.

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

## Efecto en el score local
La validación pasa de 336 a 334 horas, así que el score local cambia muy poco respecto de los experimentos anteriores. Para comparar experimentos, conviene volver a correr la línea base con este notebook. Prueba rápida (`QUICK=1`, 2 épocas en CPU): corre sin errores, con 2,574 ventanas de entrenamiento y 334 de validación.

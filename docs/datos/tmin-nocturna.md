# T mínima nocturna: modelo vs. GFS crudo (T36 · HU-04)

**Pregunta:** el aviso de helada usa la T mínima de la noche. ¿Nuestro modelo la estima mejor que el GFS crudo interpolado a 3 km?

**Respuesta corta:** sí. El error típico de la T mínima nocturna baja de **3.3 °C (GFS) a 0.9 °C (modelo)** en toda la grilla y de **2.8 °C a 0.6 °C** en Carhuaz, Huaraz y Yungay. Pero el modelo tiene un **sesgo cálido de +0.3 °C**: en la zona de cultivo dejó pasar más de la mitad de las celdas con helada, que fueron muy pocas (33 de 20,258 celdas-noche).

## Método
- **Data:** validación temporal de `train.npz` (16–30 jun 2024). El 22 jun no tiene datos, así que quedan **14 noches** y todas están completas. Nunca se usó para entrenar: los pesos (`models/modelo.pt`, línea base T03) se entrenaron hasta el 15 jun. Solo se eligió la mejor época con esta validación, lo que puede favorecer un poco al modelo.
- **Noche (la de este análisis; desde el 28 nov el producto usa 01–12 UTC y estas tablas no se han recalculado):** 19:00–07:00 en Lima, que corresponde a 00–12 UTC. Para cada noche y celda se toma la T mínima de dos formas: con las 13 horas, y con solo las horas 00, 03, 06, 09 y 12 UTC, que son las que evalúa Kaggle y las que usa el aviso.
- **Fuentes:** la referencia es el WRF (Y). El modelo se corre con `predecir()` sobre ventanas t-2, t-1, t (`train/ventanas.py`). El GFS interpolado es una interpolación bilineal de la grilla 17×13 a la de 3 km con las coordenadas reales, sin corregir por altura.
- **Ámbitos:** toda la grilla (2,450 celdas), las celdas a 4,000 m o menos según el DEM (1,447 celdas, tope aproximado de la papa) y las 3 localidades de `app/frost/zones.yaml`.
- **Para reproducir:** `python scripts/evaluar_tmin_nocturna.py` en el commit `6823f` (antes de T49), con `data/` y `models/estadisticas.npz`. Desde T49, `predecir()` carga el ensamble final, entrenado también con estas noches, y el script deja de ser una validación (ver la última sección).

## Resultados (T mínima con las 13 horas, 14 noches)

| Ámbito | Fuente | RMSE (°C) | MAE (°C) | Sesgo (°C) |
|---|---|---|---|---|
| Toda la grilla | **Modelo** | **0.93** | **0.72** | +0.35 |
| Toda la grilla | GFS interpolado | 3.26 | 2.59 | +0.05 |
| ≤ 4,000 m | **Modelo** | **0.92** | **0.72** | +0.30 |
| ≤ 4,000 m | GFS interpolado | 3.09 | 2.49 | −0.91 |
| 3 localidades | **Modelo** | **0.56** | **0.47** | +0.25 |
| 3 localidades | GFS interpolado | 2.78 | 2.27 | −1.90 |

Con solo las horas 00, 03, 06, 09 y 12 UTC, los números casi no cambian. El modelo da 0.91 / 0.90 / 0.56 °C de RMSE y el GFS 3.23 / 3.04 / 2.46 °C.

## ¿Hubo noches con T ≤ 0 °C?

| Ámbito | Noches con alguna celda ≤ 0 °C (WRF) | Celdas-noche ≤ 0 °C | Modelo: detecta / falsas alarmas | GFS: detecta / falsas alarmas |
|---|---|---|---|---|
| Toda la grilla | 14 de 14 | 4,811 (14.0 %) | 68 % / 10 % | 38 % / 62 % |
| ≤ 4,000 m | 12 de 14 | 33 (0.2 %) | 42 % / 44 % | 48 % / 98.5 % |

- Las heladas de la validación se dieron casi todas por encima de los 4,000 m: nevados y punas, no la zona de cultivo.
- En las 3 localidades **no hubo helada**. La T mínima más baja del WRF fue 4.5 °C en Carhuaz, 4.4 °C en Huaraz y 5.5 °C en Yungay. El modelo dio 5.4, 4.5 y 6.6 °C, y el GFS 0.6, 5.2 y 3.8 °C.
- En la zona de cultivo, el GFS avisa helada en 1,038 celdas-noche cuando hubo 33: el 98.5 % de sus avisos son falsas alarmas. El modelo avisa en 25 y acierta en 14 (detecta el 42 %).

## Conclusión
- El modelo mejora mucho la T mínima nocturna frente al GFS crudo: el RMSE baja entre 3 y 5 veces según el ámbito. La comparación "GFS crudo vs. nuestro modelo" del producto tiene sustento.
- **Lo que no mejora:** el modelo es un poco cálido (+0.3 °C) y suaviza los extremos, así que **subestima las heladas**. En la zona de cultivo detectó menos de la mitad de las pocas celdas con helada.
- **Para el producto (sugerencia para `app/frost/rules.py`, T15):** usar un margen de seguridad sobre el umbral, por ejemplo avisar "riesgo" desde +1 °C, y declarar la incertidumbre de unos ±1 °C en el mensaje.
- **Límites:** 14 noches de junio, una sola temporada, pocas heladas en la zona de cultivo (33 casos) y la referencia es el WRF, no estaciones de SENAMHI. Hay que repetir el cálculo con los pesos finales del envío (T50).

## ¿Y con el modelo final del producto (T50)?
No se puede repetir esta evaluación con los 10 pesos del envío 58945688: se reentrenaron con **todos** los datos de `train.npz`, incluidas estas 14 noches. Medirlos aquí sería evaluarlos con noches que ya vieron, y el resultado saldría mejor de lo real.

Lo que sí se sabe: la misma configuración, entrenada solo hasta el 15 jun, tiene un RMSE de T horaria de **0.836 °C** en esta validación, frente a **0.959 °C** de la línea base (horas que evalúa Kaggle; `docs/modelo/experimentos.md`). El modelo final debería estimar la T mínima nocturna igual o mejor que la línea base, pero esta evaluación no lo mide directamente. Las cifras de arriba son una estimación conservadora.

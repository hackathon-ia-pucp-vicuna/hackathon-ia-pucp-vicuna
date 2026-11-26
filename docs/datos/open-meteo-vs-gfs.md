# GFS de Open-Meteo vs. GFS del concurso (T13 · HU-04)

**Pregunta:** ¿la entrada que recibe el producto (GFS de Open-Meteo, `app/gfs_forecast.py`) es igual a la que vio el modelo al entrenar (X de `train.npz` y `test.npz`)? Si no lo fuera, el modelo recibiría en producción datos distintos a los de su entrenamiento.

**Respuesta corta:** sí, son idénticos. En 480 horas comparadas no hay ninguna diferencia, ni en una sola celda ni en una sola variable. No hace falta corregir sesgos ni conversiones. El riesgo que queda es otro: el plazo del pronóstico (ver "Lo que no se pudo comparar").

## Método
- Se descargó el archivo histórico del GFS de Open-Meteo (`historical-forecast-api`, modelo `gfs_global`) con la misma función que usa el producto: `get_gfs_forecast(start_date, end_date)`. Misma grilla 17×13, mismo orden de variables (RH, T, U, V), mismas conversiones (°C → K; rapidez y dirección → U, V).
- Se comparó hora a hora y celda a celda contra la X del concurso, en dos periodos:
  - **test:** 2–15 jul 2024 (336 horas)
  - **train:** 16–21 jun 2024 (144 horas, dentro de la validación temporal)
- Además de media, desviación y diferencia, se revisaron tres errores típicos: orientación invertida (correlación con la grilla volteada norte-sur), desfase de horas (correlación desplazando ±1 y ±2 h) y unidades (rangos y pendiente del ajuste lineal).
- Reproducir: `python scripts/comparar_open_meteo_gfs.py 2024-07-02 2024-07-15` y `python scripts/comparar_open_meteo_gfs.py 2024-06-16 2024-06-22`. El script no guarda los datos descargados.

## Resultados

### Test (2–15 jul 2024, 336 horas)
| Variable | Media OM | Media concurso | Desv. OM | Desv. concurso | Diferencia media (sesgo) | RMSE | Diferencia máx. | r |
|---|---|---|---|---|---|---|---|---|
| RH (%) | 73.941 | 73.941 | 18.735 | 18.735 | 0.000 | 0.000 | 0.0000 | 1.0000 |
| T (K) | 286.967 | 286.967 | 6.849 | 6.849 | 0.000 | 0.000 | 0.0000 | 1.0000 |
| U (m/s) | −0.790 | −0.790 | 1.900 | 1.900 | 0.000 | 0.000 | 0.0000 | 1.0000 |
| V (m/s) | 1.123 | 1.123 | 2.845 | 2.845 | 0.000 | 0.000 | 0.0000 | 1.0000 |

### Train (16–21 jun 2024, 144 horas)
| Variable | Media OM | Media concurso | Desv. OM | Desv. concurso | Diferencia media (sesgo) | RMSE | Diferencia máx. | r |
|---|---|---|---|---|---|---|---|---|
| RH (%) | 78.473 | 78.473 | 15.735 | 15.735 | 0.000 | 0.000 | 0.0000 | 1.0000 |
| T (K) | 286.742 | 286.742 | 6.788 | 6.788 | 0.000 | 0.000 | 0.0000 | 1.0000 |
| U (m/s) | −0.444 | −0.444 | 1.826 | 1.826 | 0.000 | 0.000 | 0.0000 | 1.0000 |
| V (m/s) | 1.336 | 1.336 | 2.851 | 2.851 | 0.000 | 0.000 | 0.0000 | 1.0000 |

El 100 % de los valores es idéntico bit a bit (diferencia máxima exactamente 0 en float32).

### Controles
- **Orientación:** con la grilla del concurso volteada norte-sur, la correlación cae a 0.06–0.30 (T: 0.10; U: ≈0). Con la orientación del producto es 1.0. Fila 0 = norte en ambos.
- **Desfase de horas:** la correlación de la T media es máxima sin desfase (1.0) y baja a 0.95 con ±1 h y a 0.82 con ±2 h. Las horas están alineadas en UTC.
- **Unidades:** rangos idénticos (T 264.8–307.1 K; RH 7–100 %); ajuste lineal con pendiente 1.000 e intercepto 0.00 en las 4 variables.
- **Huecos:** Open-Meteo tiene 24 horas (22 jun 2024) que `train.npz` no trae. Los huecos de horas son de la data del concurso, no de la fuente (relevante para T11).

## Conclusión
- **No hace falta corregir sesgos ni conversiones.** El GFS del concurso es el mismo archivo histórico de Open-Meteo que descarga `gfs_forecast.py`, con las mismas conversiones, orientación y horas. Lo más probable es que los organizadores generaran la data con esta misma función.
- El adaptador del producto (`app/entrada_gfs.py`, T35) puede pasar la salida de Open-Meteo a `predecir()` tal cual, con la misma normalización del entrenamiento.

## Lo que no se pudo comparar (con honestidad)
1. **Plazo del pronóstico.** El archivo histórico de Open-Meteo se arma con las primeras horas de cada corrida del GFS (plazo corto, casi un análisis). En producción, el aviso de la noche usa el pronóstico vigente, con un plazo de varias horas a más de un día según cuándo se consulte, y su error crece con el plazo. Lo comparado aquí no mide esa degradación. El modelo se entrenó con plazo corto, así que en producción puede rendir algo peor que en la validación. Para medirlo haría falta guardar los pronósticos del producto varios días y compararlos luego con el archivo histórico de esas mismas horas.
2. **Cobertura.** Se compararon 480 de las 3,727 horas (21 días de 2 periodos), no todo el entrenamiento. Al ser idéntico bit a bit en ambos, es poco probable que otros días difieran, pero no está verificado.
3. **Cambios futuros de la fuente.** Si Open-Meteo cambia la versión del GFS o su interpolación, la entrada del producto podría dejar de coincidir. Conviene repetir este script antes del video y de la entrega.
4. **Ninguna comparación contra observaciones reales.** Esto compara entradas entre sí; no dice qué tan bien el GFS (o el modelo) acierta la temperatura real en los pueblos. Eso es T36 (modelo vs. GFS crudo) y la prueba con un usuario real (T28).

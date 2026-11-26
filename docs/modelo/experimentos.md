# Registro de experimentos del modelo

Una fila por experimento. Validación: corte temporal (últimas dos semanas de junio), métrica oficial. No ajustar al leaderboard público.

| Fecha | Quién | Hipótesis / cambio | Validación (score y RMSE por variable) | Público en Kaggle | Decisión |
|---|---|---|---|---|---|
| 24 nov | Johan | Starter oficial, 25 épocas, CPU | val interna del starter 0.3377 | 0.42782 | Línea base |
| 25 nov | Johan | T01: mismo starter en Kaggle con GPU T4 x2 (v3 del notebook; `torch.cuda.is_available()` = True, Tesla T4). **Tiempo por época: ≈1.3 s** (primera: 5 s); en CPU (v1): ≈42 s. Corrida completa: 78.6 s | val interna del starter 0.3377 (época 21): RH 9.921, T 0.973, U 1.719, V 1.609 | no enviado | Entrenar en GPU |
| 25 nov | Johan | T02: `train/t02_linea_base.ipynb` (red e hiperparámetros del starter) con la validación de `train/validacion.py`. Kaggle, Tesla T4, Python 3.13.16, NumPy 2.1.3, PyTorch 2.11.0. Corte: train 2599 h (18 feb – 15 jun), val 336 h (16 – 30 jun) | **score_local 0.33768** (dif. −0.00002 vs 0.3377): RH 9.921, T 0.973, U 1.719, V 1.609. **score_oficial 0.44092** (112 h evaluadas): RH 9.905, T 0.959, U 1.682, V 1.572 | no enviado | Línea base reproducida; comparar experimentos con `score_oficial` |
| 25 nov | Johan | T03: pesos y estadísticas de la corrida T02 recargados con `predecir()` en un proceso nuevo, en CPU (Kaggle `johancg04/vicuna-t03-verificacion` v1, commit `abc555e`). Sin reentrenar | score_local 0.33768 (dif. < 1e-8), score_oficial 0.44092 (dif. −6.5e-8); mismo RMSE por variable que T02 (±3e-6) | no enviado | `models/` reproduce la línea base; ver `models/FICHA.md` |

## Cómo leer los scores

- `score_local`: el del starter. Divide cada RMSE entre el de la climatología en la validación (RH 17.177, T 2.010, U 2.261, V 1.946).
- `score_oficial`: fórmula del leaderboard (constantes 25.360, 2.0652, 2.6776, 2.0876) sobre las horas 00, 03, …, 21 UTC de la validación.
- La mejor época se elige mirando la validación, así que ambos scores son algo optimistas.

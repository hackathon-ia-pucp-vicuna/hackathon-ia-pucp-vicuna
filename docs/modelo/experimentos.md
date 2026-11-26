# Registro de experimentos del modelo

Una fila por experimento. Validación: corte temporal (últimas dos semanas de junio), métrica oficial. No ajustar al leaderboard público.

| Fecha | Quién | Hipótesis / cambio | Validación (score y RMSE por variable) | Público en Kaggle | Decisión |
|---|---|---|---|---|---|
| 24 nov | Johan | Starter oficial, 25 épocas, CPU | val interna del starter 0.3377 | 0.42782 | Línea base |
| 25 nov | Johan | T01: mismo starter en Kaggle con GPU T4 x2 (v3 del notebook; `torch.cuda.is_available()` = True, Tesla T4). **Tiempo por época: ≈1.3 s** (primera: 5 s); en CPU (v1): ≈42 s. Corrida completa: 78.6 s | val interna del starter 0.3377 (época 21): RH 9.921, T 0.973, U 1.719, V 1.609 | no enviado | Entrenar en GPU |
| 25 nov | Johan | T02: `train/t02_linea_base.ipynb` (red e hiperparámetros del starter) con la validación de `train/validacion.py`. Kaggle, Tesla T4, Python 3.13.16, NumPy 2.1.3, PyTorch 2.11.0. Corte: train 2599 h (18 feb – 15 jun), val 336 h (16 – 30 jun) | **score_local 0.33768** (dif. −0.00002 vs 0.3377): RH 9.921, T 0.973, U 1.719, V 1.609. **score_oficial 0.44092** (112 h evaluadas): RH 9.905, T 0.959, U 1.682, V 1.572 | no enviado | Línea base reproducida; comparar experimentos con `score_oficial` |
| 25 nov | Johan | T03: pesos y estadísticas de la corrida T02 recargados con `predecir()` en un proceso nuevo, en CPU (Kaggle `johancg04/vicuna-t03-verificacion` v1, commit `abc555e`). Sin reentrenar | score_local 0.33768 (dif. < 1e-8), score_oficial 0.44092 (dif. −6.5e-8); mismo RMSE por variable que T02 (±3e-6) | no enviado | `models/` reproduce la línea base; ver `models/FICHA.md` |
| 25 nov | Johan | **T08** (hipótesis: una pérdida alineada con la métrica mejora). `train/t08_experimentos.ipynb` (Kaggle `johancg04/vicuna-t08-experimentos` v2), T4, época final fija, 2 semillas, HR recortada: `mse` (starter), `wmse` (MSE ponderada por `(anom_sd/escala)²`) y `rmse` (media de RMSE/escala del lote) | score_oficial medio a 25 épocas: **rmse 0.44095** (RH 9.571, T 0.964, U 1.679, V 1.597) · mse 0.44023 (9.681, 0.968, 1.685, 1.585) · wmse 0.43817 (9.728, 0.983, 1.682, 1.585). Rango entre semillas hasta 0.004. Semillas fijas; la semilla 0 de `rmse` a 25 épocas, repetida en el mismo proceso, difiere 4e-8; en un proceso nuevo (Kaggle `johancg04/vicuna-t08-repro` v2, mismas versiones y T4) da 0.442792, diferencia 6e-8. Entrena también con las ventanas que cruzan huecos de horas | no enviado | La pérdida no da mejora distinguible del ruido; se usa `rmse` (decidido con la validación, sin mirar el leaderboard) |

## Cómo leer los scores

- `score_local`: el del starter. Divide cada RMSE entre el de la climatología en la validación (RH 17.177, T 2.010, U 2.261, V 1.946).
- `score_oficial`: fórmula del leaderboard (constantes 25.360, 2.0652, 2.6776, 2.0876) sobre las horas 00, 03, …, 21 UTC de la validación.
- La mejor época se elige mirando la validación, así que ambos scores son algo optimistas.

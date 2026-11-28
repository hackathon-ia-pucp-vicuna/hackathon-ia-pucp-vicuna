# train/ — Entrenamiento del modelo (Kaggle)

| Archivo | Qué es |
|---|---|
| `starter_notebook.ipynb` | Starter oficial: CNN chica con relieve (score público de la línea base: 0.42782). |
| `plantilla_experimento.ipynb` | Plantilla para un experimento: cada quien cambia una celda y la corre en su Kaggle (cortes A y B, 2 semillas). |
| `submission_notebook.ipynb` | Notebook final reproducible: entrena el modelo y genera el `submission.csv` de la submission final (T29). |

Reglas:
- Solo la data oficial de la competencia (`data/`). Nada de Open-Meteo ni de otros datasets.
- Validación siempre con corte temporal (últimas dos semanas de junio), nunca con split aleatorio.
- Semillas fijas, rutas relativas y salidas guardadas. Debe correr en Kaggle o en Colab.
- El código de entrenamiento **no importa nada de `app/`**. La red se define dentro del notebook; `app/predict.py` tiene una copia y `tests/test_paridad.py` comprueba que coinciden.
- Los pesos se guardan en `models/` (solo `state_dict`, sin el estado del optimizador).
- Cada experimento queda en `docs/modelo/experimentos.md` antes de enviar a Kaggle.

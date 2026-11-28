# Licencias

Licencia de cada dependencia, modelo, dato y servicio que usa el repositorio (T40). Regla del equipo (CLAUDE.md §5): toda dependencia nueva se anota aquí, con versión fija en su `requirements`.

Revisado el 28 nov 2026. Las licencias de los paquetes se tomaron de sus metadatos (`pip show`) y de sus repositorios oficiales.

## Código del repositorio

| Qué | Licencia |
|---|---|
| Código del equipo Vicuña (`app/`, `train/`, `tests/`, `scripts/`) | MIT (`LICENSE`) |

## Dependencias de Python

### Producto, inferencia en CPU (`requirements-app.txt`, igual a `app/web/requirements.txt`)

| Paquete | Versión | Licencia |
|---|---|---|
| torch (PyTorch, CPU) | 2.5.1+cpu | BSD-3-Clause |
| streamlit | 1.40.2 | Apache-2.0 |
| numpy | 2.1.3 | BSD-3-Clause |
| pandas | 2.2.3 | BSD-3-Clause |
| requests | 2.32.3 | Apache-2.0 |
| PyYAML | 6.0.2 | MIT |

### Pruebas y calidad (`requirements-dev.txt`)

| Paquete | Versión | Licencia |
|---|---|---|
| pytest | 8.3.4 | MIT |
| ruff | 0.8.4 | MIT |

### Entrenamiento (`requirements.txt`, en Kaggle/Colab)

| Paquete | Licencia |
|---|---|
| torch | BSD-3-Clause |
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| matplotlib | Matplotlib License (basada en la PSF, compatible con BSD) |
| requests | Apache-2.0 |
| jupyter | BSD-3-Clause |

La biblioteca estándar de Python (`sqlite3`, `json`, `argparse`, …) está bajo la PSF License.

## Modelos

| Qué | Licencia y condiciones |
|---|---|
| Pesos del equipo (`models/modelo.pt`, `models/ensamble/*.pt`, `models/modelo.json`) | Entrenados desde cero por el equipo solo con la data oficial; no se usa ningún modelo preentrenado. El código que los genera es MIT, pero los pesos derivan de data CC BY-NC-SA 4.0: se publican solo para la hackathon y **no se permite su uso comercial**. |
| Arquitectura base del `starter_notebook.ipynb` | Plantilla entregada por la organización de la hackathon para uso en la competencia. |

## Datos

| Fuente | Licencia y uso |
|---|---|
| Data oficial de la competencia (`train.npz`, `test.npz`, `static.npz`) | CC BY-NC-SA 4.0. No se redistribuye y no va al repo (`.gitignore`). Los derivados (p. ej. la climatología `clim`) tampoco se publican (ver `models/FICHA.md`). |
| GFS de NOAA, servido por Open-Meteo (`app/gfs_forecast.py`, solo producto) | Datos de Open-Meteo bajo CC BY 4.0: hay que atribuir a Open-Meteo. Uso no comercial gratuito según sus términos. El GFS original de NOAA es de dominio público. Nunca se usa para entrenar ni para el submission. |
| Umbrales de helada (`app/frost/thresholds.yaml`) | Cada umbral cita su `fuente` (SENAMHI, INIA u otra); se usan como referencia con cita, no se copian documentos. |

## Servicios y herramientas

| Servicio | Condiciones |
|---|---|
| Telegram Bot API (`app/notify/telegram.py`) | Gratuito, bajo los términos de servicio de Telegram. Se llama por HTTP con `requests`, sin bibliotecas extra. |
| Streamlit Community Cloud (despliegue) | Gratuito, bajo los términos de Streamlit/Snowflake. |
| GitHub Actions (`.github/workflows/ci.yml`) | Gratuito para repos públicos. Acciones usadas: `actions/checkout@v4` y `actions/setup-python@v5`, ambas MIT. |
| Kaggle (entrenamiento y envíos) | Términos de Kaggle y reglas de la competencia. |

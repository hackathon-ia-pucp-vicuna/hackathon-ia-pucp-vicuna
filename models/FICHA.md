# Ficha del modelo — línea base (T03)

Equipo Vicuña · Hackathon IA PUCP 2026. Reducción de escala espacial: del GFS (~27 km) a lo que daría WRF (3 km) en la Cordillera Blanca.

**Estos pesos no se han enviado a Kaggle.** El score público 0.42782 es de otra corrida del mismo starter (CPU, 24 nov), cuyos pesos no son estos.

## Archivos

| Archivo | Contenido | Tamaño (bytes) | SHA-256 |
|---|---|---|---|
| `modelo.pt` | `state_dict` de `Net` (11 tensores float32), sin optimizador | 579,761 | `ae4b86b29909dd72abc29a52575487fdfd1de574b5fc1f7fdb1ae40b4801bc1c` |
| `estadisticas.npz` | `mu_x` (4), `sd_x` (4), `clim` (24, 70, 35, 4), `anom_sd` (4), float32 | 941,836 | `5ba6ef49730e7923313ab857c9cfedd263aff198de7c05cd7c58cf71e8dcb0dd` |

Pesan en total unos 1.5 MB, muy por debajo de los 100 MB: no hace falta un enlace externo.

Los tensores se guardaron desde GPU: para cargarlos en CPU se usa `map_location="cpu"` (lo hace `app/predict.py`).

`app/predict.py` necesita además `data/static.npz` (relieve y grillas), que es data de la competencia y no va al repo.

### Estado y licencia de `estadisticas.npz`

`clim` es el promedio del WRF de entrenamiento por hora UTC y celda; las otras tres claves son medias y desviaciones. Son un **derivado de la data oficial, de licencia CC BY-NC-SA** (organización de la Hackathon IA PUCP 2026). Por eso este archivo se rige por CC BY-NC-SA (atribución, uso no comercial, misma licencia) y **queda fuera de la licencia del código**.

Se publica en este repo por decisión de johancg04 (25 nov); **los organizadores aún no lo han confirmado**. Si indican que no se puede, se retira y se regenera, desde la raíz del repo y con `data/train.npz`:

```python
import sys
import numpy as np
sys.path.insert(0, "train")
import validacion as v

d = np.load("data/train.npz")
X, Y, t = d["X"], d["Y"], d["times"].astype(str)
trn, _ = v.corte_temporal(t)
clim = v.climatologia(Y[trn], t[trn])
np.savez("models/estadisticas.npz", mu_x=X[trn].mean((0, 1, 2)), sd_x=X[trn].std((0, 1, 2)),
         clim=clim, anom_sd=(Y[trn] - clim[v.horas(t[trn])]).std((0, 1, 2)))
```

El archivo regenerado tiene los mismos valores (comprobado el 25 nov, diferencia máxima 0.0); su SHA-256 puede cambiar.

## Entrada y salida

- Entrada: `(N, 3, 17, 13, 4)`, horas t-2, t-1 y t del GFS en unidades físicas, y la hora UTC de t. Fila 0 = norte. Si falta una hora previa por un hueco, se repite la hora t.
- Salida: `(N, 70, 35, 4)` en unidades físicas. Fila 0 = sur.
- Orden de las variables: RH (%), T (K), U (m/s), V (m/s).
- Punto de entrada: `predecir(Xw, horas_utc)` en `app/predict.py`.

## Arquitectura

`Net` del starter, 143,956 parámetros:

1. Dos convoluciones 3×3 (ancho 64, GELU) sobre 14 canales: 3 horas × 4 variables, más seno y coseno de la hora.
2. Interpolación a la grilla de 3 km con `grid_sample` y las coordenadas reales.
3. Se añaden el DEM normalizado (1 canal) y un mapa estático aprendido (8 canales).
4. Tres convoluciones (3×3, 3×3, 1×1) que predicen la **anomalía normalizada** respecto de la climatología de esa hora.

Predicción = `clim[hora] + anomalía × anom_sd`.

## Entrenamiento

- Data: `train.npz` oficial. Entrenamiento 2599 h (18 feb – 15 jun 2024); validación 336 h (16 – 30 jun 2024). Corte temporal, sin split aleatorio.
- Normalización y climatología calculadas solo con las horas de entrenamiento.
- Pérdida MSE sobre la anomalía normalizada. AdamW (lr 2e-3, weight decay 1e-4), OneCycleLR, lote 32, 25 épocas, semilla 0 (PyTorch y NumPy).
- Se guarda la mejor época según la validación (época 21).
- Notebook: `train/t02_linea_base.ipynb`. Corrida de origen: notebook privado de Kaggle `johancg04/notebookb2a9db6afe`, versión 1, Tesla T4, Python 3.13.16, NumPy 2.1.3, PyTorch 2.11.0.
- Repetir el entrenamiento en GPU puede dar pesos ligeramente distintos.

## Validación (16 – 30 jun)

| Score | Horas | Resultado | RMSE RH (%) | RMSE T (K) | RMSE U (m/s) | RMSE V (m/s) |
|---|---|---|---|---|---|---|
| `score_local` (denominador: RMSE de la climatología) | 336 | 0.337680 | 9.921 | 0.973 | 1.719 | 1.609 |
| `score_oficial` (constantes del leaderboard; 00, 03, …, 21 UTC) | 112 | 0.440924 | 9.905 | 0.959 | 1.682 | 1.572 |

## Cómo comprobar que los archivos reproducen el score

Con `modelo.pt` y `estadisticas.npz` en `models/`, `train.npz` y `static.npz` en `data/`, y PyTorch instalado:

```bash
pytest -q -s tests/test_reproduce_score.py tests/test_paridad.py
```

El primero carga los archivos con `predecir()` en CPU y exige una diferencia menor que 1e-4 en ambos scores.

Última comprobación (25 nov, commit `abc555e`): script privado de Kaggle `johancg04/vicuna-t03-verificacion`, versión 1, en CPU (Python 3.13.18, NumPy 2.1.3, PyTorch 2.11.0, pandas 2.3.3), con `estadisticas.npz` tomado de la salida del entrenamiento. 9 pruebas pasaron.

| Score | En CPU, proceso nuevo | Entrenamiento | Diferencia |
|---|---|---|---|
| `score_local` | 0.33767968 | 0.33767968 | menor que 1e-8 |
| `score_oficial` | 0.44092408 | 0.44092415 | -6.5e-8 |

### Con las versiones del producto (`torch==2.5.1+cpu`)

25 nov, commit `7e704f1`: script privado de Kaggle `johancg04/vicuna-t03-torch251`, versión 1, en CPU. Entorno de Python 3.12.3 creado con `uv` (el Python de Kaggle es 3.13), con torch 2.5.1+cpu, NumPy 2.1.3 y pandas 2.2.3, las versiones de `requirements-app.txt`, y pytest 9.2.0.

- 9 pruebas pasaron, incluida la paridad con los pesos reales.
- `score_local` 0.33767968 (diferencia con el entrenamiento menor que 1e-8) y `score_oficial` 0.44092408 (-6.5e-8).
- En las 336 horas de validación, la diferencia máxima frente a las predicciones con torch 2.11.0 fue 0.0 en las cuatro variables.
- `predecir()` para 24 horas, con los pesos reales: 0.151 s en la CPU de Kaggle.

No se instaló Streamlit ni se probó el despliegue.

## Limitaciones

- La época se elige mirando la validación, así que los scores son algo optimistas.
- Solo se entrenó con febrero–junio; el test (julio–agosto) es plena estación seca.
- No predice lluvia, nubosidad, radiación ni ráfagas.
- En el producto la entrada es el GFS de Open-Meteo, que puede diferir del GFS del concurso; el desempeño con esa fuente no está medido aquí.

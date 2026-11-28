# Ficha del modelo

Equipo Vicuña · Hackathon IA PUCP 2026. Reducción de escala espacial: del GFS (~27 km) a lo que daría WRF (3 km) en la Cordillera Blanca.

## Ensamble del envío 58945688 (T50, modelo del producto)

**Envío del producto:** ref. 58945688, público **0.49702** (26 nov). Es la media simple de 10 miembros, cada uno con la HR recortada a 0–100 %. Desde T49, `app/predict.predecir()` carga los miembros que lista `modelo.json`; la línea base de abajo queda como referencia.

| Archivo | Contenido |
|---|---|
| `modelo.json` | Manifiesto: envío, variante de red, configuraciones, normalización (12 números) y los 10 miembros con su SHA-256 |
| `ensamble/r1_s0.pt` … `r1_s4.pt` | Configuración r1 (ronda 1: residuo sobre el GFS crudo + decodificador profundo), semillas 0–4 |
| `ensamble/r2_s0.pt` … `r2_s4.pt` | Configuración r2: igual + dropout 0.1 (no actúa al predecir: misma red) |
| `estadisticas_ensamble.npz` (**no va al repo**) | `mu_x`, `sd_x`, `clim`, `anom_sd` con **todas** las horas de `train.npz`: `python train/generar_estadisticas.py --ensamble` |

- **Tamaño:** 1,617,306 bytes por miembro (402,452 parámetros, `state_dict` float32), **16.2 MB** los 10.
- **Origen:** `train/t09b_capacidad.ipynb`, Kaggle `johancg04/vicuna-t09b-capacidad` v2 (Tesla T4, Python 3.13.16, NumPy 2.1.3, torch 2.11.0). Los SHA-256 coinciden con la salida descargada de Kaggle (26 nov).
- **Sin validación propia:** los 10 se reentrenaron con todos los datos (18 feb – 30 jun). La validación de esa configuración, con corte temporal (A, 16–30 jun), es **0.48626** (RMSE RH 9.190, T 0.836, U 1.555, V 1.475); ver `docs/modelo/experimentos.md`.
- **Reproducción en CPU** (Docker `python:3.12-slim`, torch 2.5.1+cpu): la inferencia del propio notebook con los 10 pesos y las estadísticas generadas por el script reproduce `submission_mezcla.csv` en las **646,800 filas**, con una diferencia máxima de 0.00054 (RH), 0.00061 (T), 0.00051 (U) y 0.00051 (V). El CSV va redondeado a 3 decimales. `tests/test_ensamble.py` repite la comparación con una muestra fija de 2,400 filas (24 horas).
- **Reproducción con `predecir()`** (26 nov, `3ab30e2`, Docker, torch 2.5.1+cpu): sobre las 646,800 filas, diferencia máxima 0.00053 (RH), 0.00063 (T), 0.00051 (U) y 0.00050 (V), dentro del límite de 0.002. La paridad entre el notebook y la app con los 10 pesos da 0 en 5 muestras (`tests/test_paridad.py`).
- **Tiempo de 24 horas en CPU con el código del notebook** (Intel i9-9880H, Docker, torch 2.5.1+cpu, mediana de 7): 4 hilos: 1 modelo 0.20 s, 5 modelos 1.12 s, **10 modelos 2.60 s**. Limitado a 2 CPU y 2 hilos: 0.34 s, 1.86 s y **4.19 s**.
- **Tiempo de 24 horas en CPU con `predecir()`**, 10 modelos (26 nov, `3ab30e2`, mismo equipo y versiones, mediana de 7): **2.11 s** con 4 hilos y **4.21 s** limitado a 2 CPU y 2 hilos, con otros contenedores en marcha (carga media 6–8). En otras corridas con la máquina cargada se midieron entre 7.8 y 22 s (PR #28). Falta medirlo en el servidor final (T47).
- **Entrada y salida:** igual que la línea base, pero la red suma su corrección al GFS de la hora t interpolado a 3 km: `pred = clim[h] + (red(x, h) + (GFS_interp − clim[h]) / anom_sd) × anom_sd`.

---

# Línea base (T03)

**Estos pesos no se han enviado a Kaggle.** El score público 0.42782 es de otra corrida del mismo starter (CPU, 24 nov), cuyos pesos no son estos.

## Archivos

| Archivo | Contenido | Tamaño (bytes) | SHA-256 |
|---|---|---|---|
| `modelo.pt` | `state_dict` de `Net` (11 tensores float32), sin optimizador | 579,761 | `ae4b86b29909dd72abc29a52575487fdfd1de574b5fc1f7fdb1ae40b4801bc1c` |
| `normalizacion.json` | `mu_x` (4), `sd_x` (4), `anom_sd` (4): los 12 números de normalización | menos de 1 kB | — |
| `estadisticas.npz` (**no va al repo**; se genera) | `mu_x` (4), `sd_x` (4), `clim` (24, 70, 35, 4), `anom_sd` (4), float32 | ≈ 942,000 | — |

Los pesos pesan unos 0.6 MB, muy por debajo de los 100 MB: no hace falta un enlace externo.

Los tensores se guardaron desde GPU: para cargarlos en CPU se usa `map_location="cpu"` (lo hace `app/predict.py`).

Para cargar la línea base, `app/predict.py` necesita además un manifiesto con la variante `starter`, `estadisticas.npz` y `data/static.npz` (relieve y grillas). Los dos `.npz` no van al repo.

### Cómo obtener `estadisticas.npz`

Con `data/train.npz` descargado de Kaggle (ver `data/README.md`) y NumPy, desde la raíz del repo:

```bash
python train/generar_estadisticas.py
```

El script calcula las cuatro claves con las horas de entrenamiento y solo escribe el archivo si `mu_x`, `sd_x` y `anom_sd` coinciden con `normalizacion.json` (diferencia relativa de hasta 1e-5).

`clim` es el promedio del WRF de entrenamiento por hora UTC y celda: un **derivado de la data oficial, de licencia CC BY-NC-SA**. Por eso el archivo no se publica en el repo (los organizadores no han confirmado que se pueda). Hasta el 25 nov estuvo versionado en la rama de T03, así que sigue visible en los commits del PR #11.

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

Predicción = `clim[hora] + anomalía × anom_sd`. Desde T10, `predecir()` recorta después la RH a 0–100 %.

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

### Con el recorte de RH (T10)

25 nov, script privado de Kaggle `johancg04/vicuna-t10-verificacion`, versión 1, en CPU (Python 3.13.18, NumPy 2.1.3, PyTorch 2.11.0, pandas 2.3.3): commit `fcda094` más los cambios de T10 (aún sin commit al ejecutarse). Mismos pesos y mismas estadísticas; solo cambia el recorte.

| Score | Sin recorte | Con recorte | Diferencia | RMSE RH sin → con |
|---|---|---|---|---|
| `score_local` | 0.337680 | 0.337793 | +1.13e-4 | 9.921 → 9.914 |
| `score_oficial` | 0.440924 | 0.440981 | +5.6e-5 | 9.905 → 9.899 |

- El WRF de `train.npz` tiene la RH entre 0 y 100 %. Sin recorte, 8,038 de 823,200 valores predichos en validación pasaban de 100 (máximo 116.8) y ninguno bajaba de 0.
- T, U y V no cambian (diferencia máxima 0.0).
- La tabla de validación de arriba es la del entrenamiento, sin recorte.

## Cómo comprobar que los archivos reproducen el score

Con `modelo.pt` en `models/`, `train.npz` y `static.npz` en `data/`, y PyTorch y pandas instalados:

```bash
pytest -q -s tests/test_reproduce_score.py tests/test_paridad.py
```

El primero genera las estadísticas en una carpeta temporal, carga los pesos con `predecir()` en CPU y exige una diferencia menor que 1e-4 en ambos scores respecto de los valores con recorte de RH (tabla de T10). La paridad con pesos reales no necesita las estadísticas.

Las dos comprobaciones siguientes se hicieron con el `estadisticas.npz` que escribió el entrenamiento (SHA-256 `5ba6ef49730e7923313ab857c9cfedd263aff198de7c05cd7c58cf71e8dcb0dd`), antes de que existiera el script.

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

### Con las estadísticas generadas por el script

25 nov, script privado de Kaggle `johancg04/vicuna-t03-generacion`, versión 1, en CPU: commit `66c2b28` más los cambios que retiran `estadisticas.npz` del repo (aún sin commit al ejecutarse). Python 3.12.3, torch 2.5.1+cpu, NumPy 2.1.3, pandas 2.2.3.

- Suite completa: 77 pruebas pasaron, ninguna omitida.
- `train/generar_estadisticas.py` produjo las cuatro claves con diferencia máxima 0.0 frente al archivo del entrenamiento, y el mismo SHA-256.
- `score_local` 0.33767968 (diferencia 0) y `score_oficial` 0.44092408 (-6.5e-8).
- Sin el archivo generado, `predecir()` falla con un mensaje que indica cómo generarlo.

## Limitaciones

- La época se elige mirando la validación, así que los scores son algo optimistas.
- Solo se entrenó con febrero–junio; el test (julio–agosto) es plena estación seca.
- No predice lluvia, nubosidad, radiación ni ráfagas.
- En el producto la entrada es el GFS de Open-Meteo, que puede diferir del GFS del concurso; el desempeño con esa fuente no está medido aquí.

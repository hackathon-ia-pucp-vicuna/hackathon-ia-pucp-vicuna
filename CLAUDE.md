# CLAUDE.md — Equipo Vicuña · Hackathon IA PUCP 2026

Instrucciones para Claude Code en este repositorio. Si las bases oficiales contradicen este archivo, prevalecen las bases.
Este repositorio es **público**: no escribir aquí secretos, claves ni datos del concurso.

## 1. Qué construimos
- **Modelo (Kaggle):** reducción de escala espacial. Entrada: pronóstico GFS en grilla 17×13 (~27 km), 4 variables en este orden: RH (%), T (K), U (m/s), V (m/s). Salida: lo que daría el modelo WRF en grilla 70×35 (3 km) para la hora t. Zona: Cordillera Blanca (Áncash). Entrada sugerida: horas t-2, t-1, t.
- **Producto: Helada Andina.** Aviso de helada por cultivo (empezamos con papa) para agricultores y técnicos del Callejón de Huaylas: riesgo de la noche, hora de inicio, duración, recomendación concreta y comparación "GFS crudo vs. nuestro modelo". Especificación en `specs/001-producto/spec.md`.
- **Meta interna de entrega: sábado 5 dic.** Fecha límite oficial: martes 8 dic 23:59 (Lima). Los días 6 y 7 son solo colchón.

## 2. Datos y métrica
- Archivos en `data/` (no versionados): `train.npz` (X 2935×17×13×4, Y 2935×70×35×4, 18 feb–30 jun 2024), `test.npz` (X 792×17×13×4, 2 jul–4 ago 2024), `static.npz` (DEM 70×35, coordenadas), `sample_submission.csv` (646,800 filas).
- Orientación: en X (GFS) la fila 0 es el **norte**; en Y y DEM la fila 0 es el **sur**; columna 0 = oeste. Tiempos en UTC (Lima = UTC−5). Hay huecos de horas: revisar `times`.
- Validación **siempre con corte temporal** (últimas dos semanas de junio). Nunca split aleatorio.
- `score = 1 − (RMSE_RH/25.360 + RMSE_T/2.0652 + RMSE_U/2.6776 + RMSE_V/2.0876)/4`. Envío: CSV `id,RH,T,U,V`, `id = YYYYMMDDHH_fila_col`, horas 00,03,…,21 UTC.
- Cupo de Kaggle: **10 envíos por día** (`maxDailySubmissions: 10` en la API de Kaggle y contador "0 / 10 used" de la pantalla de envío, verificados el 26 nov). Tope interno: 2 por día, y solo Johan sube, con aprobación. Se eligen 2 submissions finales.
- Línea base: 0.42782 en el leaderboard público (starter, 25 épocas).

## 3. Reglas que descalifican (nunca romperlas)
1. En Kaggle solo data oficial (entrenar y predecir). Se permiten modelos preentrenados públicos y variables derivadas de la data oficial.
2. `app/gfs_forecast.py` (Open-Meteo) es **solo del producto**; jamás para entrenar ni para el submission.
3. El producto usa **el mismo modelo y los mismos pesos** que Kaggle.
4. Nombre de equipo idéntico en Luma y Kaggle: **Vicuña** (con ñ). Una cuenta de Kaggle por persona.
5. No compartir código ni predicciones con otros equipos.
6. La data es CC BY-NC-SA: no se redistribuye y **no va al repo**.
7. Cuatro entregables: submissions en Kaggle, repo público, producto (link o README para correr en local) y video de máx. 5 min.

## 4. Estructura del repositorio
```
CLAUDE.md                  este archivo
README.md                  cómo reproducir y cómo ejecutar el producto
REPORTE_PRODUCTO.md        6 preguntas del reporte (con números)
LICENSE                    MIT (por crear)
requirements.txt           entrenamiento (plantilla)
requirements-app.txt       producto, inferencia en CPU (por crear)
train/                     starter_notebook.ipynb · submission_notebook.ipynb (reproducible, por crear)
models/                    pesos (.pt) + estadísticas de normalización + README (ficha del modelo)
app/
  gfs_forecast.py          plantilla: descarga GFS de Open-Meteo con caché (no tocar su interfaz)
  predict.py               predecir(): único punto de entrada al modelo (por crear)
  frost/                   rules.py · advice.py · zones.yaml · thresholds.yaml (por crear)
  web/streamlit_app.py     interfaz (por crear)
  notify/telegram.py       envío de alertas a un canal (por crear)
  storage.py               SQLite: confirmaciones de heladas (por crear)
specs/001-producto/        spec.md · plan.md · tasks.md
tests/                     pytest (por crear)
docs/                      licencias.md · modelo/experimentos.md
data/                      vacío en git (solo README)
```

## 4.1 Arquitectura
```
Open-Meteo (GFS) → gfs_forecast.py (caché) → ventanas t-2,t-1,t → predict.predecir() → T,HR,U,V a 3 km
                                                                           ↓
                              frost/rules.py (riesgo, hora de inicio, duración por cultivo) ← zones.yaml + DEM
                                                                           ↓
                           advice.py (mensaje) → Streamlit (web) · Telegram (canal) · SQLite (confirmaciones)
```
- **Aislamiento:** el código de entrenamiento no importa nada de `app/`. `app/` usa el modelo **solo** por `predecir()`.
- **Paridad:** el notebook de entrenamiento es autocontenido (define la red en línea) y `app/predict.py` tiene una copia de la red. `tests/test_paridad.py` carga los pesos guardados y comprueba que ambas dan la misma salida sobre una muestra fija.
- **Hora local:** la noche de helada en Lima (19:00–07:00) corresponde a 00, 03, 06, 09 y 12 UTC.
- **Reglas por cultivo:** `thresholds.yaml` exige el campo `fuente` (SENAMHI/INIA u otra) y `validado: true|false`. Sin fuente, solo se muestra la referencia general de 0 °C, rotulada como tal.
- **Honestidad del modelo:** no predice lluvia, nubosidad, radiación ni ráfagas. La helada se estima con T, HR y viento; el mensaje declara la incertidumbre.

## 5. Tecnologías (todas open source y de costo cero)
- Python 3.12 (Kaggle/Colab usan su propia versión; **la Mac del equipo con Python 3.14 e Intel no instala PyTorch**: entrenar en Kaggle/Colab).
- Modelo: PyTorch (CPU en inferencia), NumPy, pandas, matplotlib.
- Producto: Streamlit (interfaz), folium + OpenStreetMap (mapa), SQLite, PyYAML, requests. FastAPI solo si una spec lo justifica.
- Canal de aviso: Telegram (canal público con `python-telegram-bot` o llamadas HTTP). WhatsApp no es open source: solo opcional.
- Pruebas y calidad: pytest, ruff.
- Despliegue gratuito: Streamlit Community Cloud o Hugging Face Spaces, y siempre instrucciones para correr en local.
- Regla para dependencias: no instalar nada que no se necesite ahora; anotar licencia en `docs/licencias.md`; versiones fijas en el `requirements`.

## 6. Flujo de trabajo
- Plan y tablero: GitHub Projects "Vicuña · Hackathon IA PUCP"; las tareas T01…T50 están en `specs/001-producto/tasks.md` y las historias de usuario (HU) en `specs/001-producto/historias-usuario.md`.
- Producto: spec → plan → tareas → PRs pequeños. Una etapa por mensaje, con aprobación humana.
- Modelo: hipótesis → cambio mínimo → métrica local (corte temporal) → registrar en `docs/modelo/experimentos.md` → solo entonces un envío a Kaggle. No ajustar al leaderboard público.
- Cada tarea enlaza una HU y tiene criterios de aceptación. **Ninguna tarea pasa a Done si no cumple todos sus criterios**: el PR los lista marcados y los verifica una persona distinta a quien hizo la tarea.
- Una tarea, una rama (`feat/…`, `fix/…`, `docs/…`), un PR, commits en Conventional Commits, merge con squash.
- No push directo a `main`, no `--force`, no merge sin aprobación explícita del humano.
- No commitear `.env`, claves, `kaggle.json`, datos oficiales ni pesos mayores de 100 MB (si pasan, enlace en el README).

## 7. Comandos
```bash
# Kaggle (el CLI se usa como módulo)
python3 -m kaggle competitions submit -c hackathon-ia-pucp-2026 -f submission.csv -m "descripción"
python3 -m kaggle competitions submissions -c hackathon-ia-pucp-2026
# Datos (si faltan): python3 -m kaggle competitions download -c hackathon-ia-pucp-2026 -p data && unzip -o data/*.zip -d data
# Producto, cuando exista:  streamlit run app/web/streamlit_app.py
# Pruebas, cuando existan:  pytest -q
```

## 8. Qué modelo de Claude usar (desarrollo)
- Opus 5.5: spec, plan, rediseño de arquitectura, depuración difícil, revisión antes de merge y revisión final (en sesión nueva).
- Sonnet 5.5 (por defecto): implementación, notebooks, app, tests, README, reporte.
- Haiku 5.5: tareas mecánicas. Fable 5.2: solo si Opus no resolvió; anotar por qué.
- Empezar en Sonnet; subir a Opus tras dos fallos.

## 9. Antes de dar algo por terminado
- Correr lo que se cambió (tests o ejecución) y mirar la salida; no asumir.
- Cambios de modelo: reportar validación temporal por variable, no solo el score total.
- Cambios de producto: probar el flujo completo con datos reales de Open-Meteo (con caché) y con el modelo guardado.
- Revisar el diff: sin secretos, sin datos del concurso, sin archivos grandes.
- Verificar uno por uno los criterios de aceptación de la tarea (`tasks.md`). Si falta uno, la tarea sigue en In progress; no se mueve a Done ni se hace merge.
- Si un criterio no se puede cumplir, no se omite en silencio: se abre una discusión en el PR y se cambia el criterio con aprobación del equipo.

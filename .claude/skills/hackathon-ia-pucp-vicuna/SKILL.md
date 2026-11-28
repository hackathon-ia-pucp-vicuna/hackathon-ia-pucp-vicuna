---
name: "hackathon-ia-pucp-vicuna"
description: "Usar al trabajar en la Hackathon IA PUCP 2026 (equipo Vicuña): reglas de descalificación, flujo modelo/producto, modelo de Claude por tarea y checklist de entrega del 8 dic."
---

# Hackathon IA PUCP 2026 — equipo Vicuña

Skill de trabajo para el repo del concurso. Si el repo tiene su propio `CLAUDE.md`, ese manda en caso de conflicto. Si las bases oficiales y otro documento se contradicen, prevalecen las bases.

## Tarea del modelo (Kaggle)
- Entrada: pronóstico GFS en grilla 17×13 (~27 km), 4 variables en este orden: RH (%), T (K), U (m/s), V (m/s). Salida: lo que calcularía el modelo físico WRF en una grilla 70×35 a 3 km, para la hora t. Zona: Cordillera Blanca (Áncash). Relieve (DEM) disponible en `static.npz`. Kaggle sugiere usar las horas t-2, t-1 y t del GFS como entrada.
- Archivos: `train.npz` (X: 2935×17×13×4, Y: 2935×70×35×4, 18 feb–30 jun 2024), `test.npz` (792 horas, 2 jul–4 ago 2024), `static.npz`, `sample_submission.csv`.
- Orientación: en X (GFS) la fila 0 es el norte; en Y y en el DEM la fila 0 es el sur. Tiempos en UTC (Perú = UTC−5). La data tiene huecos de horas y días: revisar `times`.
- Validar siempre con corte temporal (por ejemplo, las últimas dos semanas de junio), nunca con split aleatorio.
- Se evalúan solo las horas 00, 03, …, 21 UTC (646,800 filas). Envío: CSV `id,RH,T,U,V` con `id = YYYYMMDDHH_fila_columna`.
- Métrica: `score = 1 − (RMSE_RH/25.360 + RMSE_T/2.0652 + RMSE_U/2.6776 + RMSE_V/2.0876)/4`. 0 = climatología; ~0.40 = baseline del starter; las 4 variables pesan igual.
- Leaderboard público ~30 %, privado ~70 % (el que cuenta). Se eligen 2 submissions finales. Límite diario confirmado: 5 envíos/día. Baseline del equipo: starter CNN, público 0.42782 (validación interna 0.3377), enviado como Vicuña.

## Contexto del concurso
- Cierre martes 8 dic 23:59 (hora de Lima), sin extensiones. Finalistas 10 dic. Final presencial sáb 12 dic, 2–5 p. m., FACI.
- Rúbrica de preselección (100): 35 Kaggle (proporcional al mejor privado), 20 valor para el usuario, 15 uso del modelo de punta a punta, 10 originalidad, 10 reproducibilidad, 10 video.
- Filtro previo: 4 entregables a tiempo, score privado mayor que 0, el producto usa tu modelo, repo y video accesibles.
- Equipo: Vicuña (idéntico en Luma y Kaggle, con la "ñ").
- Meta interna de entrega: sábado 5 dic (2–3 días antes del cierre). Vie 4 dic: video, prueba en máquina limpia y última submission. 6–7 dic: colchón solo para errores graves.
- Producto elegido: **Helada Andina** (alertas de helada para agricultores y técnicos agrarios del Callejón de Huaylas; papa primero). Salida: nivel de riesgo, hora de inicio, duración, recomendación y comparación "GFS crudo vs nuestro modelo". Ventana nocturna 19:00–07:00 Lima (00, 03, 06, 09, 12 UTC). Umbrales en `thresholds.yaml` con `fuente` y `validado`; sin fuente, solo referencia de 0 °C.
- Especificación y tareas: `specs/001-producto/` (`spec.md`, `plan.md`, `tasks.md`, T01–T51). Tablero: GitHub Project "Vicuña · Hackathon IA PUCP" (Kanban To do / In progress / Done, 3 sprints).

## Reglas que descalifican (nunca romperlas)
1. En Kaggle solo se usa la data oficial, para entrenar y predecir. Se permiten modelos preentrenados públicos y variables derivadas de la data oficial. Ningún otro dataset.
2. `app/gfs_forecast.py` es solo para el producto; nunca para entrenar ni para el submission.
3. El núcleo del producto es el mismo modelo de Kaggle (mismos pesos), usado tal cual, con el GFS de Open-Meteo como entrada.
4. El nombre del equipo en Kaggle es exactamente el de Luma. Una cuenta de Kaggle por persona.
5. No se comparte código ni predicciones con otros equipos.
6. La data es CC BY-NC-SA: no se redistribuye y no va al repo.
7. Los 4 entregables son obligatorios: submissions en Kaggle, repo público de GitHub, producto (link o README para correr en local) y video de máx. 5 min.

## Reglas de trabajo
- No push directo a `main` ni `--force`. No merge sin aprobación explícita del humano.
- No commitear `.env`, claves, `kaggle.json` ni datos privados. Si aparece un secreto en un diff, parar y avisar.
- `models/`: solo pesos (sin estado del optimizador). Si superan 100 MB, subirlos a Google Drive o Hugging Face con acceso por link y poner el link en el README.
- Costo cero (Colab, notebooks de Kaggle, servicios gratuitos). Registrar la licencia de cada modelo o dependencia en `docs/licencias.md`. Gestor: pip + `requirements.txt` con versiones.
- Una tarea, una rama, un PR, un commit (Conventional Commits). Merge con squash.
- Dependencias: no instalar nada que no se necesite ahora; preguntar antes de añadir una.

## Repo (plantilla oficial `D4rkPatata/Hackathon-IA-PUCP-2026`)
El equipo lo crea desde la plantilla, público, en la cuenta de un integrante, con los demás como colaboradores.
- `app/` (`gfs_forecast.py`, `ejemplo_producto.py`), `train/` (`starter_notebook.ipynb`, y al final `submission_notebook.ipynb` reproducible, con semillas fijas, rutas relativas y salidas guardadas), `data/` (vacío), `models/`, `README.md` propio, `REPORTE_PRODUCTO.md`, `requirements.txt` (con `torch`).
- `REPORTE_PRODUCTO.md` responde 6 preguntas, de forma concreta y con números: modelo (con tabla: arquitectura, tamaño de pesos, score público, validación local, tiempo de inferencia para 24 horas), para quién y qué problema, propuesta de valor, diferencia, cómo usa la salida del modelo, limitaciones y próximos pasos.
- Open-Meteo: una descarga completa cuesta ~221 llamadas (plan gratis: 5,000 por hora). Caché obligatoria; refrescar cada 1 a 6 horas.

## Arquitectura y estructura (Helada Andina)
- Flujo: Open-Meteo GFS → `app/gfs_forecast.py` (caché) → ventanas t-2,t-1,t → `app/predict.predecir()` → `app/frost/rules.py` (+ `zones.yaml` + DEM) → `advice.py` → Streamlit · canal de Telegram · confirmaciones en SQLite.
- Carpetas: `train/`, `models/`, `app/{gfs_forecast.py, predict.py, frost/, web/streamlit_app.py, notify/telegram.py, storage.py}`, `specs/`, `tests/`, `docs/`, `data/` (vacío).
- `predict.py` lleva una copia de la red; `tests/test_paridad.py` verifica que coincide con el notebook. El entrenamiento nunca importa `app/`.
- Stack 100 % open source y costo cero: Python 3.12, PyTorch (inferencia en CPU), NumPy/pandas/matplotlib, Streamlit, folium + OpenStreetMap, SQLite, PyYAML, requests, pytest, ruff. Telegram en lugar de WhatsApp (WhatsApp no es open source; solo opcional). Hosting: Streamlit Community Cloud o Hugging Face Spaces, más README para correr en local.
- Licencia MIT para el repo; la data (CC BY-NC-SA) nunca va al repo. Open-Meteo solo para el producto, nunca para entrenar.
- Entorno: en el Mac Intel con Python 3.14 no se instala PyTorch; entrenar en Kaggle/Colab. La GPU de Kaggle requiere verificar el teléfono.

## Elegir módulos y librerías
Orden: (1) lo que trae la plantilla; (2) biblioteca estándar; (3) lo ya instalado en Colab/Kaggle; (4) librería nueva solo si se necesita ahora, es gratuita y open source, funciona en Colab y en local, y el humano la aprueba.
- Un solo framework de deep learning: PyTorch (el de la plantilla). `numpy` para los `.npz`.
- El código de entrenamiento no importa nada de `app/`. `app/` usa el modelo por un punto de entrada único. Inferencia en CPU con los pesos finales. Seeds fijas.
- Producto: lo más simple que cumpla la spec (por defecto Streamlit o Gradio). Extras (LLM, quechua, chatbot, alertas) solo si la spec los justifica, con plan B si el servicio externo falla y sin claves en el repo.

## Historias de usuario y criterios de aceptación (obligatorios)
- Las HU están en `specs/001-producto/historias-usuario.md` (HU-01 a HU-11) y las tareas T01–T51 en `specs/001-producto/tasks.md`. Cada tarea enlaza una HU y trae sus criterios de aceptación, que también están en la tarjeta del tablero de GitHub Projects.
- **Ninguna tarea pasa a Done si no cumple TODOS sus criterios de aceptación.** Antes de mover una tarjeta a Done o de pedir merge: listar los criterios, comprobar cada uno con evidencia (prueba, salida, archivo, enlace) y marcarlos en el PR.
- La verificación final la hace una persona distinta a quien hizo la tarea. Un criterio nunca se omite en silencio: si no se puede cumplir, se discute en el PR y se cambia con aprobación del equipo.
- Al crear una tarea o HU nueva: redactar primero los criterios (Dado/Cuando/Entonces o condición medible), luego implementar.
- Responsables: johancg04 (modelo), lalo-dev8 (reglas, zonas y datos), DuDu-0912 (app, avisos y despliegue), maxrodev (repo, docs, QA, video y entrega). Tablero: Kanban To do / In progress / Done, 3 sprints, un tablero por sprint.

## Flujo
- Producto: spec (con usuario concreto y decisión que habilita) → HU con criterios → plan → tareas → implementar en PRs pequeños → verificar criterios → cierre. Una etapa por mensaje, con aprobación humana.
- Modelo: hipótesis → cambio mínimo → métrica local con corte temporal → registrar en `docs/modelo/experimentos.md` → solo entonces un envío a Kaggle. No ajustar al leaderboard público.
- Prioridad por peso: primero un envío válido (score > 0) y un producto mínimo de punta a punta; después mejorar.

## Modelo de Claude por tarea (desarrollo, no el modelo del concurso)
- Opus 5.5: spec, plan, rediseño de la arquitectura, depuración difícil, revisión antes de cada merge y revisión final de entrega (en sesión nueva).
- Sonnet 5.5 (por defecto): implementación diaria, notebooks, app, tests, README, `REPORTE_PRODUCTO.md`, borrador del guion del pitch.
- Haiku 5.5: tareas mecánicas (commits, formato, resúmenes, explorar el repo).
- Fable 5.2: solo si Opus no resolvió el problema; anotar por qué se subió.
- Empezar en Sonnet; subir a Opus si una tarea falla dos veces.

## Checklist de cierre (meta interna: sáb 5 dic; límite oficial: 8 dic 23:59)
- [ ] Todas las tareas T01–T51 están en Done con sus criterios de aceptación cumplidos y verificados.
- [ ] Predicciones enviadas a Kaggle como Vicuña y 2 submissions finales elegidas.
- [ ] Repo público con `submission_notebook.ipynb`, pesos (o link), `app/`, `requirements.txt`, README y `REPORTE_PRODUCTO.md`.
- [ ] Producto desplegado o README que lo ejecute en pocos pasos, probado desde cero.
- [ ] Video de máx. 5 min.
- [ ] Score privado mayor que 0.
- [ ] Sin secretos, `kaggle.json` ni data oficial en el historial de git.
- [ ] El producto usa el mismo modelo que Kaggle y datos actuales de GFS con caché.
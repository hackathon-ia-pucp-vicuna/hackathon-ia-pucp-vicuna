# Reglas del concurso y cómo las cumple el repo (T45 · HU-09)

Fuentes: **Bases de la Hackathon IA PUCP 2026, versión 1.3 (24 nov 2026)** y la presentación **"Hackathon IA PUCP — El reto"** (kickoff). Si algo de este archivo contradice las bases, prevalecen las bases. Revisado el 26 nov 2026 sobre `develop` (`e6a8e`).

Estado: ✅ se cumple hoy · ⚠️ se cumple en parte o depende de una tarea pendiente · 🔲 se comprueba al entregar.

## 1. Reglas que descalifican o anulan la participación

| # | Regla (bases) | Estado | Evidencia en el repo |
|---|---|---|---|
| 7d-1 | **En Kaggle, solo la data oficial**, para entrenar y para predecir. Prohibido cualquier otro dataset. | ✅ | Ningún archivo de `train/` usa Open-Meteo, `requests` ni `app/` (búsqueda en `.py` e `.ipynb`). Los notebooks cargan solo `train.npz`, `test.npz` y `static.npz` con `find()`. Regla escrita en `train/README.md`. Open-Meteo se usó solo para *comparar* entradas, sin guardar datos (`docs/datos/open-meteo-vs-gfs.md`, T13). |
| 7d-2 | La submission se debe poder **reproducir usando solo esa data**. | ⚠️ | `tests/test_reproduce_score.py` reproduce el score de la línea base. Falta el `submission_notebook.ipynb` del envío final (T29) y la prueba que reproduce ese envío (T50). |
| 7d-3 | **El núcleo del producto es el mismo modelo con el que se compitió en Kaggle**, alimentado con el GFS de Open-Meteo. "No vale un producto sin tu modelo ni un modelo decorativo." | ⚠️ **Riesgo abierto** | La app usa el modelo solo por `predecir()` (`app/predict.py`) y la entrada es Open-Meteo (`app/gfs_forecast.py`, `app/entrada_gfs.py`). La paridad app ↔ notebook pasa con pesos reales (`tests/test_paridad.py`). **Pero `models/modelo.pt` es la línea base de T03, que no es ningún envío a Kaggle.** Se resuelve con T49 (ensamble en `predecir()`, 27 nov) y T50 (publicar los pesos del envío del producto, 28 nov). T25 exige que una de las 2 finales sea la del producto. |
| 7d-4 | Open-Meteo es entrada del producto, **no data para reentrenar**. | ✅ | El código de entrenamiento no importa nada de `app/` (`tests/test_validacion.py` lo comprueba en el notebook). `app/gfs_forecast.py` lo declara en su cabecera. |
| 7c | **No compartir código ni predicciones entre equipos.** | ✅ | Regla en `CLAUDE.md` y en la skill del proyecto. Ver la duda 2 sobre el repo público. |
| 7g | En Kaggle, el equipo usa **exactamente el mismo nombre** del formulario de registro; si no, sus submissions no cuentan. | 🔲 | Nombre del equipo: **Vicuña** (con ñ). Se comprueba a mano en la pestaña *Team* de Kaggle contra el correo de confirmación del registro (T46). |
| 05 | **Los 4 entregables son obligatorios** al cierre del formulario (mar 8 dic, 23:59). Sin los 4, el equipo no pasa a la preselección. No se aceptan entregas tardías por ningún medio. | 🔲 | A) submissions en Kaggle; B) repo público; C) producto (link o README para correr en local); D) video de máx. 5 min. Checklist en T32; meta interna sáb 5 dic. |
| Filtro | Filtro previo de la preselección: 4 entregables a tiempo, **score privado mayor que 0**, el producto usa tu modelo, repo y video accesibles. | ⚠️ | Score público actual 0.49486 (envío 58926424, `docs/modelo/experimentos.md`). "El producto usa tu modelo": ver 7d-3. Repo: público ✅. Video: T31. |
| 03 | Equipos fijos de 2 a 4; cada integrante con cuenta propia de Kaggle y GitHub; inscripción en Luma y registro del equipo hasta el **lun 7 dic, 23:59**. | ✅ | 4 integrantes (`specs/001-producto/tasks.md`, tabla de responsables). Registro: T46. |

## 2. Lo que debe tener el repositorio (admisibilidad)

| Requisito | Estado | Dónde |
|---|---|---|
| Repo público de GitHub, creado desde la plantilla oficial | ✅ | `github.com/hackathon-ia-pucp-vicuna/hackathon-ia-pucp-vicuna` (PUBLIC) |
| Notebook reproducible del entrenamiento | ⚠️ | Hoy: `train/t02_linea_base.ipynb` … `t11_ventanas.ipynb`. Falta `submission_notebook.ipynb` (T29). |
| Pesos del modelo | ⚠️ | `models/modelo.pt` (línea base). Los del envío final: T50/T24. Límite de GitHub 100 MB; si pasan, enlace en el README. |
| Reporte del producto | ✅ (borrador) | `REPORTE_PRODUCTO.md` (T27). |
| Si el producto corre en local, el README permite instalarlo y ejecutarlo en pocos pasos; la organización podrá probarlo | 🔲 | `README.md` + `requirements-app.txt`. Prueba en máquina limpia: T30 (vie 4 dic). |
| El producto funciona de punta a punta con **datos actuales**; mostrar la salida del modelo sobre datos de prueba no es producto | ✅ | App v0 con Open-Meteo en vivo (T44). Probado de punta a punta el 26 nov con los pesos de la línea base. |
| Los finalistas pueden tener que reproducir su resultado desde el repo (7e) | ⚠️ | Igual que 7d-2. |

## 3. Datos, envíos y licencias

| Tema | Regla o dato | Evidencia |
|---|---|---|
| Data oficial fuera del repo | Licencia CC BY-NC-SA según la página de datos de Kaggle (no figura en las bases): no se redistribuye. | `.gitignore` bloquea `data/*`, `*.npz` y `*.csv`. `git ls-files` no tiene ningún `.npz`, `.csv` ni `.nc`. `models/estadisticas.npz` (incluye `clim`, derivado de la data) no se publica; se genera con `train/generar_estadisticas.py`. |
| Cupo de envíos | 5 por día (pantalla de envío de Kaggle; no figura en las bases). Se eligen 2 submissions finales. | `CLAUDE.md` §2. Envíos registrados en `docs/modelo/experimentos.md` antes de enviar. |
| Leaderboard | Público ~30 % / privado ~70 %; cuenta el privado (35 de los 100 puntos de la preselección, proporcional al mejor score privado). | No ajustar al leaderboard público (`CLAUDE.md` §6). Validación con corte temporal (`train/validacion.py`). |
| Secretos | Sin claves en el repo. | `.gitignore` bloquea `.env`, `.env.*` y `kaggle.json`. |
| Herramientas de IA | Permitido cualquier modelo de IA público y cualquier herramienta de desarrollo asistida por IA (7a, 7b). | — |

## 4. Fechas que no se mueven (hora de Lima)

| Fecha | Qué |
|---|---|
| lun 7 dic, 23:59 | Cierre de inscripciones y registro de equipos |
| **mar 8 dic, 23:59** | **Cierre de Kaggle y del formulario de entregables** |
| jue 10 dic | Anuncio de finalistas (correo) |
| sáb 12 dic, 2:00–5:00 p. m. | Final presencial, auditorio de FACI. Basta con que asista un integrante. |

Si el producto está desplegado, su disponibilidad es responsabilidad del equipo: no hay extensiones por fallas de plataformas externas.

## 5. Dudas para los organizadores

Las dudas de data o reglas van al grupo de participantes (no a los mentores). Anotar aquí la fecha de envío y la respuesta.

| # | Duda | Enviada | Respuesta |
|---|---|---|---|
| 1 | ¿Podemos publicar en el repo `models/estadisticas.npz`? Incluye `clim` (promedio del WRF por hora y celda), derivado de la data CC BY-NC-SA. Sin él, `predecir()` necesita que quien pruebe la app descargue la data de Kaggle. | pendiente | |
| 2 | La regla 7c prohíbe compartir código entre equipos, y el repo debe ser público al entregar. ¿Debe estar privado hasta el 8 dic, o se puede tener público desde ya? | pendiente | |
| 3 | Para la regla 7d: si el producto usa un ensamble de 5 semillas, ¿basta con que sea el mismo ensamble de una de las 2 submissions finales? | pendiente | |

## 6. Confirmación de lectura de las bases

Cada integrante confirma en el PR de T45 (o en un comentario) que leyó las bases completas, versión 1.3.

- [ ] @johancg04
- [ ] @lalo-dev8
- [ ] @DuDu-0912
- [ ] @maxrodev

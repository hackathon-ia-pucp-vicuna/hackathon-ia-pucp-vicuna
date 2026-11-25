# Tareas — Helada Andina + modelo

Fuente de verdad del plan. El tablero de GitHub Projects "Vicuña · Hackathon IA PUCP" usa los mismos identificadores.
Cada tarea enlaza una historia de usuario (HU, ver `historias-usuario.md`) y tiene criterios de aceptación.
**Ninguna tarea pasa a Done si no cumple todos sus criterios.** El PR marca cada criterio y los verifica una persona distinta a quien hizo la tarea.
Fechas en hora de Lima. Meta interna de entrega: sáb 5 dic (límite oficial: mar 8 dic 23:59).

## Sprint 1 · Base y núcleo (24–28 nov)

### T01 · Modelo — Activar GPU en Kaggle (verificar teléfono)
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] El notebook imprime GPU disponible (`torch.cuda.is_available()` es True).
- [ ] El tiempo por época en GPU queda anotado en `docs/modelo/experimentos.md`.

### T02 · Modelo — Validación temporal con la métrica oficial
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Valida con las últimas 2 semanas de junio, nunca con split aleatorio.
- [ ] Reproduce el score de la línea base (0.3377 ± 0.005) y el RMSE por variable.
- [ ] La función de validación se puede reutilizar desde el notebook.

### T03 · Modelo — Guardar pesos y estadísticas en models/ con ficha
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] `models/` tiene pesos (solo state_dict), estadísticas de normalización y `FICHA.md`.
- [ ] Cargarlos en un proceso nuevo reproduce el score de validación (diferencia < 1e-4).
- [ ] Si pesan más de 100 MB, hay enlace de descarga en el README.

### T04 · Producto — Aprobar spec 001 (Helada Andina)
HU: HU-01 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Los 4 integrantes aprueban en el PR o en un comentario.
- [ ] Las HU y criterios de `historias-usuario.md` están en `develop` (pasan a `main` al cerrar el sprint).
- [ ] No quedan preguntas abiertas en la spec.

### T05 · Equipo — Contactar usuario real y reservar mentorías
HU: HU-03 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Hay una fecha acordada con al menos 1 agricultor o técnico agrario.
- [ ] Hay mentorías reservadas (producto y modelo) con fecha y hora.
- [ ] No se guardan datos personales del contacto en el repo.

### T06 · Repo — Commit de CLAUDE.md, LICENSE MIT y estructura
HU: HU-08 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Existen LICENSE (MIT), CLAUDE.md, `requirements-app.txt` y las carpetas con README.
- [ ] `.gitignore` bloquea `data/`, `.env` y `kaggle.json`.
- [ ] `pip install -r requirements-app.txt` funciona en un entorno virtual nuevo.
- [ ] PR aprobado por otra persona.

### T07 · Equipo — Fuente de umbrales de helada para papa
HU: HU-01 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] `thresholds.yaml` tiene `fuente` (documento, institución, año y enlace) y `validado`.
- [ ] Otra persona abrió la fuente y confirmó que dice lo que el YAML afirma.
- [ ] Sin fuente verificada, la app solo muestra 0 °C como referencia general.

### T08 · Modelo — Experimento: pérdida ponderada por la métrica
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Fila en `experimentos.md`: hipótesis, cambio, score de validación por variable y decisión.
- [ ] Semilla fija; el resultado se reproduce.
- [ ] La decisión no se basa en el leaderboard público.

### T09 · Modelo — Experimento: predecir residuo sobre GFS
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Fila en `experimentos.md` con hipótesis, validación por variable y decisión.
- [ ] Comparado contra la mejor línea previa, con la misma validación.

### T10 · Modelo — Experimento: épocas, aprendizaje y recorte de HR
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Fila en `experimentos.md` con validación por variable y decisión.
- [ ] HR queda recortada a 0–100 y hay una prueba que lo verifica.

### T11 · Datos — Revisar ventanas con huecos de horas
HU: HU-09 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Ninguna ventana (t-2, t-1, t) cruza un hueco de horas.
- [ ] Hay una prueba con huecos sintéticos.
- [ ] Se reporta cuántas ventanas había antes y cuántas quedan.

### T12 · Producto — predict.py con predecir() y test de paridad
HU: HU-01 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] `predecir(ventana)` devuelve un arreglo 70×35×4 en unidades físicas.
- [ ] `tests/test_paridad.py` pasa: iguala la red del notebook (diferencia < 1e-5 en 5 muestras).
- [ ] Corre en CPU y reporta el tiempo para 24 horas.

### T13 · Datos — Comparar GFS Open-Meteo vs GFS del concurso
HU: HU-04 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Existe `docs/datos/open-meteo-vs-gfs.md` con media, desviación y diferencia por variable.
- [ ] Concluye si hace falta corregir sesgos o conversiones.
- [ ] Declara con honestidad lo que no se pudo comparar.

### T14 · Producto — zones.yaml y extracción de punto con el DEM
HU: HU-07 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Al menos 3 localidades del Callejón de Huaylas con latitud, longitud, celda (fila, columna) y elevación del DEM.
- [ ] Hay una prueba de orientación (fila 0 norte en GFS, fila 0 sur en Y y DEM).
- [ ] Una zona fuera de la grilla da un error claro.

### T15 · Producto — frost/rules.py: riesgo, inicio y duración
HU: HU-02 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Clasifica la noche en sin riesgo, vigilancia o helada probable.
- [ ] Calcula hora de inicio (hora de Lima) y duración bajo el umbral.
- [ ] Al menos 8 pruebas, con borde en 0 °C, noche sin cruce y varios cruces.
- [ ] Los umbrales se leen del YAML.

### T16 · Producto — thresholds.yaml (papa) y advice.py en español
HU: HU-03 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Hay un mensaje por nivel, de 3 oraciones como máximo, con recomendación e incertidumbre.
- [ ] Declara que el modelo no predice lluvia, nubosidad, radiación ni ráfagas.
- [ ] Una persona distinta revisó el español y una prueba verifica que no quedan campos sin rellenar.

### T22 · Producto — Prueba de despliegue gratuito con PyTorch en CPU
HU: HU-11 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Una app mínima con PyTorch en CPU corre en hosting gratuito y su URL abre.
- [ ] Se miden memoria y tiempo de arranque.
- [ ] La decisión (Streamlit Cloud u otra) queda escrita en `plan.md`.

### T35 · Producto — Adaptador Open-Meteo → entrada del modelo
HU: HU-01 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Devuelve ventanas 17×13×4 por hora con unidades y orden de variables del entrenamiento (RH, T en K, U, V; fila 0 norte).
- [ ] Usa caché con TTL y maneja horas faltantes.
- [ ] Hay una prueba con datos guardados (fixture) sin red.

### T44 · Producto — App v0 de punta a punta (zona → Open-Meteo → predecir() → reglas → mensaje)
HU: HU-02 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Abre en local desde un entorno virtual nuevo con `pip install -r requirements-app.txt` y `streamlit run`.
- [ ] Con una zona real del Callejón de Huaylas muestra nivel de riesgo, hora de inicio y duración, con un mensaje fijo en español.
- [ ] Usa `predecir()` con los pesos de la línea base y `thresholds.yaml`; si falla la red, muestra un mensaje claro en español.
- [ ] Hay una captura de la ejecución o la URL desplegada en el PR.

### T45 · Equipo — Leer reglas oficiales y checklist de descalificación
HU: HU-09 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Los 4 integrantes leyeron las bases completas y lo confirmaron en el PR o en un comentario.
- [ ] `docs/reglas.md` lista cada regla de descalificación con la evidencia de cómo la cumple el repo (datos permitidos, límite de envíos, licencias).
- [ ] Cualquier duda queda enviada a los organizadores, con fecha.

### T46 · Equipo — Registro del equipo y formulario de entrega
HU: HU-10 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Los 4 integrantes figuran en el equipo oficial y en Kaggle, y cada uno puede enviar submissions.
- [ ] El formulario de entrega y su fecha límite (hora de Lima) están anotados en `plan.md`.
- [ ] Hay una captura de la confirmación de inscripción.

## Sprint 2 · Producto mínimo y modelo final (29 nov–3 dic)

### T17 · Producto — Streamlit: zona, tabla por hora, GFS vs modelo
HU: HU-02 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Se elige zona y cultivo y se ve nivel, hora de inicio y duración.
- [ ] Se ven la tabla por hora y T mínima del GFS crudo y del modelo.
- [ ] Con caché, carga en menos de 5 segundos.
- [ ] Los errores salen en español, sin trazas técnicas.

### T18 · Producto — Aviso por canal de Telegram (script de envío)
HU: HU-05 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] El script envía el aviso al canal; el mensaje tiene como máximo 600 caracteres.
- [ ] No duplica el aviso de la misma zona y noche.
- [ ] El token se lee de variables de entorno; hay modo de prueba sin envío.

### T19 · Producto — Confirmaciones sí/no en SQLite
HU: HU-06 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] El formulario guarda zona, fecha y respuesta en SQLite.
- [ ] Una segunda respuesta de la misma zona y fecha no duplica el registro.
- [ ] No se guardan datos personales.

### T20 · Modelo — Arquitectura U-Net o residual con DEM y hora
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Fila en `experimentos.md` con validación por variable.
- [ ] Se conserva solo si mejora la validación; si no, se documenta por qué se descarta.
- [ ] Se anotan tamaño de pesos y tiempo de inferencia.

### T21 · Modelo — Ensamble de semillas
HU: HU-09 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Al menos 3 semillas; el score del ensamble se compara con una sola semilla.
- [ ] El tiempo de inferencia para 24 horas en CPU sigue siendo aceptable y está anotado.
- [ ] El tamaño total de pesos está dentro del límite del repo o hay enlace.

### T23 · Equipo — (Opcional) Mensaje en quechua revisado por hablante
HU: HU-03 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Los textos los revisó una persona hablante, con su visto bueno.
- [ ] Se muestran en la app como opción.
- [ ] Si no hay revisor, no se publica.

### T36 · Datos — Evaluar T mínima nocturna: modelo vs GFS crudo
HU: HU-04 · Responsable: @lalo-dev8
Criterios de aceptación:
- [ ] Tabla con RMSE y MAE de la T mínima nocturna del modelo y del GFS interpolado en la validación.
- [ ] Indica el número de noches evaluadas y si hubo noches con T ≤ 0 °C.
- [ ] La conclusión es honesta: si el modelo no mejora, lo dice.

### T37 · Producto — Modo demo y manejo de fallos de red
HU: HU-11 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Si falla Open-Meteo, se muestra el último pronóstico en caché con su hora y una advertencia, o un modo DEMO rotulado.
- [ ] Si falla Telegram, la app sigue funcionando y registra el error.
- [ ] Hay pruebas que simulan ambos fallos.

### T38 · Producto — Crear bot y canal de Telegram; token fuera del repo
HU: HU-05 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] El bot y el canal existen y al menos 2 integrantes están suscritos.
- [ ] El token está en `.env` (ignorado) y hay un `.env.example` sin secretos.
- [ ] Una búsqueda de secretos no encuentra el token en el historial de git.

### T40 · Repo — CI en GitHub Actions y docs/licencias.md
HU: HU-08 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Cada PR corre pytest y ruff; main está en verde.
- [ ] `docs/licencias.md` lista cada dependencia y modelo con su licencia.

### T48 · Producto — Envío programado del aviso con GitHub Actions
HU: HU-05 · Responsable: @maxrodev
Depende de: T18 (script de envío) y T38 (bot y canal).
Criterios de aceptación:
- [ ] Un workflow de GitHub Actions ejecuta el script de T18 todos los días a las 17:00 de Lima (22:00 UTC).
- [ ] El token se lee de los secretos de GitHub; no aparece en el repo ni en los logs del workflow.
- [ ] Si el workflow falla, el README explica cómo correr el script a mano.

### T47 · Modelo — Congelar el modelo final (corte de experimentos)
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] El 2 dic queda decidido qué pesos se usan y se anota en `experimentos.md`.
- [ ] Después de esa fecha no cambian los pesos salvo por un error grave, con issue y aprobación del equipo.
- [ ] `predecir()` y la app cargan los pesos congelados y el score de validación coincide.

## Sprint 3 · Entrega (meta 5 dic; límite 8 dic)

### T24 · Modelo — Entrenamiento final y pesos en models/
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Entrenado con todos los datos oficiales y con semillas fijas.
- [ ] Los pesos finales están en `models/` con su ficha.
- [ ] El `submission.csv` pasa la validación de formato (ids de `sample_submission.csv`, sin NaN).

### T25 · Modelo — Elegir las 2 submissions finales
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Hay 2 submissions marcadas en Kaggle: una segura y una más arriesgada.
- [ ] El criterio queda en `experimentos.md` y no depende solo del leaderboard público.

### T26 · Producto — Pulido de interfaz, errores y caché
HU: HU-11 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Sin red o sin datos, la app muestra un mensaje claro en español.
- [ ] La caché de Open-Meteo tiene TTL de 1 a 6 horas y no excede el límite de llamadas.
- [ ] Se revisaron textos, ortografía y contraste.

### T27 · Docs — README y REPORTE_PRODUCTO.md
HU: HU-08 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] `REPORTE_PRODUCTO.md` responde las 6 preguntas con números y la tabla del modelo completa.
- [ ] Alguien que no escribió el README lo siguió y funcionó.
- [ ] Todos los enlaces abren.

### T28 · Producto — Prueba con usuario real y feedback
HU: HU-03 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Al menos 1 persona real leyó el mensaje y se anotó si lo entendió.
- [ ] `docs/producto/feedback.md` tiene lo observado y al menos 1 cambio aplicado o justificado.

### T29 · Docs — submission_notebook.ipynb reproducible
HU: HU-08 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Corre de punta a punta en una sesión nueva de Kaggle o Colab.
- [ ] Produce el mismo `submission.csv` (diferencia 0 o hash igual).
- [ ] Semillas fijas, rutas relativas y salidas guardadas.

### T30 · QA — Prueba en máquina limpia (vie 4 dic)
HU: HU-08 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] En una máquina sin entorno previo se siguió el README: clonar, instalar, ejecutar app y notebook.
- [ ] Los fallos encontrados se registraron y se corrigieron.

### T31 · Video — Guion y grabación (máx. 5 min, vie 4 dic)
HU: HU-10 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Dura 5:00 o menos.
- [ ] Muestra problema, usuario, uso real del producto con la salida del modelo, y limitaciones.
- [ ] El audio se entiende y el enlace abre sin iniciar sesión.

### T32 · Entrega — Checklist de los 4 entregables y envío
HU: HU-10 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Submissions, repo público, producto y video se abren desde una ventana de incógnito.
- [ ] Los enlaces están en el formulario de entrega y hay captura de la confirmación.

### T33 · Entrega — Última submission a Kaggle (vie 4 dic)
HU: HU-10 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] La submission respeta el límite diario y es válida (score mayor que 0).
- [ ] Las 2 finales están seleccionadas.

### T34 · Entrega — Colchón: solo errores graves
HU: HU-10 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Cada cambio tiene issue y prueba.
- [ ] No se agrega nada nuevo.
- [ ] Se repite el checklist de T32.

### T39 · Modelo — Ficha del modelo: tamaño, tiempos y tabla del reporte
HU: HU-09 · Responsable: @johancg04
Criterios de aceptación:
- [ ] La tabla del reporte tiene arquitectura, parámetros, tamaño de pesos, score público, validación local y tiempo de inferencia de 24 horas en CPU.
- [ ] Un script (`tools/medir_inferencia.py`) reproduce las cifras.

### T41 · Producto — Despliegue final gratuito del producto
HU: HU-10 · Responsable: @DuDu-0912
Criterios de aceptación:
- [ ] Hay una URL pública que usa los pesos finales a través de `predecir()`.
- [ ] El README explica cómo correrlo en local si el hosting falla.
- [ ] Se documenta el tiempo de arranque en frío.

### T42 · QA — Auditoría del repo: secretos, datos y enlaces
HU: HU-10 · Responsable: @maxrodev
Criterios de aceptación:
- [ ] Una búsqueda de secretos (historial incluido) no halla claves, `kaggle.json` ni archivos de la data oficial.
- [ ] El repo es público y sus enlaces abren en incógnito.

### T43 · Entrega — Pitch para la final (diapositivas, guion y ensayo)
HU: HU-10 · Responsable: @johancg04
Criterios de aceptación:
- [ ] Diapositivas y guion de 5 minutos cronometrado, ensayado ante al menos 1 persona.
- [ ] Hay respuestas preparadas sobre límites del modelo, score en Kaggle y datos usados.
- [ ] Listo antes del 10 dic (anuncio de finalistas).

## Resumen por responsable

| Responsable | Tareas | N.º |
|---|---|---|
| @johancg04 | T01, T02, T03, T08, T09, T10, T20, T24, T25, T39, T43, T47 | 12 |
| @maxrodev | T04, T06, T27, T29, T30, T31, T32, T33, T34, T40, T42, T46, T48 | 13 |
| @DuDu-0912 | T05, T17, T18, T19, T22, T23, T26, T28, T37, T38, T41, T44 | 12 |
| @lalo-dev8 | T07, T11, T12, T13, T14, T15, T16, T21, T35, T36, T45 | 11 |

Total: 48 tareas. Si alguien se sobrecarga, reasignar en el tablero y actualizar esta tabla.

# Historias de usuario (HU) y criterios de aceptación — Helada Andina

Regla de oro: **ninguna tarea pasa a Done si no cumple TODOS sus criterios de aceptación.** El PR enlaza la HU y marca cada criterio; los verifica una persona distinta a quien hizo la tarea. Los criterios de cada tarea están en `tasks.md` y en la tarjeta del tablero.

Formato de criterios: Dado / Cuando / Entonces, o una condición verificable (número, archivo, prueba).

## Usuarios
- **Agricultor**: decide cada tarde si protege su cultivo esta noche.
- **Técnico agrario**: asesora a varios agricultores; quiere ver de dónde sale el aviso.
- **Jurado / evaluador**: revisa que el producto use el modelo y que todo se pueda reproducir.
- **Equipo**: entrena, valida y entrega a tiempo.

## HU del producto

**HU-01 · Ver el riesgo de helada de esta noche** — Como agricultor, quiero ver si esta noche habrá helada en mi zona, para decidir si protejo mi cultivo.
- Dado una zona y el cultivo papa, cuando abro la app, entonces veo uno de tres niveles: sin riesgo, vigilancia o helada probable.
- Dado que no hay umbral con fuente para el cultivo, entonces la app solo muestra la referencia de 0 °C y lo rotula como referencia general.
- La clasificación usa la salida de `predecir()` y no valores inventados.

**HU-02 · Saber cuándo empieza y cuánto dura** — Como agricultor, quiero la hora estimada de inicio y la duración bajo el umbral, para saber cuándo actuar.
- La hora se muestra en hora de Lima (UTC−5), no en UTC.
- Si la temperatura no baja del umbral, la app dice "no se espera" y no inventa hora.

**HU-03 · Entender el mensaje y qué hacer** — Como agricultor, quiero un mensaje corto en español claro con una recomendación, para actuar sin interpretar números.
- El mensaje tiene como máximo 3 oraciones, incluye la recomendación y una nota de incertidumbre.
- Declara que el modelo no predice lluvia, nubosidad, radiación ni ráfagas.
- Al menos una persona real (agricultor o técnico) lo leyó y dijo si lo entiende.

**HU-04 · Comparar GFS crudo contra nuestro modelo** — Como técnico agrario, quiero ver la temperatura mínima del GFS crudo y la del modelo para la misma noche, para juzgar cuánto aporta el modelo.
- Se muestran siempre ambas temperaturas mínimas y la diferencia.
- Hay una tabla por hora y un gráfico de T de la noche con las dos curvas.

**HU-05 · Recibir el aviso por Telegram** — Como agricultor, quiero recibir el aviso en un canal de Telegram la tarde previa, para no tener que abrir la app.
- El mensaje sale una vez por noche y por zona (sin duplicados).
- El token del bot nunca está en el repo; se lee de variables de entorno.

**HU-06 · Confirmar si hubo helada** — Como agricultor, quiero confirmar con sí/no al día siguiente si hubo helada, para mejorar los umbrales.
- La confirmación se guarda con zona y fecha en SQLite, sin datos personales.
- Una segunda confirmación de la misma zona y fecha reemplaza o rechaza la anterior, no la duplica.

**HU-07 · Elegir mi zona y cultivo** — Como agricultor, quiero elegir mi localidad y cultivo, para ver el aviso de mi zona.
- Hay al menos 3 localidades del Callejón de Huaylas, cada una mapeada a su celda de la grilla y a su elevación del DEM.
- Una zona fuera de la grilla muestra un error claro, no un fallo.

**HU-08 · Ejecutarlo desde cero** — Como jurado, quiero clonar el repo y ejecutar producto y notebook siguiendo el README, para comprobar la reproducibilidad.
- En una máquina limpia, el README lleva del clon a la app funcionando sin pasos ocultos.
- El notebook de entrega genera el mismo `submission.csv` con semillas fijas y rutas relativas.

**HU-09 · Mejorar el modelo sin hacer trampa** — Como equipo, queremos mejorar el score en Kaggle con validación temporal y solo con la data oficial, para sumar los 35 puntos sin riesgo de descalificación.
- Cada cambio queda en `docs/modelo/experimentos.md` con hipótesis, validación local por variable y decisión.
- No se ajusta al leaderboard público; las 2 submissions finales se eligen con criterio documentado.
- No se usa ningún dato fuera del oficial para entrenar ni predecir.

**HU-10 · Entregar a tiempo los 4 entregables** — Como equipo, queremos enviar submissions, repo público, producto y video antes del 5 dic, para tener colchón hasta el 8 dic.
- Cada enlace se abre en una ventana de incógnito sin iniciar sesión.
- El video dura como máximo 5 minutos y muestra el producto usando el modelo.

**HU-11 · Funcionar aunque falle la red** — Como agricultor, quiero que la app no se rompa si falla Open-Meteo o Telegram, para no quedarme sin información.
- Si falla Open-Meteo, la app muestra el último pronóstico en caché con su hora y una advertencia, o un modo DEMO rotulado.
- Los errores se muestran en español sin trazas técnicas.

## Matriz HU → tareas
| HU | Tareas |
|---|---|
| HU-01 | T07, T12, T15, T35 |
| HU-02 | T15, T17, T44 |
| HU-03 | T05, T16, T28 |
| HU-04 | T13, T17, T36 |
| HU-05 | T18, T38 |
| HU-06 | T19 |
| HU-07 | T14, T17 |
| HU-08 | T03, T06, T27, T29, T30, T40, T41 |
| HU-09 | T01–T03, T08–T11, T20, T21, T24, T25, T39, T45, T47 |
| HU-10 | T27, T31, T32, T33, T42, T43, T46 |
| HU-11 | T22, T26, T37 |

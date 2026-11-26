# Plan 001 — Helada Andina

## Arquitectura
```
Open-Meteo (GFS) → app/gfs_forecast.py (caché) → ventanas → app/predict.predecir() → T,HR,U,V (70×35)
  → app/frost/rules.py (zona→celda con zones.yaml y DEM; riesgo, inicio, duración con thresholds.yaml)
  → app/frost/advice.py (mensaje) → Streamlit (web) · Telegram (canal) · SQLite (confirmaciones)
```

## Decisiones
| Tema | Decisión | Motivo |
|---|---|---|
| Interfaz | Streamlit | Rápido de armar, buen video, open source |
| Canal de aviso | Canal público de Telegram, enviado por script | Evita lógica de suscripciones; WhatsApp no es open source |
| Confirmaciones | Formulario en la web + SQLite | Cierra el ciclo sin bot interactivo |
| Modelo en producto | PyTorch en CPU, copia de la red en `app/predict.py` + test de paridad | El notebook de Kaggle debe ser autocontenido |
| Despliegue | Streamlit Community Cloud o Hugging Face Spaces, más README local | Gratuito; el hosting no es open source pero el código corre en local |
| API | Solo si hace falta (FastAPI) | Menos piezas en 12 días |
| Caché | `gfs_cache.npz`, refresco de 1 a 6 h | ~221 llamadas por descarga completa; gratis hasta 5,000 por hora |

## Riesgos y verificaciones tempranas
1. **GFS de Open-Meteo vs. GFS del concurso:** pueden diferir; comparar distribuciones antes de confiar (T13).
2. **Inferencia fuera de Kaggle:** la Mac del equipo no instala PyTorch (Python 3.14, Intel). Probar el despliegue Linux CPU (T22).
3. **Umbrales sin fuente:** no inventarlos; documentar la fuente o rotular como referencia (T07).
4. **Cupo de envíos a Kaggle:** 5 por día; enviar solo lo que mejore la validación.
5. **Cambio de estación:** entrenamiento febrero–junio, test julio–agosto (más seco y frío).

## Hitos
| Hito | Fecha | Resultado |
|---|---|---|
| H0 Base | mié 25 nov | GPU, validación temporal, pesos guardados, spec aprobada, usuario real contactado |
| H1 Núcleo | sáb 28 nov | `predecir()` con test de paridad, reglas de helada, primeras mejoras del modelo |
| H2 Mínimo funcional | mar 1 dic | Alcance mínimo de la spec corriendo con el modelo baseline; modelo v1 enviado |
| H3 Finales | jue 3 dic | Modelo final, 2 submissions elegidas, producto pulido, borrador de README y reporte |
| H4 Entrega | sáb 5 dic | 4 entregables confirmados (video grabado el 4 dic) |
| Colchón | 6–7 dic | Solo errores graves |

## Despliegue gratuito con PyTorch en CPU (T22 · HU-11)
Prueba con `app/web/streamlit_app.py`: red del tamaño del starter (`Net`, pesos aleatorios, semilla 0) y entrada sintética de 24 horas, sin data oficial ni Open-Meteo. Dependencias en `app/web/requirements.txt` (torch 2.5.1+cpu, streamlit 1.40.2, numpy 2.1.3).

**Criterios de aceptación**
- [x] Una app mínima con PyTorch en CPU corre en hosting gratuito y su URL abre. → URL: https://hackathon-ia-pucp-vicunagit-hdtewbxemwnted2smdeq5m.streamlit.app/ (acceso público sin iniciar sesión, HTTP 200, 25/11/2026)
- [x] Se miden memoria y tiempo de arranque. → Tabla de mediciones.
- [x] La decisión (Streamlit Cloud u otra) queda escrita en `plan.md`. → Esta sección.

**Configuración:** Streamlit Community Cloud · rama `feat/t22-despliegue-cpu` · archivo `app/web/streamlit_app.py` · Python 3.12.16 · Linux x86_64 · torch 2.5.1+cpu (CUDA no disponible) · 16 hilos de CPU.

**Mediciones** (valores que muestra la app, 25/11/2026)
| Medida | Primera carga (proceso nuevo) | Recarga con F5 (mismo proceso) |
|---|---|---|
| Arranque del proceso → modelo listo | 10.4 s | — (no se repite: modelo en caché) |
| Importar torch | 2.47 s | — |
| Crear la red | 0.002 s | — |
| Inferencia 24 h (1.ª / 2.ª) | 0.118 s / 0.039 s | — |
| Carga total (import + red + 2 inferencias) | 2.63 s | — |
| Memoria RSS actual | 287 MB | 350 MB |
| Memoria RSS máxima | 358 MB | 358 MB |
| Parámetros y tamaño de pesos | 143,956 · 0.55 MB (float32) | — |

Lectura: con F5 Streamlit reutiliza el proceso y la caché (`st.cache_resource`), así que los tiempos no se vuelven a medir y la recarga es casi inmediata. De los 10.4 s de arranque, ~2.6 s son PyTorch y el modelo; el resto es el arranque de Python y Streamlit. Un arranque tras dormir la app (o *Reboot app*) incluye además reconstruir el contenedor, que no mide la app.

**Decisión:** usamos **Streamlit Community Cloud** para el producto. PyTorch CPU se instala y corre sin problemas; el pico de memoria (358 MB) deja margen amplio para el modelo final, `predecir()` y la caché del GFS; la inferencia de 24 horas tarda ~0.1 s. Riesgo: la app se duerme si no tiene visitas y el primer acceso tarda más; antes del video y de la evaluación hay que abrirla para despertarla. Plan B: Hugging Face Spaces (CPU gratis) y, en todo caso, README para correr en local.

## Inscripción y entrega (T46 · HU-10)
**Equipo:** Vicuña (con "ñ"), idéntico en Luma y en Kaggle. Los 4 integrantes figuran en ambos (verificado el 26/11/2026): @johancg04, @lalo-dev8, @DuDu-0912 y @maxrodev. Una cuenta de Kaggle por persona.

**Formulario de entregables:** los organizadores **aún no lo publican** (al 26/11/2026 el equipo no lo ha recibido). Según el anuncio oficial de fechas, cierra junto con Kaggle. Cuando llegue (correo o grupo oficial de WhatsApp), se anota aquí su enlace o dónde encontrarlo. Seguimiento: @maxrodev (T32).
- Se entregan los 4 entregables: submissions en Kaggle, repositorio público, producto (link o README para correr en local) y video pitch.
- **Cierre: martes 8 dic, 23:59 (hora de Lima, UTC−5)**, a la misma hora que Kaggle. Sin extensiones.
- Meta interna: sábado 5 dic. El 4 dic se graba el video, se prueba en máquina limpia y se hace la última submission; el 6 y 7 dic son solo para errores graves (T34).
- El envío y la captura de la confirmación son parte de T32.

**Fechas oficiales (hora de Lima)**
| Hito | Fecha |
|---|---|
| Cierre de inscripciones | lun 7 dic, 23:59 |
| Cierre de Kaggle y del formulario de entregables | mar 8 dic, 23:59 |
| Anuncio de finalistas (por correo) | jue 10 dic |
| Gran final presencial (PUCP) | sáb 12 dic, 2:00–5:00 p. m. |

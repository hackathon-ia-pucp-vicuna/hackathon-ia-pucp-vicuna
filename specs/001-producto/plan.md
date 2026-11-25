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
- [ ] Una app mínima con PyTorch en CPU corre en hosting gratuito y su URL abre. → URL: _pendiente_
- [ ] Se miden memoria y tiempo de arranque. → Tabla de mediciones.
- [ ] La decisión (Streamlit Cloud u otra) queda escrita en `plan.md`. → Esta sección.

**Configuración:** Streamlit Community Cloud · rama `feat/t22-despliegue-cpu` · archivo `app/web/streamlit_app.py` · Python 3.12.

**Mediciones** (valores que muestra la app; fecha de medición: _pendiente_)
| Medida | Arranque en frío | Recarga (en caliente) |
|---|---|---|
| Tiempo de build (log del hosting) | _pendiente_ | — |
| Arranque del proceso → modelo listo | _pendiente_ | _pendiente_ |
| Importar torch | _pendiente_ | — |
| Inferencia 24 h (1.ª / 2.ª) | _pendiente_ | — |
| Memoria RSS actual / máxima | _pendiente_ | _pendiente_ |
| Parámetros y tamaño de pesos | _pendiente_ | — |

**Decisión:** _pendiente de las mediciones._ Si Streamlit Community Cloud no alcanza (memoria o build), plan B: Hugging Face Spaces (CPU gratis) y, en todo caso, README para correr en local.

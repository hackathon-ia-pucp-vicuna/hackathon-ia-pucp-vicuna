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

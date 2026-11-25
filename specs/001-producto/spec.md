# Spec 001 — Helada Andina

Estado: borrador para aprobación del equipo. Los umbrales y mensajes de ejemplo son ilustrativos y deben validarse con fuentes técnicas y con usuarios reales.

## Usuario y problema
- **Usuario:** agricultores y técnicos agrarios del Callejón de Huaylas (Áncash). Cultivo inicial: papa.
- **Problema:** las heladas ocurren a escala de parcela y el pronóstico global (GFS, ~27 km) no las distingue; la pérdida llega de un día para otro.
- **Decisión que habilita:** si proteger el cultivo esta noche y con qué medida (riego ligero, humo, cobertura).

## Flujo
1. La persona elige su zona y su cultivo (o consulta una zona predefinida).
2. El sistema obtiene el pronóstico GFS de Open-Meteo (con caché) y arma ventanas t-2, t-1, t.
3. El modelo predice T, HR, U, V a 3 km.
4. Para la zona se toma la celda correspondiente y, de la noche (19:00–07:00 Lima = 00–12 UTC), la mínima, el viento y la humedad.
5. Se clasifica: sin riesgo, vigilancia o helada probable; se estima hora de inicio y duración bajo el umbral.
6. Se genera la recomendación según el cultivo y se muestra la comparación GFS crudo vs. modelo.
7. El aviso sale por la web y por un canal de Telegram la tarde previa.
8. La persona confirma al día siguiente si hubo helada; se guarda en SQLite para ajustar umbrales.

## Requisitos funcionales (alcance mínimo, listo el 1 dic)
- RF1: una zona y un cultivo (papa) con clasificación de riesgo por noche.
- RF2: hora estimada de inicio y duración bajo el umbral.
- RF3: tabla por hora y gráfico de T de la noche, con la comparación GFS vs. modelo.
- RF4: mensaje corto en español claro, con recomendación y nota de incertidumbre.
- RF5: envío del aviso a un canal de Telegram.
- RF6: registro de confirmaciones (sí/no) en SQLite.

## Ampliaciones (solo si el alcance mínimo ya funciona)
Varios cultivos, mapa de riesgo por zona (folium + OpenStreetMap), versión en quechua escrita revisada por una persona hablante, panel con historial de heladas, suscripción interactiva por Telegram.

## Reglas de negocio
- La referencia general es T ≤ 0 °C. Los umbrales por cultivo viven en `app/frost/thresholds.yaml` con `fuente` y `validado`. Sin fuente verificada, solo se muestra la referencia general, rotulada como tal.
- El modelo no predice lluvia, nubosidad, radiación ni ráfagas: el mensaje lo declara.
- El producto usa el mismo modelo y los mismos pesos que Kaggle, solo a través de `predecir()`.

## Criterios de aceptación
- [ ] Con datos reales de Open-Meteo (caché activa) el flujo completo corre en menos de unos segundos por zona.
- [ ] Muestra siempre T mínima del modelo y T mínima de GFS crudo para la misma noche.
- [ ] Una persona real (agricultor o técnico) revisó el mensaje y dijo si lo entiende.
- [ ] Se ejecuta desde cero siguiendo el README, en máquina limpia.

## Métricas de éxito
- Error de temperatura mínima del modelo frente a GFS crudo en la validación temporal (menor es mejor).
- Aciertos de noches con helada según las confirmaciones, si se logran recoger.
- Opinión de al menos un usuario real.

# app/frost/ — Reglas de helada

| Archivo | Qué hace | Tarea |
|---|---|---|
| `zones.yaml` | Localidades: latitud, longitud, celda (fila, columna) de la grilla 70×35 y elevación del DEM. | T14 |
| `thresholds.yaml` | Umbrales por cultivo, con `fuente` y `validado`. | T07, T16 |
| `rules.py` | Nivel de riesgo, hora de inicio (hora de Lima) y duración bajo el umbral. | T15 |
| `advice.py` | Mensaje en español: máximo 3 oraciones, con recomendación e incertidumbre. | T16 |

Reglas (ver `specs/001-producto/spec.md`):
- Niveles según la T mínima de la noche: más de 2 °C = sin riesgo; de 0 a 2 °C = vigilancia; 0 °C o menos = helada probable.
- Sin fuente verificada en `thresholds.yaml`, solo se muestra la referencia general de 0 °C, rotulada como tal.
- Noche: 19:00–07:00 en Lima = 00–12 UTC (todas las horas).
- El modelo no predice lluvia, nubosidad, radiación ni ráfagas: el mensaje lo declara.
- Orientación: en la grilla del modelo (70×35) y en el DEM, la fila 0 es el **sur**.

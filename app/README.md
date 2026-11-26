# app/ — Producto Helada Andina

Aviso de helada por cultivo (papa) para el Callejón de Huaylas. Usa **el mismo modelo y los mismos pesos** que la submission de Kaggle, solo a través de `predict.predecir()`.

```
gfs_forecast.py (Open-Meteo + caché) → predict.predecir() → frost/rules.py → frost/advice.py → web · notify · storage
```

| Ruta | Qué es | Tarea |
|---|---|---|
| `gfs_forecast.py` | Plantilla: descarga el GFS de Open-Meteo con caché. **No cambiar su interfaz.** Solo para el producto, nunca para entrenar ni para Kaggle. | — |
| `ejemplo_producto.py` | Ejemplo de la plantilla (modelo de relleno). | — |
| `predict.py` | `predecir(Xw, horas_utc)`: único punto de entrada al modelo. | T12 |
| `frost/` | Reglas de helada, mensajes, zonas y umbrales. | T07, T14–T16 |
| `web/` | Interfaz Streamlit. | T44, T17, T26 |
| `notify/` | Envío del aviso al canal de Telegram. | T18, T48 |
| `storage.py` | Confirmaciones sí/no en SQLite. | T19 |

Instalar y ejecutar (Python 3.12):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-app.txt
streamlit run app/web/streamlit_app.py
```

Los tokens (Telegram) van en variables de entorno o en `.env`, que está en `.gitignore`. Nunca en el código.

# tests/ — Pruebas (pytest)

```bash
pytest -q
```

- Las pruebas no usan red ni la data oficial: trabajan con datos sintéticos o fixtures pequeños.
- `test_paridad.py` (T12, T49): cada variante de red de `app/predict.py` (starter y residuo_crudo_profundo) da la misma salida que la del notebook (diferencia < 1e-5 en 5 muestras), con pesos sintéticos y con los reales.
- `test_predecir_ensamble.py` (T49): `predecir()` carga los miembros de `models/modelo.json`, da un error claro si falta uno, si cambia su SHA-256 o si las estadísticas no coinciden, y devuelve la media de los miembros con la HR recortada. Con la data oficial reproduce la muestra del envío 58945688.
- `test_telegram.py` (T18): aviso por Telegram sin red: 600 caracteres como máximo, sin duplicados por zona y noche, token desde variables de entorno, modo de prueba sin envío y errores sin el token.
- `test_streamlit_app.py` (T17): la app de Streamlit de punta a punta con AppTest, sin red (pronóstico, relieve y modelo sintéticos): zona y cultivo, nivel, inicio y duración, tabla por hora con GFS crudo y modelo, caché y errores en español sin trazas. Con la data oficial y los pesos mide la carga con caché (< 5 s).
- En CI (T40), cada PR corre `pytest` y `ruff`.

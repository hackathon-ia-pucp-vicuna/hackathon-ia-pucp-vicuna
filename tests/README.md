# tests/ — Pruebas (pytest)

```bash
pytest -q
```

- Las pruebas no usan red ni la data oficial: trabajan con datos sintéticos o fixtures pequeños.
- `test_paridad.py` (T12, T49): cada variante de red de `app/predict.py` (starter y residuo_crudo_profundo) da la misma salida que la del notebook (diferencia < 1e-5 en 5 muestras), con pesos sintéticos y con los reales.
- `test_predecir_ensamble.py` (T49): `predecir()` carga los miembros de `models/modelo.json`, da un error claro si falta uno, si cambia su SHA-256 o si las estadísticas no coinciden, y devuelve la media de los miembros con la HR recortada. Con la data oficial reproduce la muestra del envío del producto (mezcla de 15, público 0.49888).
- En CI (T40), cada PR corre `pytest` y `ruff`.

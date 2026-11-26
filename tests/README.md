# tests/ — Pruebas (pytest)

```bash
pytest -q
```

- Las pruebas no usan red ni la data oficial: trabajan con datos sintéticos o fixtures pequeños.
- `test_paridad.py` (T12): la red de `app/predict.py` da la misma salida que la del notebook (diferencia < 1e-5 en 5 muestras).
- En CI (T40), cada PR corre `pytest` y `ruff`.

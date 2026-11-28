# models/

Aquí van los **pesos de tu modelo final**: los mismos que generaron tu submission de Kaggle y que carga tu producto.

- Guarda solo los pesos, sin el estado del optimizador. Por ejemplo, en PyTorch: `torch.save(model.state_dict(), "models/modelo.pt")`.
- Si aun así pasan de **100 MB** (el límite de GitHub), súbelos a Google Drive o Hugging Face, con acceso para cualquiera con el link, y pon el link en el README principal y aquí.

Link a los pesos (solo si no están en esta carpeta): [pegar link]

En este repo: `modelo.pt` (línea base, T03) y `ensamble/` con `modelo.json` (los 30 miembros del producto, envío 59028408, 71.7 MB, T50). Detalle en `FICHA.md`.

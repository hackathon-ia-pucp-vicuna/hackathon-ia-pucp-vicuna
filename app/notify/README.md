# app/notify/ — Aviso por Telegram

- `telegram.py`: envía el aviso de la noche al canal público del equipo (T18). Máximo 600 caracteres, un aviso por zona y noche (sin duplicados), con modo de prueba sin envío.
- Envío programado: GitHub Actions, todos los días a las 17:00 de Lima (T48). Si falla, el script se corre a mano.
- El token del bot **nunca** va en el repo: se lee de variables de entorno (`.env` en local, secretos de GitHub en Actions). Ver `.env.example` (T38).
- Si Telegram falla, la app sigue funcionando y registra el error (T37).

# app/notify/ — Aviso por Telegram

- `telegram.py`: envía el aviso de la noche al canal público del equipo (T18). Un mensaje por zona, de 600 caracteres como máximo, sin duplicados (registro en `avisos_enviados.sqlite`, que no va al repo) y con modo de prueba sin envío.
- El aviso se calcula con `app/aviso.py`: el mismo `predecir()`, las mismas reglas y los mismos mensajes que la app.
- El token del bot **nunca** va en el repo: se lee de variables de entorno (`.env` en local, secretos de GitHub en Actions). Ver `.env.example` (T38).

```bash
python app/notify/telegram.py --prueba        # muestra los avisos de la próxima noche, no envía
export TELEGRAM_BOT_TOKEN=...                 # token de @BotFather
export TELEGRAM_CHAT_ID=@nombre_del_canal     # el bot debe ser administrador del canal
python app/notify/telegram.py                 # envía al canal; no repite zona y noche ya enviadas
```

- Envío programado: GitHub Actions, todos los días a las 17:00 de Lima (T48). Si falla, el script se corre a mano. En Actions el registro de enviados no persiste entre corridas: T48 debe guardarlo (caché o artefacto) o el envío diario corre una sola vez por noche.
- Si Telegram falla, la app sigue funcionando y registra el error (T37).

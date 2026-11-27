"""
Envía el aviso de helada de la noche al canal de Telegram (T18 · HU-05).

- Un mensaje por zona y noche, de 600 caracteres como máximo.
- No duplica: guarda en SQLite (app/notify/avisos_enviados.sqlite, en .gitignore) cada zona y noche
  enviada, y no vuelve a enviarla.
- El token y el canal se leen de variables de entorno (nunca van en el repo):
    TELEGRAM_BOT_TOKEN   token del bot (lo da @BotFather)
    TELEGRAM_CHAT_ID     canal, por ejemplo @helada_andina (el bot debe ser administrador del canal)
- Modo de prueba (--prueba): arma y muestra los mensajes, sin red de Telegram, sin token y sin
  registrar nada.

Uso, desde la raíz del repo:
    python app/notify/telegram.py --prueba                  # muestra los avisos, no envía
    python app/notify/telegram.py                           # envía al canal (necesita las variables)
    python app/notify/telegram.py --zonas carhuaz --noche 20261127
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

APP = Path(__file__).resolve().parent.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

MAX_CARACTERES = 600
RUTA_REGISTRO = Path(__file__).resolve().parent / "avisos_enviados.sqlite"
API = "https://api.telegram.org/bot{token}/sendMessage"
TIEMPO_ESPERA = 20
EMOJI = {"sin riesgo": "🟢", "vigilancia": "🟡", "helada probable": "🔴"}
PIE = "Solo estima temperatura, humedad y viento; no predice lluvia, nubosidad, radiación ni ráfagas."


class ConfiguracionFaltante(RuntimeError):
    """Faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID."""


class EnvioFallido(RuntimeError):
    """Telegram no aceptó el mensaje. El texto nunca incluye el token."""


def armar_mensaje(aviso):
    """Texto del aviso de una zona, de MAX_CARACTERES como máximo (se quitan partes opcionales si no cabe)."""
    r, u = aviso["resultado"], aviso["umbral"]
    ini = aviso["inicio_noche"]
    cabecera = f"{EMOJI.get(r['nivel'], '❄️')} Helada Andina · {aviso['zona']} · noche del {ini:%d/%m}"
    datos = f"Mínima: {r['tmin_c']:.0f} °C hacia las {r['hora_tmin']:%H:%M}"
    if r.get("inicio"):
        datos += f" · bajo {u['helada_c']:g} °C desde las {r['inicio']:%H:%M}, unas {r['duracion_h']:g} h"
    datos += f" ({aviso['cultivo'].lower()}, hora de Lima)."
    partes = [cabecera, f"{r['nivel'].capitalize()}. {aviso['mensaje']}", datos, PIE]
    for quitar in (None, 3, 2):                       # primero el pie, luego los datos
        if quitar is not None:
            partes = partes[:quitar]
        texto = "\n".join(partes)
        if len(texto) <= MAX_CARACTERES:
            return texto
    return texto[:MAX_CARACTERES - 1] + "…"


def _registro(ruta=None):
    con = sqlite3.connect(ruta or RUTA_REGISTRO)
    con.execute("CREATE TABLE IF NOT EXISTS enviados (zona TEXT, noche TEXT, enviado_utc TEXT, "
                "message_id INTEGER, PRIMARY KEY (zona, noche))")
    return con


def ya_enviado(con, zona, noche):
    return con.execute("SELECT 1 FROM enviados WHERE zona = ? AND noche = ?", (zona, noche)).fetchone() is not None


def marcar_enviado(con, zona, noche, message_id=None):
    con.execute("INSERT OR IGNORE INTO enviados VALUES (?, ?, ?, ?)",
                (zona, noche, datetime.now(timezone.utc).isoformat(timespec="seconds"), message_id))
    con.commit()


def configuracion():
    """(token, chat_id) desde las variables de entorno."""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(), os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    faltan = [n for n, v in (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat)) if not v]
    if faltan:
        raise ConfiguracionFaltante(f"Faltan las variables de entorno: {', '.join(faltan)}. "
                                    "Usa --prueba para ver los avisos sin enviarlos.")
    return token, chat


def enviar(texto, token, chat_id):
    """Envía `texto` al canal y devuelve el message_id. Los errores no muestran el token."""
    try:
        resp = requests.post(API.format(token=token), timeout=TIEMPO_ESPERA,
                             json={"chat_id": chat_id, "text": texto, "disable_web_page_preview": True})
        datos = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise EnvioFallido(f"No se pudo conectar con Telegram ({type(e).__name__}).") from None
    if not datos.get("ok"):
        detalle = str(datos.get("description", f"HTTP {resp.status_code}")).replace(token, "***")
        raise EnvioFallido(f"Telegram rechazó el mensaje: {detalle}")
    return datos["result"]["message_id"]


def avisar(avisos, prueba=False, ruta_registro=None, salida=print):
    """
    Envía (o muestra, en prueba) un mensaje por aviso, sin repetir zona y noche ya enviadas.
    Devuelve {"enviados": [...], "omitidos": [...], "fallidos": [...]} con las claves de zona.
    """
    res = {"enviados": [], "omitidos": [], "fallidos": []}
    textos = [(a, armar_mensaje(a)) for a in avisos]
    if prueba:
        for a, texto in textos:
            salida(f"--- {a['clave']} · noche {a['noche']} · {len(texto)} caracteres (prueba, no se envía)\n{texto}")
        return res
    token, chat = configuracion()
    con = _registro(ruta_registro)
    try:
        for a, texto in textos:
            if ya_enviado(con, a["clave"], a["noche"]):
                res["omitidos"].append(a["clave"])
                salida(f"{a['clave']} · noche {a['noche']}: ya se envió, no se repite.")
                continue
            try:
                mid = enviar(texto, token, chat)
            except EnvioFallido as e:
                res["fallidos"].append(a["clave"])
                salida(f"{a['clave']} · noche {a['noche']}: {e}")
                continue
            marcar_enviado(con, a["clave"], a["noche"], mid)
            res["enviados"].append(a["clave"])
            salida(f"{a['clave']} · noche {a['noche']}: enviado ({len(texto)} caracteres).")
    finally:
        con.close()
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="Envía el aviso de helada de la noche al canal de Telegram.")
    p.add_argument("--prueba", action="store_true", help="muestra los mensajes sin enviarlos")
    p.add_argument("--zonas", nargs="*", help="claves de zones.yaml (por defecto, todas)")
    p.add_argument("--noche", help="YYYYMMDD en UTC (por defecto, la próxima noche)")
    p.add_argument("--cultivo", default="papa")
    a = p.parse_args(argv)
    from aviso import calcular_avisos

    try:
        if not a.prueba:
            configuracion()                            # falla antes de calcular si no hay token
        res = avisar(calcular_avisos(a.noche, a.cultivo, a.zonas), prueba=a.prueba)
    except Exception as e:                             # sin trazas: el mensaje ya explica qué pasó
        print(f"No se envió el aviso: {e}", file=sys.stderr)
        return 1
    return 1 if res["fallidos"] else 0


if __name__ == "__main__":
    sys.exit(main())

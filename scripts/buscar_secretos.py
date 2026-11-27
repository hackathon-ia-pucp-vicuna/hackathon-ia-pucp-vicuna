"""
Busca secretos en todo el historial de git (T38 · HU-05). Solo biblioteca estándar.

Revisa el diff de todos los commits de todas las ramas buscando:
  - tokens de bots de Telegram (123456789:AA…, 35 caracteres después de los dos puntos);
  - claves de Kaggle ("key": "…32 hex…") y archivos kaggle.json o .env versionados;
  - asignaciones con valor de TELEGRAM_BOT_TOKEN, KAGGLE_KEY o KAGGLE_API_TOKEN.
Nunca imprime el secreto: solo el commit, el archivo y el tipo. Sale con código 1 si encuentra algo.

Uso, desde la raíz del repo:
    python scripts/buscar_secretos.py
"""
import re
import subprocess
import sys

PATRONES = {
    "token de Telegram": re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"),
    "clave de Kaggle": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "asignación de secreto": re.compile(r"\b(TELEGRAM_BOT_TOKEN|KAGGLE_KEY|KAGGLE_API_TOKEN)\s*=\s*['\"]?[A-Za-z0-9_:-]{20,}"),
}
ARCHIVOS_PROHIBIDOS = re.compile(r"(^|/)(kaggle\.json|\.env|access_token)$")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def buscar():
    hallazgos = []
    commits = git("rev-list", "--all").split()
    for c in commits:
        for ruta in git("show", "--name-only", "--format=", c).split("\n"):
            if ruta and ARCHIVOS_PROHIBIDOS.search(ruta):
                hallazgos.append((c[:7], ruta, "archivo de credenciales versionado"))
        archivo = None
        for linea in git("show", "--format=", "--unified=0", c).split("\n"):
            if linea.startswith("+++ "):
                archivo = linea[6:] if linea.startswith("+++ b/") else linea[4:]
            elif linea.startswith("+") and not linea.startswith("+++"):
                for tipo, patron in PATRONES.items():
                    if patron.search(linea):
                        hallazgos.append((c[:7], archivo, tipo))
    return len(commits), hallazgos


def main():
    n, hallazgos = buscar()
    for c, ruta, tipo in hallazgos:
        print(f"{c} · {ruta} · {tipo}")
    print(f"{n} commits revisados · {len(hallazgos)} posibles secretos")
    return 1 if hallazgos else 0


if __name__ == "__main__":
    sys.exit(main())

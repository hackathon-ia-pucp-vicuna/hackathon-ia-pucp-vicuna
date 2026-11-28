"""
Busca secretos en todo el historial de git (T38 · HU-05). Solo biblioteca estándar.

Revisa el diff de todos los commits de todas las ramas locales y remotas que tenga el clon, incluidos
los commits de merge (git log --all -p -m), buscando:
  - tokens de bots de Telegram (123456789:AA…, 35 caracteres después de los dos puntos), también
    pegados a "bot" como en la URL de la API (https://api.telegram.org/bot123456789:AA…/sendMessage);
  - claves de Kaggle: la antigua de kaggle.json ("key": "…32 hex…") y el token de API nuevo (KGAT_…);
  - asignaciones con valor de TELEGRAM_BOT_TOKEN, KAGGLE_KEY o KAGGLE_API_TOKEN;
  - archivos kaggle.json, .env o access_token versionados.
Nunca imprime el secreto: solo el commit, el archivo y el tipo. Sale con código 1 si encuentra algo.

Qué NO cubre: refs que el clon no tiene (refs/pull/* de GitHub, forks, ramas borradas en el remoto y
nunca descargadas), ni los objetos sueltos sin commit. Para incluir los PRs de GitHub:
    git fetch origin "+refs/pull/*/head:refs/remotes/origin/pr/*"

Uso, desde la raíz del repo:
    python scripts/buscar_secretos.py
"""
import re
import subprocess
import sys

PATRONES = {
    # (?<![0-9]) en vez de \b: "bot123456789:AA…" no tiene borde de palabra antes del número
    "token de Telegram": re.compile(r"(?<![0-9])\d{8,10}:[A-Za-z0-9_-]{35}\b"),
    "clave de Kaggle": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    # El SDK de Kaggle habla de tokens "KGAT" frente a la clave antigua; el patrón es amplio a propósito
    "token de API de Kaggle": re.compile(r"\bKGAT_[A-Za-z0-9]{16,}"),
    "asignación de secreto": re.compile(r"\b(TELEGRAM_BOT_TOKEN|KAGGLE_KEY|KAGGLE_API_TOKEN)\s*=\s*['\"]?[A-Za-z0-9_:-]{20,}"),
}
ARCHIVOS_PROHIBIDOS = re.compile(r"(^|/)(kaggle\.json|\.env|access_token)$")
MARCA = "@@commit "


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def buscar():
    """(commits revisados, hallazgos sin repetir [(commit, archivo, tipo)])."""
    hallazgos = {}
    commit = None
    for ruta in git("log", "--all", "-m", "--name-only", f"--format={MARCA}%h").split("\n"):
        if ruta.startswith(MARCA):
            commit = ruta[len(MARCA):]
        elif ruta and ARCHIVOS_PROHIBIDOS.search(ruta):
            hallazgos[(commit, ruta, "archivo de credenciales versionado")] = None
    archivo = None
    for linea in git("log", "--all", "-p", "-m", "--unified=0", "--no-color", "--no-ext-diff",
                     f"--format={MARCA}%h").split("\n"):
        if linea.startswith(MARCA):
            commit, archivo = linea[len(MARCA):], None
        elif linea.startswith("+++ "):
            archivo = linea[6:] if linea.startswith("+++ b/") else linea[4:]
        elif linea.startswith("+"):
            for tipo, patron in PATRONES.items():
                if patron.search(linea):
                    hallazgos[(commit, archivo, tipo)] = None
    n = len(git("rev-list", "--all").split())
    return n, list(hallazgos)


def main():
    n, hallazgos = buscar()
    for c, ruta, tipo in hallazgos:
        print(f"{c} · {ruta} · {tipo}")
    print(f"{n} commits revisados (incluidos los de merge) · {len(hallazgos)} posibles secretos")
    return 1 if hallazgos else 0


if __name__ == "__main__":
    sys.exit(main())

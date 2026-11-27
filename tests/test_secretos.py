"""
Token fuera del repo (T38 · HU-05).

- .env está en .gitignore y .env.example no trae valores secretos.
- scripts/buscar_secretos.py reconoce un token de Telegram y una clave de Kaggle, y en un repositorio
  de prueba encuentra el token aunque se haya borrado en un commit posterior (está en el historial).

Se ejecuta con:  pytest -q tests/test_secretos.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))

import buscar_secretos  # noqa: E402

TOKEN_FALSO = "123456789:" + "AAbbCC_dd-" * 3 + "EeFfG"          # 35 caracteres tras los dos puntos


def test_env_ignorado_y_env_example_sin_secretos():
    ignorados = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignorados and "!.env.example" in ignorados
    ejemplo = (RAIZ / ".env.example").read_text(encoding="utf-8")
    valores = dict(linea.split("=", 1) for linea in ejemplo.splitlines() if "=" in linea and not linea.startswith("#"))
    assert valores["TELEGRAM_BOT_TOKEN"] == ""
    assert valores["TELEGRAM_CHAT_ID"].startswith("@")
    assert not any(p.search(ejemplo) for p in buscar_secretos.PATRONES.values())


def test_los_patrones_reconocen_secretos():
    assert len(TOKEN_FALSO.split(":")[1]) == 35
    assert buscar_secretos.PATRONES["token de Telegram"].search(f"token = '{TOKEN_FALSO}'")
    assert buscar_secretos.PATRONES["clave de Kaggle"].search('{"username": "x", "key": "' + "a1" * 16 + '"}')
    assert buscar_secretos.PATRONES["asignación de secreto"].search(f"TELEGRAM_BOT_TOKEN={TOKEN_FALSO}")
    assert not buscar_secretos.PATRONES["asignación de secreto"].search("TELEGRAM_BOT_TOKEN=")
    assert buscar_secretos.ARCHIVOS_PROHIBIDOS.search("config/kaggle.json")
    assert not buscar_secretos.ARCHIVOS_PROHIBIDOS.search(".env.example")


@pytest.mark.skipif(shutil.which("git") is None, reason="sin git")
def test_encuentra_un_token_borrado_en_el_historial(tmp_path, monkeypatch, capsys):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "prueba@example.com")
    git("config", "user.name", "prueba")
    (tmp_path / "config.py").write_text(f"TOKEN = '{TOKEN_FALSO}'\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "con token")
    (tmp_path / "config.py").write_text("TOKEN = None\n", encoding="utf-8")
    git("commit", "-qam", "sin token")
    monkeypatch.chdir(tmp_path)
    assert buscar_secretos.main() == 1
    salida = capsys.readouterr().out
    assert "config.py · token de Telegram" in salida
    assert TOKEN_FALSO not in salida                                 # nunca imprime el secreto

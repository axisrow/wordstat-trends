"""Инварианты контейнера сбора (issue #8).

Dockerfile и entrypoint — шелл-инфраструктура, unit-тестов на них нет, поэтому
здесь механические проверки ключевых свойств безопасности: cookies и ключи не
попадают в образ, CDP не слушает наружу, паузы между фразами существуют.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def dockerfile() -> str:
    return (REPO / "deploy" / "Dockerfile").read_text()


def entrypoint() -> str:
    return (REPO / "deploy" / "entrypoint.sh").read_text()


def upload_cookies() -> str:
    return (REPO / "scripts" / "upload_cookies.sh").read_text()


def test_dockerfile_copies_nothing_but_declared_files():
    """В образ копируются только pyproject/uv.lock и entrypoint — не том и не секреты."""
    copies = [line for line in dockerfile().splitlines() if line.startswith("COPY ")]
    assert copies, "в Dockerfile нет ни одной COPY-инструкции"
    for line in copies:
        assert "data" not in line and "cookies" not in line and "key" not in line


def test_dockerfile_has_no_env_secrets():
    for line in dockerfile().splitlines():
        if line.startswith("ENV "):
            assert "cookie" not in line.lower()
            assert "token" not in line.lower()
            assert "key" not in line.lower()


def test_cdp_binds_loopback_only():
    """CDP — полное управление браузером: слушает только 127.0.0.1."""
    assert "--remote-debugging-address=127.0.0.1" in entrypoint()


def test_wait_depends_on_curl_and_image_has_curl():
    """entrypoint ждёт CDP через curl — пакет обязан стоять в образе
    (python:*-slim его не содержит; без него — crash-loop с первого деплоя)."""
    assert "curl -fsS" in entrypoint()
    apt_line = next(line for line in dockerfile().splitlines() if "apt-get install" in line)
    assert "curl" in apt_line.split("install", 1)[1]


def test_entrypoint_has_pause_between_phrases():
    assert "PHRASE_DELAY_S" in entrypoint()
    assert "sleep" in entrypoint()


def test_upload_cookies_never_prints_content():
    """Скрипт заливки не печатает содержимое cookies и не тащит их через env."""
    script = upload_cookies()
    assert "cat " not in script
    assert "set -eu" in script
    # файл уходит scp-потоком и в /tmp на хосте остаётся с правами 600
    assert "install -o 1000 -g 1000 -m 600" in script
    assert "rm -f" in script

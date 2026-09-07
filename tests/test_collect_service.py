"""HTTP-эндпоинт запуска сбора (issue #24).

Проверяется контракт из issue: аутентификация по токену, только POST со
списком фраз, лимит списка, отказ при параллельном запуске (409, не второй
Chrome), наружу — только статус. Сбор подменяется заглушкой subprocess.
"""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from wordstat_trends.collect_service import CollectService, Config

TOKEN = "test-token-123"


@pytest.fixture()
def service(tmp_path, monkeypatch):
    cfg = Config(
        phrases_file=tmp_path / "phrases.txt",
        results_dir=tmp_path / "results",
        git_dir=tmp_path / "repo",  # без .git — коммит просто пропускается
        phrase_delay_s=0,
        trigger_token=TOKEN,
        trigger_port=0,  # ThreadingHTTPServer выберет свободный порт
    )
    svc = CollectService(cfg)

    collected = []
    started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    class FakeProc:
        returncode = 0

    def fake_run(cmd, **_):
        assert cmd[0] == "wordstat" and cmd[1] == "collect"
        with lock:
            collected.append(cmd[2])
        started.set()
        release.wait(timeout=10)
        return FakeProc()

    monkeypatch.setattr("wordstat_trends.collect_service.subprocess.run", fake_run)
    # Живость Chrome (падение → os._exit всего процесса) в тестах не проверяем.
    monkeypatch.setattr(CollectService, "_ensure_chrome_alive", lambda self: None)

    httpd = svc.make_server()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield svc, f"http://127.0.0.1:{httpd.server_port}", collected, started, release
    httpd.shutdown()


def request(url, method="POST", token: str | None = TOKEN, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_status_idle(service):
    _, base, *_ = service
    code, body = request(f"{base}/status", method="GET", token=None)
    assert code == 200
    assert body["state"] == "idle"


def test_collect_requires_token(service):
    _, base, *_ = service
    assert request(f"{base}/collect", token=None)[0] == 401
    assert request(f"{base}/collect", token="wrong")[0] == 401


def test_only_post_on_collect(service):
    _, base, *_ = service
    assert request(f"{base}/collect", method="GET")[0] == 405
    assert request(f"{base}/collect", method="PUT")[0] == 405


def test_unknown_paths_404(service):
    _, base, *_ = service
    assert request(f"{base}/", method="GET", token=None)[0] == 404
    assert request(f"{base}/json/version", method="GET", token=None)[0] == 404  # CDP не проксируется
    assert request(f"{base}/anything", body={"phrases": ["x"]})[0] == 404


def test_collect_bad_payloads(service):
    _, base, *_ = service
    assert request(f"{base}/collect", body={})[0] == 400
    assert request(f"{base}/collect", body={"phrases": "не список"})[0] == 400
    assert request(f"{base}/collect", body={"phrases": [1]})[0] == 400
    assert request(f"{base}/collect", body={"phrases": []})[0] == 400
    assert request(f"{base}/collect", body={"phrases": ["   "]})[0] == 400


def test_collect_phrase_limit(service):
    _, base, *_ = service
    many = [f"фраза {i}" for i in range(51)]  # лимит по умолчанию 50
    code, body = request(f"{base}/collect", body={"phrases": many})
    assert code == 400
    assert "too many" in body["error"]


def test_collect_accepts_and_runs(service):
    svc, base, collected, started, release = service
    code, body = request(f"{base}/collect", body={"phrases": ["фраза", "фраза", " дубликат "]})
    assert code == 202
    assert body["phrases"] == ["фраза", "дубликат"]  # дубликаты и пробелы схлопнуты
    assert started.wait(timeout=5)
    code, status = request(f"{base}/status", method="GET", token=None)
    assert status["state"] == "running"
    release.set()
    for _ in range(200):
        if svc.status()["state"] == "idle":
            break
        threading.Event().wait(0.01)
    assert svc.status()["state"] == "idle"
    assert collected == ["фраза", "дубликат"]


def test_second_request_while_running_gets_409(service):
    _, base, _, started, release = service
    assert request(f"{base}/collect", body={"phrases": ["a"]})[0] == 202
    assert started.wait(timeout=5)
    code, body = request(f"{base}/collect", body={"phrases": ["b"]})
    assert code == 409
    assert body["error"] == "collection already running"
    release.set()


def test_scheduler_and_endpoint_share_lock(service):
    """Расписание и HTTP-запрос делят одну блокировку: пока идёт плановый прогон,
    запрос по HTTP получает 409. Поток планировщика не должен умирать после
    прогона (двойной release лока — регрессия из ревью PR #42)."""
    svc, base, _, started, release = service
    svc.cfg.phrases_file.write_text("# комментарий\nплановая фраза\n\n")
    thread_errors: list[BaseException] = []

    def excepthook(args):
        thread_errors.append(args.exc_value)

    default_hook = threading.excepthook
    threading.excepthook = excepthook
    try:
        scheduler = threading.Thread(target=svc.scheduler_once, daemon=True)
        scheduler.start()
        assert started.wait(timeout=5)
        assert request(f"{base}/collect", body={"phrases": ["c"]})[0] == 409
        release.set()
        scheduler.join(timeout=10)
    finally:
        threading.excepthook = default_hook
    assert not scheduler.is_alive(), "поток планировщика умер или завис"
    assert not thread_errors, f"исключение в потоке планировщика: {thread_errors}"
    # Лок действительно отпущен ровно один раз — следующий запуск возможен.
    assert svc.try_start(["ещё фраза"])
    release.set()
    for _ in range(200):
        if svc.status()["state"] == "idle":
            break
        threading.Event().wait(0.01)
    assert svc.status()["phrases"] == ["ещё фраза"]


def test_collect_garbage_content_length_is_400(service):
    """Мусорный Content-Length — осмысленный 400, а не оборванное соединение.
    http.client сам валидирует заголовок, поэтому сырой сокет."""
    import socket
    from urllib.parse import urlparse

    _, base, *_ = service
    parsed = urlparse(base)
    with socket.create_connection((parsed.hostname, parsed.port), timeout=10) as sock:
        sock.sendall(
            b"POST /collect HTTP/1.1\r\n"
            b"Host: trigger\r\n"
            b"Authorization: Bearer " + TOKEN.encode() + b"\r\n"
            b"Content-Length: \xd0\xbd\xd0\xb5-\xd1\x87\xd0\xb8\xd1\x81\xd0\xbb\xd0\xbe\r\n\r\n"
        )
        data = sock.recv(4096).decode("utf-8", "replace")
    assert " 400 " in data.splitlines()[0], f"ожидали 400, получили: {data.splitlines()[0]}"
    assert "invalid body size" in data


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("TRIGGER_TOKEN", "t")
    monkeypatch.setenv("TRIGGER_PORT", "9001")
    monkeypatch.setenv("PHRASE_DELAY_S", "5")
    cfg = Config.from_env()
    assert cfg.trigger_token == "t"
    assert cfg.trigger_port == 9001
    assert cfg.phrase_delay_s == 5.0


def test_no_token_means_no_endpoint(monkeypatch):
    """Fail closed: без TRIGGER_TOKEN конфигурация не включает эндпоинт,
    main() в этом случае не поднимает HTTP-сервер (см. collect_service.main)."""
    monkeypatch.delenv("TRIGGER_TOKEN", raising=False)
    cfg = Config.from_env()
    assert cfg.trigger_token is None
    service = (Path(__file__).resolve().parents[1] / "src" / "wordstat_trends" / "collect_service.py").read_text()
    assert "if cfg.trigger_token:" in service

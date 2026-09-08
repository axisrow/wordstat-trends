"""HTTP-эндпоинт запуска сбора по требованию (issue #24).

Наружу торчит одна операция: ``POST /collect`` с явным списком фраз плюс
``GET /status``. CDP не проксируется, произвольные команды браузеру не
принимаются, результаты наружу не отдаются — они, как и в расписании,
попадают в репозиторий коммитом.

Суточный планировщик живёт в этом же процессе отдельным потоком и проходит
через ту же блокировку, что и HTTP-запросы: эндпоинт дополняет расписание,
а второй сбор при активном получает 409, а не второй Chrome.

Без ``TRIGGER_TOKEN`` HTTP-часть не поднимается вовсе (fail closed):
эндпоинт без аутентификации не существует.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from wordstat_trends.collect.config import DynamicsWindow, default_window

log = logging.getLogger("wordstat_trends.collect_service")

# Верхняя граница размера запроса и элементов списка: «только POST с явным
# списком фраз», без попыток протащить через эндпоинт что-то ещё.
DEFAULT_MAX_PHRASES = 50
MAX_BODY_BYTES = 32 * 1024
MAX_PHRASE_LEN = 512


@dataclass
class Config:
    cdp_port: int = 9222
    phrases_file: Path = Path("/app/data/phrases.txt")
    results_dir: Path = Path("/app/data/results")
    git_dir: Path = Path("/app/data/repo")
    phrase_delay_s: float = 30.0
    collect_interval_s: float = 86400.0
    trigger_token: str | None = None
    trigger_port: int = 8899
    trigger_max_phrases: int = DEFAULT_MAX_PHRASES
    chrome_pid: int | None = None
    # Окно динамики (issue #5): пятилетний дефолт вместо платформенных 24 мес.
    # None → default_window() считается при запуске сбора.
    dynamics_window: DynamicsWindow | None = None

    @classmethod
    def from_env(cls) -> Config:
        pid = os.environ.get("CHROME_PID")
        return cls(
            cdp_port=int(os.environ.get("CDP_PORT", "9222")),
            phrases_file=Path(os.environ.get("PHRASES_FILE", "/app/data/phrases.txt")),
            results_dir=Path(os.environ.get("RESULTS_DIR", "/app/data/results")),
            git_dir=Path(os.environ.get("GIT_DIR", "/app/data/repo")),
            phrase_delay_s=float(os.environ.get("PHRASE_DELAY_S", "30")),
            collect_interval_s=float(os.environ.get("COLLECT_INTERVAL_S", "86400")),
            trigger_token=os.environ.get("TRIGGER_TOKEN") or None,
            trigger_port=int(os.environ.get("TRIGGER_PORT", "8899")),
            trigger_max_phrases=int(os.environ.get("TRIGGER_MAX_PHRASES", str(DEFAULT_MAX_PHRASES))),
            chrome_pid=int(pid) if pid and pid.isdigit() else None,
        )


@dataclass
class _RunStatus:
    state: str = "idle"  # idle | running
    started_at: float | None = None
    phrases: list[str] = field(default_factory=list)
    collected: int = 0
    failed: int = 0
    last_error: str | None = None
    finished_at: float | None = None

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "phrases": self.phrases,
            "collected": self.collected,
            "failed": self.failed,
            "last_error": self.last_error,
        }


class CollectService:
    """Сбор + блокировка от параллельного запуска + HTTP-интерфейс статуса."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._status = _RunStatus()

    # --- статус ---------------------------------------------------------------

    def status(self) -> dict:
        return self._status.as_dict()

    def is_running(self) -> bool:
        return self._status.state == "running"

    # --- запуск сбора -----------------------------------------------------------

    def try_start(self, phrases: list[str]) -> bool:
        """Занять слот сбора; False, если сбор уже идёт (второй Chrome не нужен)."""
        if not self._lock.acquire(blocking=False):
            return False
        thread = threading.Thread(target=self._run_locked, args=(list(phrases),), daemon=True)
        thread.start()
        # Даём потоку встать в state=running, чтобы ответ 202 соответствовал
        # действительности (и параллельный запрос сразу получил 409).
        for _ in range(100):
            if self.is_running():
                break
            time.sleep(0.01)
        return True

    def _run_locked(self, phrases: list[str]) -> None:
        assert self._lock.locked()
        self._status = _RunStatus(state="running", started_at=time.time(), phrases=phrases)
        try:
            self._collect(phrases)
        except Exception as exc:  # noqa: BLE001 — статус прогона важнее чистоты исключения
            self._status.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("прогон завершился ошибкой")
        finally:
            self._status.state = "idle"
            self._status.finished_at = time.time()
            self._lock.release()

    def _collect(self, phrases: list[str]) -> None:
        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)
        # Осознанный дефолт периода (issue #5): окно задаётся флагами CLI,
        # а не молчаливым 24-месячным дефолтом интерфейса. Валидация рамок —
        # до обращения к браузеру, в том числе для явно заданного окна.
        window = self.cfg.dynamics_window
        if window is None:
            window = default_window()
        else:
            window.validate()
        log.info("окно динамики: %s — %s (%s)", window.date_from, window.date_to, window.granularity)
        for i, phrase in enumerate(phrases):
            self._ensure_chrome_alive()
            log.info("сбор фразы <%s>", phrase)
            proc = subprocess.run(  # noqa: S603 — фиксированная команда, не пользовательский ввод
                [
                    "wordstat",
                    "collect",
                    phrase,
                    "--output-dir",
                    str(self.cfg.results_dir),
                    *window.cli_flags(),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                # Известное ограничение wordstat-cli#2: прогон может падать на
                # середине — фиксируем и идём дальше.
                self._status.failed += 1
                log.warning("фраза <%s> не собрана (rc=%s), продолжаем", phrase, proc.returncode)
            else:
                self._status.collected += 1
            if i < len(phrases) - 1:
                time.sleep(self.cfg.phrase_delay_s)
        self._commit_results()

    def _ensure_chrome_alive(self) -> None:
        """Chrome жив? Если упал — завершаем процесс, Dokku поднимет контейнер."""
        alive = False
        if self.cfg.chrome_pid is not None:
            try:
                os.kill(self.cfg.chrome_pid, 0)
                alive = True
            except OSError:
                alive = False
        else:
            try:
                urllib.request.urlopen(  # noqa: S310 — фиксированный loopback-адрес
                    f"http://127.0.0.1:{self.cfg.cdp_port}/json/version", timeout=5
                ).read()
                alive = True
            except OSError:
                alive = False
        if not alive:
            log.critical("Chrome умер посреди прогона — выходим, Dokku перезапустит")
            os._exit(1)

    def _commit_results(self) -> None:
        """Результаты — коммитом в репозиторий (машинный ключ на томе), не в HTTP."""
        git = self.cfg.git_dir
        if not (git / ".git").is_dir():
            log.info("git-репозиторий результатов не настроен (%s), коммит пропущен", git)
            return
        subprocess.run(["git", "-C", str(git), "add", "."], check=False, capture_output=True)
        commit = subprocess.run(
            [
                "git", "-C", str(git),
                "-c", "user.name=collector", "-c", "user.email=collector@localhost",
                "commit", "-m", f"chore(data): автосбор {time.strftime('%Y-%m-%d', time.gmtime())}",
            ],
            check=False, capture_output=True, text=True,
        )
        if commit.returncode != 0:
            log.warning("коммит результатов не удался (не фатально)")
            return
        push = subprocess.run(
            ["git", "-C", str(git), "push", "--quiet", "origin", "HEAD:main"],
            check=False, capture_output=True, text=True,
        )
        if push.returncode != 0:
            log.warning("пуш результатов не удался (не фатально)")
        else:
            log.info("результаты закоммичены и отправлены")

    # --- планировщик ------------------------------------------------------------

    def scheduler_loop(self) -> None:
        """Первый прогон сразу, далее раз в COLLECT_INTERVAL_S — как в issue #8."""
        while True:
            self.scheduler_once()
            time.sleep(self.cfg.collect_interval_s)

    def scheduler_once(self) -> None:
        """Один плановый прогон по PHRASES_FILE; через ту же блокировку, что и HTTP.

        Лок берём явно и БЕЗ `with`: _run_locked сам отпускает его в finally —
        `with` дал бы второй release() и RuntimeError, убивающий поток
        планировщика после первого же прогона. Блокирующе — плановый прогон
        ждёт завершения запущенного по HTTP, а не конкурирует с ним.
        """
        phrases = self._phrases_from_file()
        if phrases is None:
            log.info("список фраз не найден (%s), прогон по расписанию пропущен", self.cfg.phrases_file)
            return
        self._lock.acquire()
        self._run_locked(phrases)

    def _phrases_from_file(self) -> list[str] | None:
        try:
            lines = self.cfg.phrases_file.read_text().splitlines()
        except OSError:
            return None
        phrases = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return phrases or None

    # --- HTTP ---------------------------------------------------------------------

    def make_server(self) -> ThreadingHTTPServer:
        service = self

        class Handler(_TriggerHandler):
            trigger_service = service

        return ThreadingHTTPServer(("0.0.0.0", self.cfg.trigger_port), Handler)

    def serve_forever(self) -> None:
        with self.make_server() as httpd:
            log.info("HTTP-эндпоинт запуска сбора на 0.0.0.0:%s (CDP наружу не отдаётся)",
                     self.cfg.trigger_port)
            httpd.serve_forever()


class _TriggerHandler(BaseHTTPRequestHandler):
    trigger_service: CollectService
    server_version = "wordstat-trigger/1"

    # Стандартный шум в stderr не нужен — свои строки через logging.
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002, A003
        log.debug(format, *args)

    # --- утилиты ответа ------------------------------------------------------------

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        token = self.trigger_service.cfg.trigger_token
        expected = f"Bearer {token}".encode() if token else None
        return expected is not None and hmac.compare_digest(header.encode(), expected)

    # --- маршруты: ровно две операции, остальное не существует ----------------------

    def do_GET(self) -> None:  # noqa: N802 — имя метода диктует http.server
        if self.path == "/collect":
            self._json(405, {"error": "POST only"})
            return
        if self.path != "/status":
            self._json(404, {"error": "not found"})
            return
        self._json(200, self.trigger_service.status())

    def _method_not_allowed(self) -> None:
        self._json(405, {"error": "POST only"})

    do_PUT = _method_not_allowed  # noqa: N815 — имя метода диктует http.server
    do_PATCH = _method_not_allowed  # noqa: N815
    do_DELETE = _method_not_allowed  # noqa: N815

    def do_POST(self) -> None:  # noqa: N802
        svc = self.trigger_service
        if self.path != "/collect":
            self._json(404, {"error": "not found"})
            return
        if not svc.cfg.trigger_token:
            self._json(503, {"error": "endpoint disabled"})
            return
        if not self._authorized():
            log.warning("отказ: неверный токен, клиент %s", self.client_address[0])
            self._json(401, {"error": "unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            length = -1  # мусорный заголовок — как отсутствующий, ответ 400, а не обрыв
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"error": "invalid body size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "invalid json"})
            return
        phrases = self._validate_phrases(payload)
        if isinstance(phrases, str):
            self._json(400, {"error": phrases})
            return

        # Кто, когда, какие фразы — единственный след при компрометации токена.
        log.info("запуск сбора: клиент=%s фразы=%s", self.client_address[0], phrases)
        if svc.try_start(phrases):
            self._json(202, {"status": "started", "phrases": phrases})
        else:
            log.warning("отказ: сбор уже идёт, клиент %s", self.client_address[0])
            self._json(409, {"error": "collection already running", "status": svc.status()})

    def _validate_phrases(self, payload: object) -> list[str] | str:
        limit = self.trigger_service.cfg.trigger_max_phrases
        if not isinstance(payload, dict) or not isinstance(payload.get("phrases"), list):
            return "expected json object with 'phrases' list"
        phrases: list[str] = []
        for item in payload["phrases"]:
            if not isinstance(item, str):
                return "phrases must be strings"
            phrase = item.strip()
            if not phrase:
                continue
            if len(phrase) > MAX_PHRASE_LEN:
                return f"phrase too long (>{MAX_PHRASE_LEN})"
            if phrase not in phrases:
                phrases.append(phrase)
        if not phrases:
            return "empty phrases list"
        if len(phrases) > limit:
            return f"too many phrases ({len(phrases)} > {limit})"
        return phrases


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = Config.from_env()
    svc = CollectService(cfg)
    threading.Thread(target=svc.scheduler_loop, daemon=True, name="scheduler").start()
    if cfg.trigger_token:
        svc.serve_forever()
    else:
        log.error("TRIGGER_TOKEN не задан — HTTP-эндпоинт отключён, работает только расписание")
        threading.Event().wait()  # планировщик — daemon, ждём вечно


if __name__ == "__main__":
    main()

"""Импорт Яндекс-сессии из cookies.txt на томе в Chrome-профиль (issue #51).

``scripts/upload_cookies.sh`` кладёт файл в том и перезапускает контейнер;
этот модуль — вторая половина пути: до первого ``wordstat collect`` прочитать
файл, установить куки в Chrome через CDP ``Storage.setCookies`` и подтвердить
авторизацию тем же probe, которым живой сбор проверяет сессию.

Формат — Netscape cookies.txt (curl/расширения «cookies.txt»): 7 полей через
TAB — ``domain  tailmatch  path  secure  expires  name  value``; строка с
префиксом ``#HttpOnly_`` — httpOnly-кука, прочие строки на ``#`` — комментарии.
Из файла импортируются только куки доменов Яндекса — остальное оператор мог
выгрузить из браузера целиком, сторонние сессии контейнеру не нужны.

Контракт безопасности (проверяется тестами):

- содержимое кук не логируется и не попадает в env — наружу только счётчики,
  номера строк и домены;
- сообщения об ошибках чужих библиотек (browser_use/cdp_use) проходят через
  ``_sanitize``: вхождения значений и имён кук заменяются на ``[redacted]`` —
  CDP-библиотеки не обещают кратких сообщений, дамп параметров запроса мог бы
  унести куку в лог контейнера;
- ``Storage.setCookies`` перезаписывает только точные совпадения
  (name, domain, path) — импорт идемпотентен и не трогает чужие куки профиля.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from cdp_use.cdp.network.types import CookieParam

log = logging.getLogger("wordstat_trends.cookies_import")

# Хвостовой allowlist доменов Яндекса, точное совпадение или поддомен
# (`host == suffix or host.endswith("." + suffix)`). Копия YANDEX_COOKIE_DOMAIN_
# SUFFIXES из wordstat-cli: в его пине (uv.lock) auth.py ещё нет, а тянуть
# константу из чужого пакета ради четырёх строк не стоит — при смене списка
# в wordstat-cli обновить и здесь (обе стороны про одну и ту же сессию).
YANDEX_COOKIE_DOMAIN_SUFFIXES = ("yandex.ru", "ya.ru", "yandex.com", "yandex.net")

# Реальные экспорты сессии Яндекса — единицы килобайт; лимит отсекает
# случайно подложенный гигантский файл до того, как он уедет в память.
MAX_COOKIES_FILE_BYTES = 1024 * 1024

# Те же константы UI, что в wordstat-cli collector.py: URL, селектор строки
# поиска и probe авторизации. Probe обязан совпадать с тем, что проверяет
# живой сбор, — иначе импорт «подтвердит» сессию, которую collect тут же
# отвергнет.
WORDSTAT_URL = "https://wordstat.yandex.ru/"
QUERY_SELECTOR = 'input[placeholder="Введите слово или словосочетание"]'
AUTH_PROBE = """() => JSON.stringify({
    url: location.href,
    hasLogout: [...document.querySelectorAll('a')].some((link) => link.textContent?.trim() === 'Выйти'),
})"""

RENDER_TIMEOUT_S = 20.0


class CookiesImportError(Exception):
    """Импорт не удался. Сообщение не содержит имён и значений кук."""


@dataclass
class ImportReport:
    """Счётчики импорта — только числа и домены, никогда имена/значения кук."""

    imported: int = 0
    skipped_foreign: int = 0
    skipped_expired: int = 0
    skipped_empty: int = 0
    domains: list[str] = field(default_factory=list)


# --- парсер -------------------------------------------------------------------


def _is_yandex_host(domain: str) -> bool:
    host = domain.removeprefix(".")
    return any(host == suffix or host.endswith("." + suffix) for suffix in YANDEX_COOKIE_DOMAIN_SUFFIXES)


def parse_netscape_cookies(text: str) -> tuple[list[dict], ImportReport]:
    """Netscape cookies.txt -> (список CDP CookieParam, отчёт-счётчики).

    Контракт:

    - строка с 7 TAB-полями проходит; всё иное — ошибка с номером строки,
      но без её содержимого (в мусорной строке может оказаться значение);
    - куки не-Яндексовых доменов пропускаются молча (счётчик), это ожидаемо
      для полной выгрузки браузера;
    - просроченные и пустые пропускаются со счётчиком, как в wordstat login;
    - два совпадения по (name, domain, path) — ошибка: порядок применения
      не определён, выбирать «правильную» недетерминированно нельзя;
    - expires <= 0 — сессионная кука, поле ``expires`` не передаётся.
    """
    report = ImportReport()
    cookies: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # Строку не стрипим целиком: пустое последнее поле (value) — это
        # хвостовой TAB, strip() превратил бы 7 полей в 6. splitlines уже
        # снял переводы строк, включая CRLF; пустоту проверяем по strip-виду.
        if not raw_line.strip() or (
            raw_line.startswith("#") and not raw_line.startswith("#HttpOnly_")
        ):
            continue

        http_only = raw_line.startswith("#HttpOnly_")
        line = raw_line.removeprefix("#HttpOnly_") if http_only else raw_line

        fields = line.split("\t")
        if len(fields) != 7:
            raise CookiesImportError(
                f"строка {lineno}: ожидалось 7 TAB-полей (Netscape cookies.txt), найдено {len(fields)}"
            )
        domain_raw, tailmatch, path, secure, expires, name, value = fields

        if not domain_raw or not name:
            raise CookiesImportError(f"строка {lineno}: пустое домен-поле или имя куки")
        if path[:1] != "/":
            raise CookiesImportError(f"строка {lineno}: path должен начинаться с /")
        tailmatch = tailmatch.strip().upper()
        secure_flag = secure.strip().upper()
        if tailmatch not in ("TRUE", "FALSE") or secure_flag not in ("TRUE", "FALSE"):
            raise CookiesImportError(f"строка {lineno}: поля tailmatch/secure должны быть TRUE или FALSE")
        try:
            expires_at = int(expires.strip())
        except ValueError as error:
            raise CookiesImportError(f"строка {lineno}: expires не целое число (epoch-секунды)") from error

        # tailmatch TRUE — доменная кука (точка), FALSE — host-only (без точки):
        # флаг и точка в Netscape-формате дублируют друг друга, точка — то,
        # что понимает CDP.
        domain = domain_raw if tailmatch == "TRUE" else domain_raw.removeprefix(".")
        if tailmatch == "TRUE" and not domain.startswith("."):
            domain = "." + domain

        if not _is_yandex_host(domain):
            report.skipped_foreign += 1
            continue
        if expires_at > 0 and expires_at < time.time():
            report.skipped_expired += 1
            continue
        if not value:
            report.skipped_empty += 1
            continue

        key = (name, domain, path)
        if key in seen_keys:
            raise CookiesImportError(
                f"строка {lineno}: две куки с одинаковым ключом (domain={domain}, path={path}) — "
                "порядок их применения не определён, отказ"
            )
        seen_keys.add(key)

        cookie: dict = {
            "name": name,
            "value": value,
            "domain": domain,
            "path": path,
            "secure": secure_flag == "TRUE",
            "httpOnly": http_only,
        }
        if expires_at > 0:
            cookie["expires"] = float(expires_at)
        cookies.append(cookie)

    if not cookies:
        raise CookiesImportError(
            "нет пригодных кук Яндекса в файле "
            f"(пропущено: {report.skipped_foreign} чужих доменов, "
            f"{report.skipped_expired} просроченных, {report.skipped_empty} пустых)"
        )
    return cookies, report


# --- CDP ----------------------------------------------------------------------


def _browser_session(cdp_url: str):
    """browser_use-сессия к уже поднятым Chrome (CDP только на 127.0.0.1).

    Импорт browser_use ленив и живёт за отдельной функцией: тесты подменяют
    её фейком и не платят за импорт тяжёлой библиотеки, остальной код модуля
    работает на чистом stdlib.
    """
    from browser_use.browser import BrowserSession

    return BrowserSession(
        cdp_url=cdp_url,
        is_local=True,
        downloads_path=None,
        allowed_domains=["wordstat.yandex.ru", "passport.yandex.ru"],
        keep_alive=True,
    )


async def _wait_for_search_input(page) -> None:
    """Ждать, пока Wordstat отрисует строку поиска (тот же селектор, что сбор)."""
    selector = json.dumps(QUERY_SELECTOR)
    deadline = time.monotonic() + RENDER_TIMEOUT_S
    while time.monotonic() < deadline:
        found = await page.evaluate(f"() => Boolean(document.querySelector({selector}))")
        # Живой CDP возвращает bool, странные бекенды — строку; всё прочее
        # (None при ошибке evaluate и т.п.) не считается «нашлось» и циклится
        # до таймаута, а не молча принимается.
        if found is True or found == "true":
            return
        await asyncio.sleep(0.25)
    raise CookiesImportError(f"Wordstat не отрисовал строку поиска за {RENDER_TIMEOUT_S:g}с")


async def _assert_authorized(page) -> None:
    """Тот же probe, что _assert_authenticated в wordstat-cli collector."""
    state = json.loads(await page.evaluate(AUTH_PROBE))
    if "passport.yandex.ru" in state["url"] or not state["hasLogout"]:
        raise CookiesImportError(
            "куки установлены, но Wordstat отвечает страницей входа — сессия в файле "
            "невалидна или истекла"
        )


def _sanitize(message: str, cookies: list[dict]) -> str:
    """Убрать вхождения имён и значений кук из чужого сообщения об ошибке.

    CDP-библиотеки не обещают кратких сообщений; дамп параметров запроса
    унёс бы куку в лог контейнера. Короткие строки не трогаем — иначе
    «вырежется» половина любого сообщения.
    """
    redacted = message
    for cookie in cookies:
        for field_name in ("value", "name"):
            secret = str(cookie.get(field_name) or "")
            if len(secret) >= 6:
                redacted = redacted.replace(secret, "[redacted]")
    return redacted


async def _set_cookies_and_verify(cdp_url: str, cookies: list[dict]) -> None:
    """Установить куки в Chrome и подтвердить авторизацию открытием Wordstat.

    Storage.setCookies действует из корневого CDP-клиента на весь дефолтный
    контекст браузера — страница нужна только для проверки. Разбор закрытия
    как в wordstat-cli auth.py: сначала страница, потом сессия, ошибки teardown
    глотаются, чтобы не маскировать результат.
    """
    session = _browser_session(cdp_url)
    try:
        await session.start()
        page = None
        try:
            # SameSite не передаём: Netscape-формат его не несёт, Chrome
            # применит дефолт (Lax); навигация по Wordstat — same-site,
            # сессии достаточно.
            await session.cdp_client.send.Storage.setCookies(
                params={"cookies": cast("list[CookieParam]", cookies)}
            )
            page = await session.new_page()
            await page.goto(WORDSTAT_URL)
            await _wait_for_search_input(page)
            await _assert_authorized(page)
        finally:
            if page is not None:
                try:
                    await session.close_page(page)
                except Exception:  # noqa: BLE001 — teardown не должен маскировать результат
                    pass
    except CookiesImportError:
        raise
    except Exception as exc:  # noqa: BLE001 — чужие исключения переупаковываются с санитизацией
        detail = _sanitize(f"{type(exc).__name__}: {exc}", cookies)[:300]
        raise CookiesImportError(f"импорт через CDP не удался: {detail}") from exc
    finally:
        try:
            await session.stop()
        except Exception:  # noqa: BLE001
            pass


def import_session_from_file(cookies_file: Path, cdp_url: str) -> ImportReport:
    """Прочитать файл, импортировать куки в Chrome, подтвердить авторизацию.

    Любая неудача — CookiesImportError; вызывающая сторона обязана завершаться
    (fail closed), а не запускать сбор без сессии.
    """
    try:
        data = cookies_file.read_bytes()
    except OSError as exc:
        raise CookiesImportError(f"файл cookies не читается ({cookies_file}): {type(exc).__name__}") from exc
    if len(data) > MAX_COOKIES_FILE_BYTES:
        raise CookiesImportError(
            f"файл cookies больше лимита {MAX_COOKIES_FILE_BYTES} байт — это не похоже на экспорт сессии"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CookiesImportError(f"файл cookies не является корректным UTF-8 ({cookies_file})") from exc

    cookies, report = parse_netscape_cookies(text)
    asyncio.run(_set_cookies_and_verify(cdp_url, cookies))
    report.imported = len(cookies)
    report.domains = sorted({str(cookie["domain"]).lstrip(".") for cookie in cookies})
    log.info(
        "импорт cookies: %d кук, домены %s; пропущено: %d чужих, %d просроченных, %d пустых",
        report.imported, ", ".join(report.domains),
        report.skipped_foreign, report.skipped_expired, report.skipped_empty,
    )
    return report

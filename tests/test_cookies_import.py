"""Контракт импорта cookies.txt в Chrome-профиль (issue #51).

Проверяется: строгий формат Netscape (7 TAB-полей, только Яндекс-домены),
fail closed при любой неудаче импорта/проверки авторизации, и главный
инвариант — содержимое кук не попадает ни в исключения, ни в лог-строки,
ни в env. CDP-часть гоняется на фейковой сессии: browser_use в юнит-тестах
не поднимается, как и живой Chrome.
"""

import json
import time
from pathlib import Path

import pytest

from wordstat_trends import cookies_import
from wordstat_trends.collect_service import Config, bootstrap_session
from wordstat_trends.cookies_import import CookiesImportError, import_session_from_file

SECRET = "kX9-SECRET-cookie-value-do-not-print"


def cookie_line(
    domain=".yandex.ru",
    tailmatch="TRUE",
    path="/",
    secure="TRUE",
    expires: int | str = 4102444800,  # 2100-01-01; str — для тестов мусорного значения
    name="Session",
    value=SECRET,
) -> str:
    return f"{domain}\t{tailmatch}\t{path}\t{secure}\t{expires}\t{name}\t{value}"


# --- парсер ---------------------------------------------------------------------


def test_valid_lines_become_cdp_params():
    cookies, report = cookies_import.parse_netscape_cookies(
        "# Netscape HTTP Cookie File\n"
        "\n"
        + cookie_line()
        + "\n"
        + cookie_line(domain="wordstat.yandex.ru", tailmatch="FALSE", secure="FALSE", name="uid", value="42")
        + "\n"
        + "#HttpOnly_" + cookie_line(name="yandex_gid", value="77")
    )
    assert [c["name"] for c in cookies] == ["Session", "uid", "yandex_gid"]
    first = cookies[0]
    assert first["domain"] == ".yandex.ru"  # tailmatch TRUE — доменная кука с точкой
    assert first["secure"] is True
    assert first["httpOnly"] is False
    assert first["expires"] == 4102444800.0
    assert first["value"] == SECRET

    host_only = cookies[1]
    assert host_only["domain"] == "wordstat.yandex.ru"  # FALSE — host-only, точки нет
    assert host_only["secure"] is False

    assert cookies[2]["httpOnly"] is True  # префикс #HttpOnly_
    assert report.imported == 0  # отчёт заполняет только импорт целиком
    assert report.skipped_foreign == 0


def test_session_cookie_has_no_expires():
    cookies, _ = cookies_import.parse_netscape_cookies(cookie_line(expires=0))
    assert "expires" not in cookies[0]
    cookies, _ = cookies_import.parse_netscape_cookies(cookie_line(expires=-1))
    assert "expires" not in cookies[0]


def test_dot_domain_without_tailmatch_normalised():
    """tailmatch FALSE + ведущая точка — точка снимается: host-only кука."""
    cookies, _ = cookies_import.parse_netscape_cookies(cookie_line(domain=".ya.ru", tailmatch="FALSE"))
    assert cookies[0]["domain"] == "ya.ru"


def test_foreign_domains_skipped():
    text = "\n".join(
        [
            cookie_line(domain=".example.com", name="a"),
            cookie_line(domain=".yandex.zoom.us", name="b"),  # похожее, но не Яндекс
            cookie_line(domain=".notyandex.ru", name="c"),
            cookie_line(domain=".metrica.yandex.ru", name="d"),  # поддомен — импортируется
        ]
    )
    cookies, report = cookies_import.parse_netscape_cookies(text)
    assert [c["name"] for c in cookies] == ["d"]
    assert report.skipped_foreign == 3


def test_expired_and_empty_skipped():
    past = int(time.time()) - 1000
    text = "\n".join(
        [
            cookie_line(name="old", expires=past),
            cookie_line(name="empty", value=""),
            cookie_line(name="live"),
        ]
    )
    cookies, report = cookies_import.parse_netscape_cookies(text)
    assert [c["name"] for c in cookies] == ["live"]
    assert report.skipped_expired == 1
    assert report.skipped_empty == 1


def test_no_yandex_cookies_is_error():
    with pytest.raises(CookiesImportError, match="нет пригодных кук"):
        cookies_import.parse_netscape_cookies(cookie_line(domain=".example.com"))


def test_malformed_line_reports_number_not_content():
    """Мусорная строка — ошибка с номером строки; содержимое (значение куки!)
    в сообщение не попадает — мусорная строка может содержать его целиком."""
    text = f"# comment\n{SECRET}\tsome\tgarbage\n"
    with pytest.raises(CookiesImportError) as excinfo:
        cookies_import.parse_netscape_cookies(text)
    assert "строка 2" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


@pytest.mark.parametrize(
    "line",
    [
        cookie_line(path="no-slash"),
        cookie_line(tailmatch="YES"),
        cookie_line(secure="1"),
        cookie_line(expires="soon"),
        cookie_line(name=""),
        "only\tthree\tfields",
    ],
)
def test_bad_fields_rejected(line):
    with pytest.raises(CookiesImportError, match="строка 1"):
        cookies_import.parse_netscape_cookies(line)


def test_duplicate_key_rejected():
    text = cookie_line() + "\n" + cookie_line(value="other")
    with pytest.raises(CookiesImportError) as excinfo:
        cookies_import.parse_netscape_cookies(text)
    assert "строка 2" in str(excinfo.value)
    # в сообщении нет имени куки — только домен и путь
    assert "Session" not in str(excinfo.value)


# --- CDP-оркестрация на фейке ----------------------------------------------------


class FakePage:
    def __init__(self, authorized: bool):
        self.authorized = authorized
        self.gotos: list[str] = []

    async def goto(self, url: str) -> None:
        self.gotos.append(url)

    async def evaluate(self, script: str):
        if "JSON.stringify" in script:  # AUTH_PROBE
            state = {
                "url": "https://wordstat.yandex.ru/" if self.authorized else "https://passport.yandex.ru/auth",
                "hasLogout": self.authorized,
            }
            return json.dumps(state)
        return True  # QUERY_SELECTOR найден — как живой CDP, bool


class FakeSession:
    def __init__(self, page: FakePage):
        self.page = page
        self.stopped = False
        self.set_params: dict | None = None
        self.start_error: Exception | None = None

        outer = self

        class _Storage:
            async def setCookies(self, params):  # noqa: N802 — имя диктует cdp_use
                outer.set_params = params

        class _Send:
            Storage = _Storage()

        class _Client:
            send = _Send()

        self.cdp_client = _Client()

    async def start(self) -> None:
        if self.start_error:
            raise self.start_error

    async def new_page(self) -> FakePage:
        return self.page

    async def close_page(self, page) -> None:  # noqa: ARG002
        pass

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture()
def fake_session(monkeypatch):
    sessions: list[FakeSession] = []

    def make(page: FakePage) -> FakeSession:
        session = FakeSession(page)
        sessions.append(session)
        monkeypatch.setattr(cookies_import, "_browser_session", lambda cdp_url: session)
        return session

    make.all = sessions  # type: ignore[attr-defined]
    return make


def write_cookies(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cookies.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_import_sets_cookies_and_verifies_authorization(tmp_path, fake_session):
    session = fake_session(FakePage(authorized=True))
    report = import_session_from_file(
        write_cookies(tmp_path, cookie_line() + "\n" + cookie_line(name="other", value="v2")),
        "http://127.0.0.1:9222",
    )
    assert session.set_params is not None
    sent = session.set_params["cookies"]
    assert [c["name"] for c in sent] == ["Session", "other"]
    assert session.page.gotos == [cookies_import.WORDSTAT_URL]
    assert session.stopped
    assert report.imported == 2
    assert report.domains == ["yandex.ru"]


def test_import_fails_closed_when_not_authorized(tmp_path, fake_session):
    fake_session(FakePage(authorized=False))
    with pytest.raises(CookiesImportError, match="страницей входа"):
        import_session_from_file(write_cookies(tmp_path, cookie_line()), "http://127.0.0.1:9222")


def test_third_party_error_sanitized(tmp_path, fake_session):
    """Стороннее исключение с кукой в сообщении не уносит её наружу."""
    session = fake_session(FakePage(authorized=True))
    session.start_error = RuntimeError(f"cdp handshake failed, echoed params: {SECRET}")
    with pytest.raises(CookiesImportError) as excinfo:
        import_session_from_file(write_cookies(tmp_path, cookie_line()), "http://127.0.0.1:9222")
    assert SECRET not in str(excinfo.value)
    assert "[redacted]" in str(excinfo.value)


def test_oversized_file_rejected(tmp_path, fake_session):
    big = tmp_path / "cookies.txt"
    big.write_bytes(b"# padding\n" * 131072)  # > 1 MiB
    with pytest.raises(CookiesImportError, match="лимита"):
        import_session_from_file(big, "http://127.0.0.1:9222")
    assert fake_session.all == []  # до CDP даже не дошли


def test_non_utf8_file_rejected(tmp_path, fake_session):
    bad = tmp_path / "cookies.txt"
    bad.write_bytes(b"\xff\xfe" + cookie_line().encode("utf-8"))
    with pytest.raises(CookiesImportError, match="UTF-8"):
        import_session_from_file(bad, "http://127.0.0.1:9222")


def test_missing_file_rejected(tmp_path, fake_session):
    with pytest.raises(CookiesImportError, match="не читается"):
        import_session_from_file(tmp_path / "nope.txt", "http://127.0.0.1:9222")


# --- bootstrap в сервисе: fail closed --------------------------------------------


def test_config_reads_cookies_file_from_env(monkeypatch):
    monkeypatch.setenv("COOKIES_FILE", "/somewhere/cookies.txt")
    assert Config.from_env().cookies_file == Path("/somewhere/cookies.txt")


def test_bootstrap_without_file_is_silent(monkeypatch, tmp_path):
    """Файла нет — работаем с профилем на томе, процесс живёт."""
    monkeypatch.setenv("COOKIES_FILE", str(tmp_path / "absent.txt"))
    bootstrap_session(Config.from_env())  # не должно бросать


def test_bootstrap_fails_closed_on_import_error(monkeypatch, tmp_path):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text(cookie_line())

    def broken(_path, _cdp_url):
        raise CookiesImportError("куки установлены, но Wordstat отвечает страницей входа")

    monkeypatch.setattr("wordstat_trends.collect_service.import_session_from_file", broken)
    cfg = Config(cookies_file=cookies)
    with pytest.raises(SystemExit) as excinfo:
        bootstrap_session(cfg)
    assert excinfo.value.code == 2


def test_bootstrap_uses_cdp_port(monkeypatch, tmp_path):
    """CDP-адрес для импорта строится из того же порта, что поднят Chrome."""
    cookies = tmp_path / "cookies.txt"
    cookies.write_text(cookie_line())
    seen: list[str] = []

    def ok(_path, cdp_url):
        seen.append(cdp_url)

    monkeypatch.setattr("wordstat_trends.collect_service.import_session_from_file", ok)
    bootstrap_session(Config(cookies_file=cookies, cdp_port=9333))
    assert seen == ["http://127.0.0.1:9333"]

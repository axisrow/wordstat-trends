"""Сборка статической витрины (issue #21) в два языка: ru (корень) и zh (/zh/...).

Без фреймворков и внешних зависимостей: stdlib только. Весь текст интерфейса
живёт в site/locales/*.json и подставляется в шаблоны через {{ключ}}.

Критерий приёмки issue #21: строку интерфейса нельзя добавить в обход
механизма локализации. Его обеспечивает `lint_templates`:
  1) буквальный текст в текстовых узлах шаблона — ошибка сборки;
  2) буквальный текст в атрибутах title/alt/aria-label/placeholder — ошибка;
  3) ключ, отсутствующий в любой из локалей, — ошибка;
  4) неизвестный ключ в шаблоне (опечатка) — ошибка.
Запуск: `python scripts/build_site.py [выходной-каталог]` (по умолчанию _site).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent.parent / "site"
# В CI витрина собирается без установки зависимостей проекта: i18n.py — pure
# stdlib, поэтому src/ достаточно добавить в путь при отсутствии пакета.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wordstat_trends.i18n import DEFAULT_LOCALE, format_date  # noqa: E402

TEMPLATES_DIR = SITE_DIR / "templates"
LOCALES_DIR = SITE_DIR / "locales"
ASSETS_DIR = SITE_DIR / "assets"

BASE_URL = "https://axisrow.github.io/wordstat-trends"
REPO_URL = "https://github.com/axisrow/wordstat-trends"

# страница → (файл, ключ заголовка)
PAGES: dict[str, tuple[str, str]] = {
    "index": ("index.html", "index.title"),
    "trends": ("trends.html", "trends.title"),
    "about": ("about.html", "about.title"),
}

PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")
# Буквы любого алфавита (включая CJK) — то, что нельзя писать в шаблоне напрямую.
LETTER_RE = re.compile(r"[\w]", re.UNICODE)
TEXT_ATTRS = ("title", "alt", "aria-label", "placeholder")


class BuildError(Exception):
    """Нарушение механизма локализации или сломанный шаблон."""


def load_locales() -> dict[str, dict[str, str]]:
    """Читает site/locales/*.json; контролирует одинаковый набор ключей."""
    locales: dict[str, dict[str, str]] = {}
    for path in sorted(LOCALES_DIR.glob("*.json")):
        locales[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if DEFAULT_LOCALE not in locales:
        raise BuildError(f"нет файла локали по умолчанию: {DEFAULT_LOCALE}.json")
    base = locales[DEFAULT_LOCALE]
    for code, messages in locales.items():
        missing = base.keys() - messages.keys()
        extra = messages.keys() - base.keys()
        if missing:
            raise BuildError(f"локаль {code} не содержит ключи: {sorted(missing)}")
        if extra:
            raise BuildError(f"локаль {code} содержит лишние ключи: {sorted(extra)}")
    return locales


def lint_templates() -> None:
    """Гейт «ни одной строки текста напрямую в шаблоне».

    Работает по исходникам шаблонов, до подстановки: если в текстовом узле
    или текстовом атрибуте останется буква (кириллица, латиница, CJK) вне
    плейсхолдера {{ключ}} — сборка падает с указанием файла и фрагмента.
    """
    for path in sorted(TEMPLATES_DIR.glob("*.html")):
        src = path.read_text(encoding="utf-8")
        for attr in TEXT_ATTRS:
            for value in re.findall(rf'{attr}="([^"]*)"', src):
                stripped = PLACEHOLDER_RE.sub("", value)
                if LETTER_RE.search(stripped):
                    raise BuildError(f"{path.name}: буквальный текст в атрибуте {attr}: {value!r}")
        text_only = re.sub(r"<[^>]+>", " ", src)
        text_only = PLACEHOLDER_RE.sub("", text_only)
        if LETTER_RE.search(text_only):
            snippet = " ".join(text_only.split())[:80]
            raise BuildError(f"{path.name}: буквальный текст вне механизма i18n: {snippet!r}")


def render(template: str, messages: dict[str, str], context: dict[str, str], where: str) -> str:
    """Подставляет {{ключ}} из переводов и структурного контекста.

    Неизвестный ключ — ошибка: опечатка в шаблоне не может пройти молча.
    """

    def resolve(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            return context[key]
        if key in messages:
            return messages[key]
        raise BuildError(f"{where}: неизвестный ключ {key!r}")

    return PLACEHOLDER_RE.sub(resolve, template)


def build(out_dir: Path, build_date: date | None = None) -> list[Path]:
    """Собирает страницы обеих локалей и ассеты; возвращает список страниц."""
    locales = load_locales()
    lint_templates()
    layout = (TEMPLATES_DIR / "layout.html").read_text(encoding="utf-8")
    fragments = {name: (TEMPLATES_DIR / f"{name}.html").read_text(encoding="utf-8") for name in PAGES}
    build_date = build_date or date.today()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(ASSETS_DIR, out_dir / "assets")

    written: list[Path] = []
    for code, messages in locales.items():
        prefix = "" if code == DEFAULT_LOCALE else f"{code}/"
        assets = "assets/" if code == DEFAULT_LOCALE else "../assets/"
        alt_code = next(c for c in locales if c != code) if len(locales) > 1 else code
        alt_prefix = "" if alt_code == DEFAULT_LOCALE else f"{alt_code}/"

        for name, (filename, title_key) in PAGES.items():
            hrefs = {f"{other}.href": f"{prefix}{other}.html" for other in PAGES}
            # Переключатель языка ведёт на ту же страницу другой локали;
            # из /zh/... в корень — относительный ../имя.html.
            alt_href = f"../{name}.html" if prefix and not alt_prefix else f"{alt_prefix}{name}.html"
            context = {
                **hrefs,
                "alt.href": alt_href,
                "alt.lang": alt_code,
                "html.lang": code,
                "assets": assets,
                "page.canonical": f"{BASE_URL}/{prefix}{name}.html",
                "page.title": messages[title_key],
                "build.date": format_date(build_date, code),
                "repo.href": REPO_URL,
                **{f"nav.{other}.current": "page" if other == name else "" for other in PAGES},
            }
            body = render(fragments[name], messages, context, f"{code}/{filename}")
            page = render(layout, messages, {**context, "content": body.strip()}, f"{code}/layout")
            target = out_dir / f"{prefix}{filename}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(page, encoding="utf-8")
            written.append(target)
    return written


def main(argv: list[str]) -> int:
    out_dir = Path(argv[1]) if len(argv) > 1 else Path("_site")
    try:
        pages = build(out_dir)
    except BuildError as exc:
        print(f"ошибка сборки: {exc}", file=sys.stderr)
        return 1
    print(f"собрано {len(pages)} страниц ({', '.join(sorted(set(p.parent.name or '.' for p in pages)))}) → {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

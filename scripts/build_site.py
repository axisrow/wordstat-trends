"""Сборка статической витрины (issue #21) в два языка: ru (корень) и zh (/zh/...).

Без фреймворков и внешних зависимостей: stdlib только. Весь текст интерфейса
живёт в site/locales/*.json и подставляется в шаблоны через {{ключ}}.

Данные витрины (issue #104) — JSON-артефакт ранжирования (схема
``showcase/v1``, пишется ядром ``wordstat_trends.showcase``). Сборка читает
его сама (json, не пакет: артефакт пишется в CI без зависимостей проекта);
отсутствие или пустота артефакта — пустое состояние, не ошибка (#21 п.7).

Критерий приёмки issue #21: строку интерфейса нельзя добавить в обход
механизма локализации. Его обеспечивает `lint_templates`:
  1) буквальный текст в текстовых узлах шаблона — ошибка сборки;
  2) буквальный текст в любом неструктурном атрибуте (title, alt, aria-label,
     data-*, ...) — ошибка; структурные атрибуты — allow-list STRUCTURAL_ATTRS;
  3) ключ, отсутствующий в любой из локалей, — ошибка;
  4) неизвестный ключ в шаблоне (опечатка) — ошибка.
HTML-фрагменты данных (строки таблиц, секции) строятся здесь с теми же
{{ключами}} и проходят через `render` — опечатка в ключе падает так же.
Запуск: `python scripts/build_site.py [выходной-каталог] [--artifact путь]`
(по умолчанию _site; --artifact опционален).
"""

from __future__ import annotations

import argparse
import html
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

from wordstat_trends.i18n import (  # noqa: E402
    DEFAULT_LOCALE,
    format_date,
    format_score,
)

TEMPLATES_DIR = SITE_DIR / "templates"
LOCALES_DIR = SITE_DIR / "locales"
ASSETS_DIR = SITE_DIR / "assets"

#: Схема артефакта (см. wordstat_trends.showcase.SCHEMA); строкой, а не
#: импортом: showcase тянет pandas через trends.ranking, а сборка сайта
#: обязана идти на голом stdlib (CI без зависимостей проекта).
SHOWCASE_SCHEMA = "showcase/v1"

#: Сколько фраз каждой секции показывать на главной: тренды — сразу
#: (эпик #1), без прокрутки простыня всех классов; полная таблица — на
#: странице трендов.
INDEX_TOP = 5

#: Порядок секций витрины — CLASS_ORDER из ядра (trends/ranking.py),
#: машинные имена TrendClass → ключ локали заголовка секции.
SECTION_KEYS: dict[str, str] = {
    "GROWING": "trends.class.growing",
    "SEASONAL": "trends.class.seasonal",
    "FALLING": "trends.class.falling",
    "STABLE": "trends.class.stable",
}

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
# Структурные атрибуты: их значения содержат буквы по смыслу (URL, классы,
# роли) и переводом не являются. Всё, чего нет в списке, обязано быть либо
# плейсхолдером, либо не содержать букв — иначе строку можно протащить мимо
# механизма локализации (например, через data-*).
STRUCTURAL_ATTRS = frozenset({
    "href", "class", "id", "lang", "rel", "scope", "role", "charset",
    "type", "name", "media", "http-equiv", "aria-current",
})


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
    или в любом НЕ структурном атрибуте (title, alt, data-*, ...) останется
    буква (кириллица, латиница, CJK) вне плейсхолдера {{ключ}} — сборка
    падает с указанием файла и атрибута. Структурные атрибуты (href, class,
    lang, ...) проверяются по allow-list STRUCTURAL_ATTRS: расширять его
    нужно осознанно, новый текстовый атрибут в него не добавляется.
    """
    for path in sorted(TEMPLATES_DIR.glob("*.html")):
        src = path.read_text(encoding="utf-8")
        for tag in re.findall(r"<[a-zA-Z][^>]*>", src):
            for attr, value in re.findall(r'([a-zA-Z-]+)="([^"]*)"', tag):
                if attr in STRUCTURAL_ATTRS:
                    continue
                # content= мета-вьюпорта — конфигурация браузера («width=device-
                # width, initial-scale=1»), не текст. content= описания страницы
                # остаётся под гейтом и обязан быть плейсхолдером.
                if attr == "content" and "name=\"viewport\"" in tag:
                    continue
                stripped = PLACEHOLDER_RE.sub("", value)
                if LETTER_RE.search(stripped):
                    raise BuildError(f"{path.name}: буквальный текст в атрибуте {attr}: {value!r}")
        text_only = re.sub(r"<[^>]+>", " ", src)
        text_only = PLACEHOLDER_RE.sub("", text_only)
        if LETTER_RE.search(text_only):
            snippet = " ".join(text_only.split())[:80]
            raise BuildError(f"{path.name}: буквальный текст вне механизма i18n: {snippet!r}")


def load_artifact(path: Path | str | None) -> dict | None:
    """Артефакт витрины целиком; None — файла нет или фраз в нём нет.

    Битая схема — ошибка сборки: молча показать пустое состояние на
    испорченном артефакте значило бы публиковать витрину «данных нет»
    поверх существующих данных.
    """

    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SHOWCASE_SCHEMA:
        raise BuildError(f"{path}: не артефакт витрины (ожидалась схема {SHOWCASE_SCHEMA})")
    phrases = data.get("phrases")
    if not isinstance(phrases, list):
        raise BuildError(f"{path}: артефакт без списка phrases")
    return data if phrases else None


def _class_label(klass: str, messages: dict[str, str]) -> str:
    key = SECTION_KEYS.get(klass)
    if key is None:
        raise BuildError(f"неизвестный класс фразы в артефакте: {klass!r}")
    return messages[key]


def escape_data(value: object) -> str:
    """Данные артефакта → текстовый узел HTML.

    ``html.escape`` гасит разметку, но не трогает фигурные скобки: фраза
    вида ``{{trends.empty.text}}`` иначе попала бы в позицию шаблона и
    подставилась локализованной строкой (или уронила сборку на неизвестном
    ключе) — поэтому ``{{`` нейтрализуется сразу.
    """

    return html.escape(str(value)).replace("{{", "{ {")


def render_index_sections(artifact: dict) -> str:
    """Секции классов для главной: тренды на первом экране (эпик #1).

    Фразы — данные (экранируются), заголовки секций — {{ключи}} локали;
    фрагмент затем проходит через `render`, поэтому опечатка в ключе
    падает сборкой, как и в шаблонах.
    """

    phrases = artifact["phrases"]
    sections: list[str] = []
    for klass, key in SECTION_KEYS.items():
        rows = [p for p in phrases if p["class"] == klass][:INDEX_TOP]
        if not rows:
            continue
        items = "".join(
            f"<li>{escape_data(p['phrase'])}</li>" for p in rows
        )
        sections.append(
            f'<section class="trend-section">\n'
            f'<h2>{{{{{key}}}}}</h2>\n<ul class="trend-list">\n{items}\n</ul>\n</section>'
        )
    return "\n".join(sections)


def render_trends_body(
    artifact: dict | None, messages: dict[str, str], code: str
) -> str:
    """Тело страницы трендов: пустое состояние или таблица полной витрины."""

    if not artifact:

        return (
            '<div class="empty-state">\n'
            '<p class="empty-title">{{trends.empty.title}}</p>\n'
            '<p>{{trends.empty.text}}</p>\n'
            '<p class="empty-hint">{{trends.empty.hint}}</p>\n'
            '</div>'
        )
    data_note = ""
    generated_at = artifact.get("generated_at")
    if generated_at:
        day = date.fromisoformat(generated_at)
        data_note = (
            f'<p class="data-date">{{{{trends.generated}}}} '
            f"{format_date(day, code)}</p>"
        )
    rows = []
    phrases = artifact["phrases"]
    for p in phrases:
        phrase = escape_data(p["phrase"])
        klass = html.escape(_class_label(str(p["class"]), messages))
        rank = escape_data(p["rank"])
        score = format_score(float(p["score"]), code) if p.get("components") else "—"
        rows.append(
            f"<tr><td>{rank}</td><td>{phrase}</td><td>{klass}</td><td>{score}</td></tr>"
        )
    body = "\n".join(rows)
    return (
        f"{data_note}\n"
        '<table class="trends-table">\n'
        '<caption class="visually-hidden">{{trends.title}}</caption>\n'
        "<thead>\n<tr>\n"
        '<th scope="col">{{trends.col.rank}}</th>\n'
        '<th scope="col">{{trends.col.phrase}}</th>\n'
        '<th scope="col">{{trends.col.class}}</th>\n'
        '<th scope="col">{{trends.col.score}}</th>\n'
        "</tr>\n</thead>\n"
        f"<tbody>\n{body}\n</tbody>\n"
        "</table>"
    )


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


def build(
    out_dir: Path, build_date: date | None = None, artifact_path: Path | str | None = None
) -> list[Path]:
    """Собирает страницы обеих локалей и ассеты; возвращает список страниц.

    ``artifact_path`` — JSON-артефакт ранжирования (showcase/v1); его
    отсутствие/пустота → пустое состояние (сборка не падает, #21 п.7).
    """
    locales = load_locales()
    lint_templates()
    artifact = load_artifact(artifact_path)
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
            # Данные витрины (#104): секции главной и тело страницы трендов
            # собираются из артефакта и проходят через тот же render —
            # текст интерфейса в них живёт за {{ключами}} локалей.
            if name == "index":
                context["index.sections"] = (
                    render(
                        render_index_sections(artifact),
                        messages,
                        context,
                        f"{code}/index.sections",
                    )
                    if artifact
                    else ""
                )
            elif name == "trends":
                context["trends.body"] = render(
                    render_trends_body(artifact, messages, code),
                    messages,
                    context,
                    f"{code}/trends.body",
                )
            body = render(fragments[name], messages, context, f"{code}/{filename}")
            page = render(layout, messages, {**context, "content": body.strip()}, f"{code}/layout")
            target = out_dir / f"{prefix}{filename}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(page, encoding="utf-8")
            written.append(target)
    return written


def main(argv: list[str]) -> int:
    # argv — как из sys.argv (с именем скрипта), parse_args ждёт аргументы без него
    args = parse_args(argv[1:])
    try:
        pages = build(args.out_dir, artifact_path=args.artifact)
    except BuildError as exc:
        print(f"ошибка сборки: {exc}", file=sys.stderr)
        return 1
    source = f" из {args.artifact}" if args.artifact else ""
    dirs = ", ".join(sorted(set(p.parent.name or "." for p in pages)))
    print(f"собрано {len(pages)} страниц ({dirs}) → {args.out_dir}/{source}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сборка статической витрины (ru + zh) из шаблонов и артефакта"
    )
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("_site"))
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="JSON-артефакт витрины (showcase/v1); без него — пустое состояние",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

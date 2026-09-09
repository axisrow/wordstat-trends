"""Сборка статической витрины (issue #21) в два языка: ru (корень) и zh (/zh/...).

Без фреймворков и внешних зависимостей: stdlib только. Весь текст интерфейса
живёт в site/locales/*.json и подставляется в шаблоны через {{ключ}}.

Данные витрины (issue #104) — JSON-артефакт ранжирования (схема
``showcase/v2``, пишется ядром ``wordstat_trends.showcase``). Сборка читает
его сама (json, не пакет: артефакт пишется в CI без зависимостей проекта);
отсутствие или пустота артефакта — пустое состояние, не ошибка (#21 п.7).
Читается и v1 — карточки без графика и объяснения (ряд появился в v2, #105).

Карточки трендов (issue #105): заголовок-фраза, класс, скор, график истории
(inline SVG, генерируется здесь — статика без JS и внешних библиотек:
ограничение Pages-сборки ``uv run --no-project``) и объяснение «почему фраза
в списке» — шаблоны ``explain.*`` локалей с подстановкой чисел тем же
механизмом, что в ``wordstat_trends.explain`` (#20; сам модуль ядру здесь
недоступен — тянет pandas).

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
(по умолчанию _site и site_data/showcase.json — артефакт, который
Dokku-сборка коммитит в репозиторий, issue #107).
"""

from __future__ import annotations

import argparse
import ast
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
    format_month,
    format_number,
    format_ratio,
    format_score,
)
from wordstat_trends.i18n_phrases import (  # noqa: E402
    DEFAULT_CACHE_PATH as PHRASES_CACHE_PATH,
)
from wordstat_trends.i18n_phrases import (  # noqa: E402
    CacheError,
    TranslatedPhrase,
    TranslationCache,
    display_phrase,
)

TEMPLATES_DIR = SITE_DIR / "templates"
LOCALES_DIR = SITE_DIR / "locales"
ASSETS_DIR = SITE_DIR / "assets"

#: Каталог артефактов витрины в репозитории (issue #107): Dokku-сборка
#: коммитит сюда JSON ранжирования; артефакт каждого прогона заменяет
#: предыдущий, история живёт в git. Каталог не исключается .gitignore.
SITE_DATA_DIR = SITE_DIR.parent / "site_data"

#: Артефакт витрины по умолчанию (showcase/v2 из wordstat_trends.showcase);
#: файла нет (первые сборки до коммита результатов) — пустое состояние.
DEFAULT_ARTIFACT_PATH = SITE_DATA_DIR / "showcase.json"

#: Схема артефакта (см. wordstat_trends.showcase.SCHEMA); строкой, а не
#: импортом: showcase тянет pandas через trends.ranking, а сборка сайта
#: обязана идти на голом stdlib (CI без зависимостей проекта).
SHOWCASE_SCHEMA = "showcase/v2"

#: Читаемые схемы: v1 (#104) не содержит ряда и ratio — карточки без
#: графика и объяснения (расширение сериализации — #105).
READABLE_SCHEMAS = frozenset({"showcase/v1", SHOWCASE_SCHEMA})

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

#: Метка класса в карточке («растёт», не заголовок секции «Растущие
#: запросы») — ключи локали из #20.
CARD_CLASS_KEYS: dict[str, str] = {
    "GROWING": "trend.class.growing",
    "SEASONAL": "trend.class.seasonal",
    "FALLING": "trend.class.falling",
    "STABLE": "trend.class.stable",
}

#: Шаблон объяснения «почему в списке» по классу — те же ключи локалей,
#: что в wordstat_trends.explain (#20): {ratio} и {months} подставляются
#: числами, отформатированными по локали.
EXPLAIN_KEYS: dict[str, str] = {
    "GROWING": "explain.growing",
    "SEASONAL": "explain.seasonal",
    "FALLING": "explain.falling",
    "STABLE": "explain.stable",
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
    # геометрия SVG-графика (issue #105): значения — числа/ключевые слова
    # раскладки, перевода не содержат
    "viewBox", "preserveAspectRatio",
    "x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "r",
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


def _lint_html(src: str, where: str) -> None:
    """Общий гейт «ни одной строки текста напрямую» для HTML-исходника.

    Работает до подстановки: если в текстовом узле или в любом НЕ
    структурном атрибуте (title, alt, data-*, ...) останется буква
    (кириллица, латиница, CJK) вне плейсхолдера {{ключ}} — ошибка сборки
    с указанием места и атрибута. Структурные атрибуты (href, class,
    lang, ...) проверяются по allow-list STRUCTURAL_ATTRS: расширять его
    нужно осознанно, новый текстовый атрибут в него не добавляется.
    """

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
                raise BuildError(f"{where}: буквальный текст в атрибуте {attr}: {value!r}")
    text_only = re.sub(r"<[^>]+>", " ", src)
    text_only = PLACEHOLDER_RE.sub("", text_only)
    if LETTER_RE.search(text_only):
        snippet = " ".join(text_only.split())[:80]
        raise BuildError(f"{where}: буквальный текст вне механизма i18n: {snippet!r}")


def lint_templates() -> None:
    """Гейт шаблонов site/templates/*.html: весь текст интерфейса — {{ключами}}."""

    for path in sorted(TEMPLATES_DIR.glob("*.html")):
        _lint_html(path.read_text(encoding="utf-8"), path.name)


#: Сентинел подстановки данных при разборе f-строк линтом фрагментов:
#: буквы не содержит — сам по себе гейт не триггерит.
_FRAGMENT_SENTINEL = "\x00"


def _reassemble_fstring(node: ast.JoinedStr) -> str:
    """f-строка → строка-«шаблон»: интерполяции заменены сентинелом.

    Литеральные ``{{``/``}}`` f-строка отдаёт как ``{``/``}`` — после
    сборки чанки ``{{ключ}}`` склеиваются обратно в плейсхолдер, который
    проверяется тем же PLACEHOLDER_RE, что и в шаблонах.
    """

    parts: list[str] = []
    for item in node.values:
        if isinstance(item, ast.Constant):
            parts.append(str(item.value))
        else:  # FormattedValue — данные фрагмента (фраза, числа)
            parts.append(_FRAGMENT_SENTINEL)
    return "".join(parts)


def lint_fragments(source: str | None = None) -> None:
    """Гейт литералов HTML-фрагментов в f-строках этого файла (issue #121).

    Карточки, ниши и списки собираются f-строками в ``build_site.py``, а не
    шаблонами — буквальный текст интерфейса можно было бы протащить мимо
    ``lint_templates``. Линт разбирает собственный исходник через ast и
    прогоняет каждый HTML-фрагмент (строку или f-строку с тегом) через тот
    же ``_lint_html``: критерий приёмки #21 — «строку нельзя добавить в
    обход механизма» — держится проверкой, а не соглашением. Существующие
    проверки не ослабляются: правила те же, что у шаблонов.

    Докстринги (``ast.Expr`` со строковым значением) под гейт не попадают:
    пример HTML в документации — не интерфейсная строка, а падение сборки
    на нём выглядело бы «буквальным текстом» без очевидной причины.
    """

    tree = ast.parse(source if source is not None else Path(__file__).read_text(encoding="utf-8"))
    # литеральные чанки внутри f-строк walk отдаёт отдельными Constant-узлами
    # (обрезанные теги) — их проверяет родительская JoinedStr целиком
    inside_fstring = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for child in node.values
    }
    # докстринги модулей/функций/классов — Constant в позиции Expr-выражения
    docstrings = {
        id(stmt.value)
        for stmt in ast.walk(tree)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.JoinedStr, ast.Constant)):
            continue
        if isinstance(node, ast.JoinedStr):
            value = _reassemble_fstring(node)
        elif (
            id(node) in inside_fstring
            or id(node) in docstrings
            or not isinstance(node.value, str)
        ):
            continue
        else:
            value = node.value
        if re.search(r"<[a-zA-Z]", value):
            _lint_html(value, f"{Path(__file__).name}:{node.lineno}")


def load_artifact(path: Path | str | None) -> dict | None:
    """Артефакт витрины целиком; None — файла нет или фраз в нём нет.

    Отсутствие файла — не ошибка (пустое состояние, #21 п.7): первые
    сборки Pages идут до того, как Dokku-контейнер закоммитит первый
    артефакт в site_data/. Битая схема — ошибка сборки: молча показать
    пустое состояние на испорченном артефакте значило бы публиковать
    витрину «данных нет» поверх существующих данных.
    """

    if path is None:
        return None
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") not in READABLE_SCHEMAS:
        raise BuildError(
            f"{path}: не артефакт витрины (ожидается одна из схем {sorted(READABLE_SCHEMAS)})"
        )
    phrases = data.get("phrases")
    if not isinstance(phrases, list):
        raise BuildError(f"{path}: артефакт без списка phrases")
    return data if phrases else None


def escape_data(value: object) -> str:
    """Данные артефакта → текстовый узел HTML.

    ``html.escape`` гасит разметку, но не трогает фигурные скобки: фраза
    вида ``{{trends.empty.text}}`` иначе попала бы в позицию шаблона и
    подставилась локализованной строкой (или уронила сборку на неизвестном
    ключе) — поэтому ``{{`` нейтрализуется сразу.
    """

    return html.escape(str(value)).replace("{{", "{ {")


def load_phrases_cache(path: Path | str | None) -> TranslationCache:
    """Кэш переводов фраз (issue #120): сборка читает кэш, сеть не трогает.

    Кэш — источник истины (#20): пополнение — отдельный локальный прогон
    ``scripts/translate_phrases.py`` с ключом, сборка сайта воспроизводима
    и без сети. Битый JSON-кэш — ошибка сборки (контракт
    ``TranslationCache.load``), а не молчаливая потеря переводов.
    """

    if path is None:
        return TranslationCache()
    try:
        return TranslationCache.load(Path(path))
    except CacheError as exc:
        raise BuildError(str(exc)) from exc


def phrase_html(phrase: str, code: str, cache: TranslationCache) -> str:
    """Фраза для показа в локали: перевод + оригинал кириллицей (#20, #120).

    zh — ``display_phrase`` («перевод（оригинал）», оригинал обязателен:
    закупщик сверяет фразу по Вордстату); непереведённая фраза — оригинал
    с маркером ``{{trend.phrase.untranslated}}`` локали, не молча. ru —
    оригинал как есть. Возвращается готовый текстовый узел: и перевод,
    и оригинал экранируются здесь.
    """

    item = TranslatedPhrase(phrase=phrase, translation=cache.get(phrase))
    if code == DEFAULT_LOCALE or item.is_translated:
        return escape_data(display_phrase(item, code))
    # Непереведённая фраза не-дефолтной локали: маркер — {{ключ}} локали,
    # фрагмент затем проходит через render, опечатка падает сборкой.
    return f"{escape_data(phrase)} {{{{trend.phrase.untranslated}}}}"


def render_index_sections(artifact: dict, code: str, cache: TranslationCache) -> str:
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
            f"<li>{phrase_html(p['phrase'], code, cache)}</li>" for p in rows
        )
        sections.append(
            f'<section class="trend-section">\n'
            f'<h2>{{{{{key}}}}}</h2>\n<ul class="trend-list">\n{items}\n</ul>\n</section>'
        )
    return "\n".join(sections)


def render_niche_section(artifact: dict, code: str, cache: TranslationCache) -> str:
    """Секция ниш для главной: темы, а не отдельные фразы (эпик #14).

    Ниши приходят из артефакта (посчитаны ядром, docs/TRENDS.md): метка-тема
    и класс — данные (экранируются; класс — ключом локали), топ-фразы состава —
    ссылки на строки таблицы трендов (якоря ``#phrase-N``).
    """

    niches = artifact.get("niches") or []
    if not niches:
        return ""
    items: list[str] = []
    for niche in niches:
        key = SECTION_KEYS.get(str(niche["class"]))
        if key is None:
            raise BuildError(f"неизвестный класс ниши в артефакте: {niche['class']!r}")
        phrases = "".join(
            f'<li><a href="{{{{trends.href}}}}#phrase-{html.escape(str(entry["rank"]), quote=True)}">'
            f"{phrase_html(entry['phrase'], code, cache)}</a></li>"
            for entry in niche["phrases"]
        )
        items.append(
            f'<li class="niche">\n'
            f'<h3 class="niche-topic">{phrase_html(niche["topic"], code, cache)}</h3>\n'
            f'<p class="niche-class">{{{{{key}}}}}</p>\n'
            f'<ul class="niche-phrases">\n{phrases}\n</ul>\n'
            f"</li>"
        )
    return (
        f'<section class="niche-section">\n'
        f"<h2>{{{{niches.title}}}}</h2>\n"
        f'<ul class="niche-list">\n{"".join(items)}\n</ul>\n'
        f"</section>"
    )


def render_trends_body(
    artifact: dict | None,
    messages: dict[str, str],
    code: str,
    cache: TranslationCache,
) -> str:
    """Тело страницы трендов: пустое состояние или карточки витрины (#105)."""

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
        try:
            day = date.fromisoformat(generated_at)
        except ValueError as exc:
            # кривая дата — BuildError, а не сырой ValueError: тот же
            # контракт, что у load_artifact и битого ряда истории
            raise BuildError(
                f"generated_at в артефакте не ISO-датой: {generated_at!r}"
            ) from exc
        data_note = (
            f'<p class="data-date">{{{{trends.generated}}}} '
            f"{format_date(day, code)}</p>"
        )
    cards = "\n".join(
        render_trend_card(p, messages, code, cache) for p in artifact["phrases"]
    )
    return f"{data_note}\n<div class=\"trend-cards\">\n{cards}\n</div>"


def render_trend_card(
    p: dict, messages: dict[str, str], code: str, cache: TranslationCache
) -> str:
    """Карточка тренда: фраза, класс, скор, график истории, объяснение.

    График и объяснение — только в v2-артефакте (есть ``series`` и
    ``ratio``); карточка из v1 собирается без них. ``id="phrase-N"`` —
    якорь, на который ссылаются ниши главной (#106, раньше — строки
    таблицы трендов).
    """

    phrase = phrase_html(p["phrase"], code, cache)
    klass = str(p["class"])
    if klass not in CARD_CLASS_KEYS:
        raise BuildError(f"неизвестный класс фразы в артефакте: {klass!r}")
    rank = escape_data(p["rank"])
    anchor = html.escape(str(p["rank"]), quote=True)
    score = format_score(float(p["score"]), code) if p.get("components") else "—"
    chart = render_series_svg(p.get("series"), p.get("window_months", 0), code)
    explain = render_explanation(p, messages, code)
    return (
        f'<article class="trend-card" id="phrase-{anchor}">\n'
        f'<h2 class="trend-heading"><span class="trend-rank">{{{{trends.col.rank}}}} '
        f"{rank}</span> {phrase}</h2>\n"
        f'<p class="trend-meta"><span class="trend-class">{{{{{CARD_CLASS_KEYS[klass]}}}}}'
        f'</span> · <span class="trend-score">{{{{trends.col.score}}}}: {score}</span></p>\n'
        f"{chart}"
        f"{explain}"
        "</article>"
    )


#: Геометрия графика истории (issue #105): viewBox фиксирован, ширина —
#: 100% через CSS; подписи — данные (числа/месяцы по локале), не текст
#: интерфейса — текст интерфейса здесь только {{trend.chart.caption}}.
_CHART_W, _CHART_H = 600, 200
_CHART_PAD_X, _CHART_PAD_TOP, _CHART_PAD_BOTTOM = 10, 18, 30


def chart_points(values: list[float], width: int = _CHART_W, height: int = _CHART_H) -> list[tuple[float, float]]:
    """Значения ряда → координаты точек SVG: x равномерно по месяцам,
    y — от нуля до максимума ряда (график частот честен базовой линией).

    Отдельная чистая функция: тест проверяет соответствие точек ряду по
    ней, а не разбором готового SVG на глаз.
    """

    n = len(values)
    if n == 0:
        return []
    top = max(float(v) for v in values)
    if top <= 0:
        top = 1.0
    plot_w = width - 2 * _CHART_PAD_X
    plot_h = height - _CHART_PAD_TOP - _CHART_PAD_BOTTOM
    step = plot_w / (n - 1) if n > 1 else 0.0
    return [
        (
            _CHART_PAD_X + i * step,
            _CHART_PAD_TOP + plot_h * (1.0 - float(v) / top),
        )
        for i, v in enumerate(values)
    ]


def _period_date(period: str) -> date:
    year, month = period.split("-")[:2]
    return date(int(year), int(month), 1)


def render_series_svg(
    series: dict | None, window_months: int, code: str
) -> str:
    """Месячный ряд → inline SVG: история + выделенное окно скоринга.

    Статика без JS и внешних библиотек (ограничение Pages-сборки из #21:
    ``uv run --no-project``). Ось Y — от нуля, подписи краёв оси X и
    максимума — форматтеры ``wordstat_trends.i18n`` по локали; цвета —
    только через CSS-классы (светлая/тёмная темы сайта), не атрибуты.
    ``series=None`` (v1-артефакт) — пустая строка, карточка без графика.
    """

    if not series:
        return ""
    # Битый ряд — BuildError, а не сырой KeyError/ValueError: тот же
    # контракт, что у load_artifact (испорченный артефакт — громкий отказ).
    try:
        values = [float(v) for v in series["values"]]
        periods = [str(p) for p in series["periods"]]
        edge_months = [_period_date(p) for p in (periods[0], periods[-1])]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise BuildError(f"ряд истории в артефакте битый: {exc}") from exc
    if len(values) != len(periods) or not values:
        raise BuildError("ряд истории в артефакте битый: длины periods/values не совпадают")
    points = chart_points(values)
    # Окно скоринга — последние window_months точек (GrowthScore.actual).
    window = max(0, min(int(window_months), len(points)))
    split = len(points) - window
    history = " ".join(f"{x:.1f},{y:.1f}" for x, y in points[: split + 1])
    window_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points[split:])
    dots = "".join(
        f'<circle class="chart-dot" cx="{x:.1f}" cy="{y:.1f}" r="3"/>'
        for x, y in points[split:]
    )
    top_value = max(values)
    first_month = format_month(edge_months[0], code)
    last_month = format_month(edge_months[1], code)
    baseline = _CHART_H - _CHART_PAD_BOTTOM
    return (
        f'<figure class="trend-figure">\n'
        f'<figcaption class="visually-hidden">{{{{trend.chart.caption}}}}</figcaption>\n'
        f'<svg class="trend-chart" viewBox="0 0 {_CHART_W} {_CHART_H}" '
        f'role="img" preserveAspectRatio="xMidYMid meet">\n'
        f'<line class="chart-axis" x1="{_CHART_PAD_X}" y1="{baseline}" '
        f'x2="{_CHART_W - _CHART_PAD_X}" y2="{baseline}"/>\n'
        f'<polyline class="chart-history" points="{history}"/>\n'
        f'<polyline class="chart-window" points="{window_line}"/>\n'
        f"{dots}\n"
        f'<text class="chart-label" x="{_CHART_PAD_X}" y="12">{escape_data(format_number(top_value, code))}</text>\n'
        f'<text class="chart-label" x="{_CHART_PAD_X}" y="{_CHART_H - 8}">{escape_data(first_month)}</text>\n'
        f'<text class="chart-label chart-label-end" x="{_CHART_W - _CHART_PAD_X}" '
        f'y="{_CHART_H - 8}">{escape_data(last_month)}</text>\n'
        f"</svg>\n"
        f"</figure>\n"
    )


def render_explanation(p: dict, messages: dict[str, str], code: str) -> str:
    """Объяснение «почему фраза в списке» (#14): шаблон класса + числа.

    Подстановка — тем же контрактом, что ``wordstat_trends.explain``
    (#20): ``{ratio}`` — отношение факт/прогноз, ``{months}`` — возраст
    роста, оба уже отформатированы по локали; лишние параметры
    ``str.format`` игнорирует, недостающий плейсхолдер — ошибка сборки.
    В v1-артефакте нет ``ratio`` — объяснение опускается.
    """

    if "ratio" not in p:
        return ""
    klass = str(p["class"])
    params: dict[str, str] = {"ratio": format_ratio(float(p["ratio"]), code)}
    components = p.get("components")
    if components:
        params["months"] = escape_data(components.get("growth_age_months", ""))
    try:
        text = messages[EXPLAIN_KEYS[klass]].format(**params)
    except KeyError as exc:
        raise BuildError(f"шаблон объяснения {klass!r}: плейсхолдер {exc} не получил значения") from exc
    return (
        f'<p class="trend-explain"><strong>{{{{trend.why}}}}:</strong> '
        f"{escape_data(text)}</p>\n"
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
    out_dir: Path,
    build_date: date | None = None,
    artifact_path: Path | str | None = None,
    phrases_cache_path: Path | str | None = PHRASES_CACHE_PATH,
) -> tuple[list[Path], dict | None]:
    """Собирает страницы обеих локалей и ассеты.

    ``artifact_path`` — JSON-артефакт ранжирования (showcase/v2, читается
    и v1 — карточки без графика и объяснения). ``phrases_cache_path`` — кэш
    переводов фраз (#120): сборка из кэша, без сети и ключа; пополнение —
    отдельный прогон scripts/translate_phrases.py. Возвращает (страницы,
    артефакт): артефакт — загруженный JSON или None при отсутствии/пустоте —
    тогда пустое состояние (сборка не падает, #21 п.7). Вызывающий (main)
    использует его для итогового сообщения, не перечитывая файл.
    """
    locales = load_locales()
    lint_templates()
    lint_fragments()
    artifact = load_artifact(artifact_path)
    cache = load_phrases_cache(phrases_cache_path)
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
            # Страницы одной локали лежат в одном каталоге (корень или /zh/),
            # поэтому навигационные href-ы — «имя.html» без префикса локали:
            # на /zh/*.html «zh/trends.html» резолвился бы в /zh/zh/… (#119).
            hrefs = {f"{other}.href": f"{other}.html" for other in PAGES}
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
                        render_index_sections(artifact, code, cache)
                        + "\n"
                        + render_niche_section(artifact, code, cache),
                        messages,
                        context,
                        f"{code}/index.sections",
                    )
                    if artifact
                    else ""
                )
            elif name == "trends":
                context["trends.body"] = render(
                    render_trends_body(artifact, messages, code, cache),
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
    return written, artifact


def main(argv: list[str]) -> int:
    # argv — как из sys.argv (с именем скрипта), parse_args ждёт аргументы без него
    args = parse_args(argv[1:])
    try:
        pages, artifact = build(
            args.out_dir,
            artifact_path=args.artifact,
            phrases_cache_path=args.phrases_cache,
        )
    except BuildError as exc:
        print(f"ошибка сборки: {exc}", file=sys.stderr)
        return 1
    # источник печатается, только когда артефакт реально загружен (файл есть
    # и фразы в нём есть): дефолтный путь задан всегда, а файла может не быть —
    # витрина тогда пустая, и «из site_data/showcase.json» в логе вводило бы
    # в заблуждение
    source = f" из {args.artifact}" if artifact else ""
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
        default=DEFAULT_ARTIFACT_PATH,
        help=f"JSON-артефакт витрины (по умолчанию {DEFAULT_ARTIFACT_PATH}); "
        "нет файла или фраз — пустое состояние",
    )
    parser.add_argument(
        "--phrases-cache",
        type=Path,
        default=PHRASES_CACHE_PATH,
        help=f"кэш переводов фраз (по умолчанию {PHRASES_CACHE_PATH}); "
        "пополнение — scripts/translate_phrases.py, сборка сеть не трогает",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

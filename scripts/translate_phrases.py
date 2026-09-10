"""Пополнение кэша переводов фраз (issue #20, фаза 5) — локальный прогон.

В CI не входит и сетевых вызовов в тестах не делает: запуск руками,

    GOOGLE_TRANSLATE_API_KEY=... uv run python scripts/translate_phrases.py

Ключ — только в секретах Actions / переменных окружения, не в репозитории.
Без ключа скрипт не падает: показывает, что уже в кэше и что осталось
непереведённым (кэш — источник истины, сервис — пополнение для новых фраз).

Кандидаты на перевод — **артефакт витрины** ``site_data/showcase.json``
(issue #127): ``phrases[].phrase`` и ``niches[].topic`` — ровно те строки,
которые рендер ищет в кэше при сборке. Фикстуры ``tests/fixtures/
dynamics_*.csv`` (фразы из заголовков выгрузки) плюс темы кластеров из
:mod:`wordstat_trends.nlp.clustering` — fallback, когда артефакта ещё нет.

Несовпадение ключей («курсы английского» в витрине против «курсы
английского языка» в кэше, issue #127) решено переводом точных фраз
артефакта, а не нормализацией при поиске: рендер (``build_site.phrase_html``)
смотрит кэш дословно, и фаззи-поиск ради экономии одного запроса к сервису
менял бы семантику подстановки на каждом чтении кэша. Лишний почти-дубль
ключа в кэше дешевле и предсказуемее.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wordstat_trends.i18n_phrases import (  # noqa: E402
    API_KEY_ENV,
    DEFAULT_CACHE_PATH,
    TranslationCache,
    translate_phrases,
)

FIXTURES = ROOT / "tests" / "fixtures"

#: Живой артефакт витрины — первоисточник кандидатов (issue #127).
SHOWCASE_PATH = ROOT / "site_data" / "showcase.json"

#: Заголовок выгрузки динамики Вордстата — фраза в «кавычках-ёлочках».
_PHRASE_RE = re.compile(r"запросов «([^»]+)»")


def showcase_phrases(path: Path = SHOWCASE_PATH) -> list[str]:
    """Кандидаты из живого артефакта витрины: ``phrases[].phrase`` + ``niches[].topic``.

    Это ровно строки, которые ``build_site`` ищет в кэше при рендере zh-страниц,
    поэтому ключи кэша совпадают с запросами рендера дословно. Отсутствующий
    файл — не ошибка (артефакта может ещё не быть — fallback на фикстуры);
    битый JSON — ошибка: молча перевести не те фразы хуже, чем упасть громко.
    """

    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    phrases = [item["phrase"] for item in data.get("phrases", []) if item.get("phrase")]
    phrases += [
        item["topic"] for item in data.get("niches", []) if item.get("topic")
    ]
    return list(dict.fromkeys(phrases))


def fixture_phrases() -> list[str]:
    """Фразы из заголовков фикстур dynamics_*.csv (формат docs/DATA.md)."""

    phrases: list[str] = []
    for path in sorted(FIXTURES.glob("dynamics_*.csv")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        match = _PHRASE_RE.search(lines[0]) if lines else None
        if match:
            phrases.append(match.group(1))
    return list(dict.fromkeys(phrases))


def cluster_topics(phrases: list[str]) -> list[str]:
    """Метку-тема каждого кластера — тоже кандидаты перевода.

    Degenerate случаи (меньше фраз, чем минимальное k сетки) не роняют
    прогон: метки кластеров — дополнение к фразам, а не обязательство.
    """

    try:
        from wordstat_trends.nlp.clustering import cluster_phrases

        result = cluster_phrases(phrases, pipeline="tfidf")
        return [cluster.topic for cluster in result.clusters if cluster.topic]
    except Exception as exc:  # noqa: BLE001 — прогон пополнения не ради кластеров
        print(f"i18n: кластеризация не выполнена ({exc}) — кандидаты только фикстуры")
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE_PATH, help="путь к кэшу переводов"
    )
    parser.add_argument(
        "--showcase",
        type=Path,
        default=SHOWCASE_PATH,
        help="артефакт витрины — первоисточник кандидатов (issue #127)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="не записывать кэш на диск"
    )
    args = parser.parse_args(argv)

    phrases = showcase_phrases(args.showcase)
    source = f"артефакт витрины {args.showcase}"
    if not phrases:
        # Fallback до первого живого сбора: артефакта нет — кандидаты те же,
        # что видны в тестах (фикстуры + темы кластеров).
        phrases = fixture_phrases()
        phrases += [topic for topic in cluster_topics(phrases) if topic not in phrases]
        source = "фикстуры + темы кластеров (артефакт витрины не найден)"
    if not phrases:
        print("i18n: кандидаты на перевод не найдены")
        return 1
    print(f"i18n: источник кандидатов — {source} ({len(phrases)} фраз)")

    api_key = os.environ.get(API_KEY_ENV)
    cache = TranslationCache.load(args.cache)
    translated, report = translate_phrases(phrases, cache, api_key=api_key)

    if api_key and report.translated_now and not args.dry_run:
        cache.save(args.cache)
        print(f"i18n: кэш обновлён → {args.cache} (+{report.translated_now})")
    elif report.untranslated:
        print(
            f"i18n: ключа {API_KEY_ENV} нет (или сервис недоступен) — "
            f"{len(report.untranslated)} непереведённых фраз остаются оригиналом, кэш не менялся"
        )
    print(
        f"i18n: итого {report.total}: из кэша {report.from_cache}, "
        f"переведено сейчас {report.translated_now}, без перевода {len(report.untranslated)}"
    )
    for item in translated:
        mark = item.translation if item.translation is not None else "—"
        print(f"  {item.phrase} → {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

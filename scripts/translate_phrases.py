"""Пополнение кэша переводов фраз (issue #20, фаза 5) — локальный прогон.

В CI не входит и сетевых вызовов в тестах не делает: запуск руками,

    GOOGLE_TRANSLATE_API_KEY=... uv run python scripts/translate_phrases.py

Ключ — только в секретах Actions / переменных окружения, не в репозитории.
Без ключа скрипт не падает: показывает, что уже в кэше и что осталось
непереведённым (кэш — источник истины, сервис — пополнение для новых фраз).

Кандидаты на перевод — фразы фикстур ``tests/fixtures/dynamics_*.csv``
(парсятся из заголовка выгрузки Вордстата «Динамика частотности запросов
«фраза», …») плюс темы кластеров из :mod:`wordstat_trends.nlp.clustering`
(метки-темы витрины тоже показываются китайскому закупщику; конвейер
``tfidf`` — без extra ``nlp``).
"""

from __future__ import annotations

import argparse
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

#: Заголовок выгрузки динамики Вордстата — фраза в «кавычках-ёлочках».
_PHRASE_RE = re.compile(r"запросов «([^»]+)»")


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
        "--dry-run", action="store_true", help="не записывать кэш на диск"
    )
    args = parser.parse_args(argv)

    phrases = fixture_phrases()
    phrases += [topic for topic in cluster_topics(phrases) if topic not in phrases]
    if not phrases:
        print("i18n: кандидаты на перевод не найдены")
        return 1

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

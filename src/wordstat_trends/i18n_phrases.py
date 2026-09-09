"""Перевод поисковых фраз для витрины (issue #20, фаза 5).

Второй пласт текста витрины (первый — интерфейс, он в ``site/locales``):
сами фразы Вордстата («новогодние подарки», «купить телефон»). Их тысячи
и с каждым сбором прибавляются, поэтому переводятся не руками, а внешним
детерминированным сервисом перевода — **на этапе сборки сайта**, не в
ML-пайплайне (граница принципа «без LLM», issue #20: перевод подписи —
не число, не кластер и не тренд; ранжирование и скоринг остаются без
перевода).

Воспроизводимость: **кэш в репозитории — источник истины**,
``data/i18n/phrases.zh.json``; сервис — способ его пополнить для новых
фраз. Сборка без ключа не падает: берёт из кэша и предупреждает о
непереведённых. Ключ живёт в секретах Actions, не в репозитории.

Клиент сервиса — Google Translate v2 через ``urllib`` (stdlib): ни одной
новой зависимости, CI-гейт «без LLM» (``scripts/check_no_llm.py``) на него
не срабатывает — и не должен: это детерминированный переводчик, не
LLM-генерация.

Сетевых вызовов в тестах нет — внешний ``fetch`` подменяется моком;
реальный прогон пополнения — ``scripts/translate_phrases.py``, локально.
"""

from __future__ import annotations

import html
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from wordstat_trends.i18n import Locale, normalize_locale

#: Кэш-источник истины по умолчанию (issue #20: переводы версионируются
#: в репозитории, иначе сайт невоспроизводим, а сборка зависит от сети).
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "i18n" / "phrases.zh.json"
)

#: Имя переменной окружения с ключом сервиса; в CI — секрет Actions.
API_KEY_ENV = "GOOGLE_TRANSLATE_API_KEY"

#: Эндпоинт Google Translate v2 — детерминированный сервис перевода.
_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"

#: Сколько фраз в одном запросе к сервису: лимит q-параметров v2.
_BATCH_SIZE = 64

class CacheError(Exception):
    """Кэш переводов не читается или не парсится."""


class TranslationServiceError(Exception):
    """Сервис перевода недоступен или ответил ошибкой."""


@dataclass(frozen=True)
class TranslatedPhrase:
    """Фраза для витрины: оригинал рядом с переводом (issue #20, п. 4).

    Китайский закупщик не может проверить машинный перевод по Вордстату —
    оригинал кириллицей рядом с переводом позволяет. Поэтому структура
    данных несёт оба поля; рендер (вёрстка #14) показывает их вместе.
    """

    phrase: str  # оригинал кириллицей — как в Вордстате
    translation: str | None  # None — перевода нет (нет в кэше, сервиса не было)

    @property
    def is_translated(self) -> bool:
        return self.translation is not None


@dataclass(frozen=True)
class TranslationReport:
    """Итог пополнения переводов — для предупреждений сборки."""

    total: int
    from_cache: int
    translated_now: int
    untranslated: list[str]  # фразы без перевода — их покажет оригиналом


class TranslationCache:
    """Кэш переводов ``фраза → перевод`` поверх JSON в репозитории.

    Запись детерминирована: сортировка по ключу, ``ensure_ascii=False``,
    финальный перевод строки — diffs между запусками минимальны.
    """

    def __init__(self, entries: dict[str, str] | None = None):
        self._entries: dict[str, str] = dict(entries or {})

    @classmethod
    def load(cls, path: Path) -> TranslationCache:
        """Прочитать кэш; пустой/отсутствующий файл — пустой кэш, а не ошибка.

        Сборка сайта не должна падать из-за отсутствия кэша (нет переводов —
        показываем оригиналы), но битый JSON — ошибка: молча выбросить
        половину переводов хуже, чем упасть громко.
        """

        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CacheError(f"кэш {path} не является валидным JSON: {exc}") from exc
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise CacheError(f"кэш {path}: ожидается отображение строка → строка")
        return cls(data)

    def get(self, phrase: str) -> str | None:
        return self._entries.get(phrase)

    def update(self, pairs: dict[str, str]) -> None:
        self._entries.update(pairs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            dict(sorted(self._entries.items())), ensure_ascii=False, indent=2
        )
        path.write_text(body + "\n", encoding="utf-8")

    def __len__(self) -> int:
        return len(self._entries)


def google_translate_fetch(
    phrases: list[str], api_key: str, source: str = "ru", target: str = "zh"
) -> list[str]:
    """Пакет фраз → переводы через Google Translate v2 (детерминированный сервис).

    Отдельная функция, а не метод: инъецируется в ``translate_phrases``
    моком в тестах — сетевых вызовов в тестах нет.
    """

    params = urllib.parse.urlencode(
        [("q", p) for p in phrases] + [("source", source), ("target", target), ("format", "text")]
    )
    # Ключ — заголовком, не query string: URL с ключом оседает в логах
    # прокси и раннеров (заметка ревью PR #108).
    request = urllib.request.Request(
        f"{_ENDPOINT}?{params}", headers={"x-goog-api-key": api_key}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TranslationServiceError(f"сервис перевода недоступен: {exc}") from exc
    try:
        translated = [item["translatedText"] for item in payload["data"]["translations"]]
    except (KeyError, TypeError) as exc:
        raise TranslationServiceError(f"неожиданный ответ сервиса перевода: {payload!r}") from exc
    # format=text всё равно возвращает HTML-escaped амперсанды — снимаем.
    return [html.unescape(t) for t in translated]


def _default_fetch(phrases: list[str], api_key: str) -> list[str]:
    return google_translate_fetch(phrases, api_key)


def translate_phrases(
    phrases: list[str],
    cache: TranslationCache,
    api_key: str | None = None,
    fetch: Callable[[list[str], str], list[str]] | None = None,
) -> tuple[list[TranslatedPhrase], TranslationReport]:
    """Список фраз → переводы: кэш — источник истины, сервис — пополнение.

    Порядок работы (issue #20): уже переведённое — только из кэша (сборка
    не дёргает сеть ради того, что уже зафиксировано); новые фразы —
    сервисом, если есть ключ, с записью в кэш; без ключа сборка не
    падает — непереведённые фразы возвращаются с ``translation=None`` и
    попадают в предупреждение. Ошибка сервиса не роняет сборку по той же
    причине: сеть — способ пополнения, а не обязательство.
    """

    fetch = fetch or _default_fetch
    unique = list(dict.fromkeys(phrases))

    results: dict[str, str] = {}
    missing: list[str] = []
    for phrase in unique:
        cached = cache.get(phrase)
        if cached is not None:
            results[phrase] = cached
        else:
            missing.append(phrase)

    translated_now = 0
    if missing and api_key:
        for start in range(0, len(missing), _BATCH_SIZE):
            batch = missing[start : start + _BATCH_SIZE]
            try:
                batch_translated = fetch(batch, api_key)
                if len(batch_translated) != len(batch):
                    raise TranslationServiceError(
                        f"сервис вернул {len(batch_translated)} переводов на {len(batch)} фраз"
                    )
            except TranslationServiceError as exc:
                # Любая ошибка сервиса — не фатальна: сеть способ пополнения,
                # а не обязательство (контракт issue #20, заметка ревью #108).
                print(f"i18n: {exc} — сборка продолжает на кэше", file=sys.stderr)
                break
            cache.update(dict(zip(batch, batch_translated)))
            results.update(dict(zip(batch, batch_translated)))
            translated_now += len(batch_translated)

    untranslated = [p for p in unique if p not in results]
    if untranslated:
        preview = ", ".join(untranslated[:5])
        more = "" if len(untranslated) <= 5 else f" … (ещё {len(untranslated) - 5})"
        print(
            f"i18n: без перевода {len(untranslated)} фраз: {preview}{more} — "
            f"пополните кэш прогоном scripts/translate_phrases.py с ключом",
            file=sys.stderr,
        )

    translated = [TranslatedPhrase(phrase=p, translation=results.get(p)) for p in unique]
    report = TranslationReport(
        total=len(unique),
        from_cache=len(unique) - len(missing),
        translated_now=translated_now,
        untranslated=untranslated,
    )
    return translated, report


def display_phrase(item: TranslatedPhrase, locale: str) -> str:
    """Строка показа «перевод + оригинал рядом» для локали.

    zh: «新年礼物（новогодние подарки）» — оригинал в скобках, чтобы
    закупщик мог сверить фразу по Вордстату (issue #20, п. 4).
    ru: оригинал как есть — перевод не нужен. Непереведённая фраза в
    любой локали показывается оригиналом.
    """

    if normalize_locale(str(locale)) is not Locale.ZH or item.translation is None:
        return item.phrase
    return f"{item.translation}（{item.phrase}）"


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_CACHE_PATH",
    "CacheError",
    "TranslationCache",
    "TranslationReport",
    "TranslationServiceError",
    "TranslatedPhrase",
    "display_phrase",
    "google_translate_fetch",
    "translate_phrases",
]

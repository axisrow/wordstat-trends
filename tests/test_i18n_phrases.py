"""Тесты движка перевода фраз (issue #20, фаза 5).

Сетевых вызовов нет: внешний ``fetch`` подменяется моком. Проверяются
главные контракты issue: кэш — источник истины (переведённое не ходит
в сервис), без ключа сборка не падает (непереведённые — оригиналом,
с предупреждением), ошибка сервиса не роняет сборку, оригинал хранится
рядом с переводом.
"""

from __future__ import annotations

import json

import pytest

from wordstat_trends.i18n_phrases import (
    CacheError,
    TranslatedPhrase,
    TranslationCache,
    TranslationServiceError,
    display_phrase,
    translate_phrases,
)


def test_cache_roundtrip_sorted_and_unicode(tmp_path):
    cache = TranslationCache({"новогодние подарки": "新年礼物", "купить телефон": "买手机"})
    path = tmp_path / "phrases.zh.json"
    cache.save(path)
    body = path.read_text(encoding="utf-8")
    # Сортировка и читаемые CJK — детерминированные диффы между прогонами.
    assert list(json.loads(body)) == sorted(json.loads(body))
    assert "新年礼物" in body  # ensure_ascii=False
    assert TranslationCache.load(path).get("купить телефон") == "买手机"


def test_cache_missing_file_is_empty_not_error(tmp_path):
    assert len(TranslationCache.load(tmp_path / "нет.json")) == 0


def test_cache_broken_json_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{не json", encoding="utf-8")
    with pytest.raises(CacheError):
        TranslationCache.load(path)


def test_translated_phrases_come_from_cache_without_service(capsys):
    cache = TranslationCache({"новогодние подарки": "新年礼物"})

    def no_network(phrases, key):  # pragma: no cover - тест обязан упасть при вызове
        raise AssertionError("кэш — источник истины: сервис не вызывается")

    translated, report = translate_phrases(
        ["новогодние подарки"], cache, api_key="ключ", fetch=no_network
    )
    assert translated == [TranslatedPhrase("новогодние подарки", "新年礼物")]
    assert (report.from_cache, report.translated_now, report.untranslated) == (1, 0, [])
    assert capsys.readouterr().err == ""


def test_without_key_build_does_not_fail_and_warns(capsys):
    cache = TranslationCache({"новогодние подарки": "新年礼物"})
    translated, report = translate_phrases(
        ["новогодние подарки", "купить телефон"], cache, api_key=None
    )
    assert translated[1].translation is None
    assert translated[1].is_translated is False
    assert report.untranslated == ["купить телефон"]
    err = capsys.readouterr().err
    assert "без перевода 1" in err and "купить телефон" in err


def test_service_fills_missing_and_updates_cache():
    cache = TranslationCache({"новогодние подарки": "新年礼物"})
    calls: list[list[str]] = []

    def fake_fetch(phrases, key):
        calls.append(list(phrases))
        return [f"译:{p}" for p in phrases]

    translated, report = translate_phrases(
        ["новогодние подарки", "купить телефон"], cache, api_key="k", fetch=fake_fetch
    )
    assert calls == [["купить телефон"]]  # из кэша сервис не дёргается
    assert translated[1].translation == "译:купить телефон"
    assert report.translated_now == 1
    assert cache.get("купить телефон") == "译:купить телефон"  # кэш пополнен


def test_service_error_is_not_fatal(capsys):
    cache = TranslationCache()

    def broken_fetch(phrases, key):
        raise TranslationServiceError("503")

    translated, report = translate_phrases(
        ["купить телефон"], cache, api_key="k", fetch=broken_fetch
    )
    assert translated[0].translation is None
    assert report.untranslated == ["купить телефон"]
    assert "сборка продолжает на кэше" in capsys.readouterr().err


def test_duplicate_phrases_deduplicated():
    cache = TranslationCache()
    translated, report = translate_phrases(
        ["фраза", "фраза"], cache, api_key=None
    )
    assert len(translated) == 1
    assert report.total == 1


def test_display_phrase_shows_original_next_to_translation():
    translated = TranslatedPhrase("новогодние подарки", "新年礼物")
    assert display_phrase(translated, "zh") == "新年礼物（новогодние подарки）"
    assert display_phrase(translated, "ru") == "новогодние подарки"
    assert display_phrase(TranslatedPhrase("фраза", None), "zh") == "фраза"

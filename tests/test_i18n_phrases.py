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


def test_length_mismatch_is_not_fatal(capsys):
    # Ревью PR #108: несовпадение числа переводов — тоже ошибка сервиса,
    # контракт «сборка не падает» распространяется и на неё.
    cache = TranslationCache()

    def short_fetch(phrases, key):
        return ["один перевод"]

    translated, report = translate_phrases(
        ["фраза 1", "фраза 2"], cache, api_key="k", fetch=short_fetch
    )
    assert report.untranslated == ["фраза 1", "фраза 2"]
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


def test_display_phrase_normalizes_locale_tag():
    # Ревью PR #108: «zh-CN» — та же локаль zh, перевод не теряется.
    translated = TranslatedPhrase("новогодние подарки", "新年礼物")
    assert display_phrase(translated, "zh-CN") == "新年礼物（новогодние подарки）"


class TestShowcaseCandidates:
    """Кандидаты пополнения — живой артефакт витрины (issue #127).

    Ключ кэша должен совпадать с фразой рендера дословно: строится из тех же
    полей ``phrases[].phrase`` + ``niches[].topic``, что читает ``build_site``.
    """

    def test_phrases_and_niche_topics_are_candidates(self, tmp_path):
        from scripts.translate_phrases import showcase_phrases

        artifact = {
            "phrases": [
                {"phrase": "курсы английского"},
                {"phrase": "купить телефон"},
            ],
            "niches": [{"topic": "английский, курс"}, {"topic": "телефон, купить"}],
        }
        path = tmp_path / "showcase.json"
        path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
        assert showcase_phrases(path) == [
            "курсы английского",
            "купить телефон",
            "английский, курс",
            "телефон, купить",
        ]

    def test_missing_artifact_is_empty_for_fallback(self, tmp_path):
        from scripts.translate_phrases import showcase_phrases

        assert showcase_phrases(tmp_path / "нет.json") == []

    def test_broken_artifact_json_raises(self, tmp_path):
        # Битый артефакт — громкая ошибка: молча перевести не те фразы хуже.
        from scripts.translate_phrases import showcase_phrases

        path = tmp_path / "broken.json"
        path.write_text("{не json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            showcase_phrases(path)

    def test_main_takes_candidates_from_artifact_not_fixtures(self, tmp_path, capsys):
        # Живой сбор: фразы витрины («подарки на новый год») не входят в
        # фикстуры, но обязаны попасть в кандидатский список отчёта.
        from scripts.translate_phrases import main

        artifact = {
            "phrases": [{"phrase": "курсы английского"}, {"phrase": "подарки на новый год"}],
            "niches": [{"topic": "английский, курс"}],
        }
        showcase = tmp_path / "showcase.json"
        showcase.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(
            json.dumps({"курсы английского": "英语课程"}, ensure_ascii=False),
            encoding="utf-8",
        )

        assert main(["--showcase", str(showcase), "--cache", str(cache_path)]) == 0
        out = capsys.readouterr().out
        assert "артефакт витрины" in out
        assert "подарки на новый год" in out  # фраза живого сбора — в отчёте
        assert "английский, курс" in out  # метка кластера витрины — тоже
        # Без ключа кэш не меняется.
        assert json.loads(cache_path.read_text(encoding="utf-8")) == {
            "курсы английского": "英语课程"
        }

    def test_empty_artifact_warns_and_falls_back(self, tmp_path, capsys):
        # Артефакт есть, но пуст — проблема сбора не маскируется молчаливым
        # fallback (заметка ревью PR #130).
        from scripts.translate_phrases import main

        showcase = tmp_path / "showcase.json"
        showcase.write_text(
            json.dumps({"phrases": [], "niches": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("{}", encoding="utf-8")

        assert main(["--showcase", str(showcase), "--cache", str(cache_path)]) == 0
        out = capsys.readouterr().out
        assert "существует, но не содержит фраз" in out
        assert "fallback — фикстуры" in out

"""Тесты каркаса витрины (issue #21).

Главный критерий приёмки: строку интерфейса нельзя добавить в обход
механизма локализации. Здесь это проверяется напрямую: в шаблон с
буквальным текстом сборка обязана упасть, чистые шаблоны — собраться.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from scripts import build_site
from wordstat_trends.i18n import format_date, format_month, format_number, format_share

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def site_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("site")
    build_site.build(root, build_date=date(2026, 9, 9))
    return root


@pytest.fixture(scope="module")
def pages(site_root) -> dict[str, str]:
    return {
        str(p.relative_to(site_root)): p.read_text(encoding="utf-8")
        for p in sorted(site_root.rglob("*.html"))
    }


# --- механизм i18n: гейт против буквального текста -------------------------


def test_lint_rejects_literal_text_in_template(tmp_path, monkeypatch):
    bad = tmp_path / "evil.html"
    bad.write_text("<p>Привет, мир</p>", encoding="utf-8")
    monkeypatch.setattr(build_site, "TEMPLATES_DIR", tmp_path)
    with pytest.raises(build_site.BuildError, match="вне механизма i18n"):
        build_site.lint_templates()


def test_lint_rejects_literal_text_in_attribute(tmp_path, monkeypatch):
    bad = tmp_path / "evil.html"
    bad.write_text('<a title="Ссылка">{{nav.index}}</a>', encoding="utf-8")
    monkeypatch.setattr(build_site, "TEMPLATES_DIR", tmp_path)
    with pytest.raises(build_site.BuildError, match="в атрибуте title"):
        build_site.lint_templates()


def test_lint_rejects_literal_text_in_arbitrary_attribute(tmp_path, monkeypatch):
    # data-* и прочие атрибуты вне allow-list — тоже гейт, не только title/alt.
    bad = tmp_path / "evil.html"
    bad.write_text('<span data-label="Подпись">{{nav.index}}</span>', encoding="utf-8")
    monkeypatch.setattr(build_site, "TEMPLATES_DIR", tmp_path)
    with pytest.raises(build_site.BuildError, match="в атрибуте data-label"):
        build_site.lint_templates()


def test_unknown_key_fails_build():
    unknown = "{{nav.nowhere}}"
    with pytest.raises(build_site.BuildError, match="неизвестный ключ"):
        build_site.render(unknown, {"nav.index": "x"}, {}, "test")


def test_locales_must_have_identical_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(build_site, "LOCALES_DIR", tmp_path)
    (tmp_path / "ru.json").write_text('{"a": "1"}', encoding="utf-8")
    (tmp_path / "zh.json").write_text('{"a": "1", "b": "2"}', encoding="utf-8")
    with pytest.raises(build_site.BuildError, match="лишние ключи"):
        build_site.load_locales()


# --- структура страниц и человекочитаемые URL ------------------------------


def test_pages_exist_for_both_locales(pages):
    assert set(pages) == {
        "index.html", "trends.html", "about.html",
        "zh/index.html", "zh/trends.html", "zh/about.html",
    }


def test_ru_pages_render_russian(pages):
    assert "Тренды Вордстата" in pages["index.html"]
    assert 'lang="ru"' in pages["index.html"]


def test_zh_pages_render_chinese_and_urls(pages):
    assert "搜索需求趋势" in pages["zh/trends.html"]
    assert 'lang="zh"' in pages["zh/trends.html"]


def test_language_switch_points_to_same_page_in_other_locale(pages):
    assert 'href="zh/index.html"' in pages["index.html"]
    assert 'href="../index.html"' in pages["zh/index.html"]
    assert 'href="../trends.html"' in pages["zh/trends.html"]


def test_zh_assets_use_relative_prefix(pages):
    assert 'href="../assets/style.css"' in pages["zh/index.html"]
    assert 'href="assets/style.css"' in pages["index.html"]


def test_trends_empty_state_not_crash(pages):
    assert "Данных пока нет" in pages["trends.html"]
    assert "暂无数据" in pages["zh/trends.html"]


def test_footer_date_localized(pages):
    assert "9 сентября 2026 г." in pages["index.html"]
    assert "2026年9月9日" in pages["zh/index.html"]


# --- форматы чисел и дат на реальных значениях из tests/fixtures -----------


def _fixture_rows(name: str) -> list[dict[str, Any]]:
    with (FIXTURES / name).open(encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    data = rows[1:]
    periods = [r[0] for r in data if r and r[0]]
    queries = [int(r[1].replace(" ", "").replace("\xa0", "")) for r in data if len(r) > 1 and r[1]]
    shares = [float(r[2].replace(",", ".")) for r in data if len(r) > 2 and r[2]]
    return [{"period": p, "queries": q, "share": s} for p, q, s in zip(periods, queries, shares)][:3]


@pytest.mark.parametrize("fixture", sorted(p.name for p in FIXTURES.glob("dynamics_*.csv")))
def test_number_format_on_fixture_values(fixture):
    for row in _fixture_rows(fixture):
        ru = format_number(row["queries"], "ru")
        zh = format_number(row["queries"], "zh")
        assert int(ru.replace(" ", "").replace(",", "")) == row["queries"]
        assert int(zh.replace(",", "")) == row["queries"]
        assert "," not in ru  # ru: разделитель — узкий неразрывный пробел
        assert "," in zh or row["queries"] < 1000


@pytest.mark.parametrize("fixture", sorted(p.name for p in FIXTURES.glob("dynamics_*.csv")))
def test_share_format_on_fixture_values(fixture):
    for row in _fixture_rows(fixture):
        # ru: узкий неразрывный пробел перед %, десятичная запятая (как в Вордстате).
        assert format_share(row["share"], "ru").endswith("\u202f%")
        assert format_share(row["share"], "zh").endswith("%")
        assert "," in format_share(row["share"], "ru")
        assert "." in format_share(row["share"], "zh")


def test_month_and_date_formats():
    d = date(2024, 8, 1)
    assert format_month(d, "ru") == "август 2024"
    assert format_month(d, "zh") == "2024年8月"
    assert format_date(date(2026, 9, 9), "ru") == "9 сентября 2026 г."
    assert format_date(date(2026, 9, 9), "zh") == "2026年9月9日"


def test_number_format_matches_wordstat_style():
    # В выгрузке Вордстата 997 977 идёт с неразрывным пробелом — ru сохраняет стиль.
    assert format_number(997977, "ru") == "997 977"
    assert format_number(997977, "zh") == "997,977"


# --- данные витрины из артефакта (issue #104) ------------------------------


def _artifact_file(tmp_path) -> Path:
    from tests.test_showcase import _records
    from wordstat_trends.showcase import save_showcase

    return save_showcase(_records(), tmp_path / "showcase.json", generated_at=date(2026, 9, 9))


@pytest.fixture(scope="module")
def filled_site(tmp_path_factory) -> dict[str, str]:
    artifact = _artifact_file(tmp_path_factory.mktemp("artifact"))
    root = tmp_path_factory.mktemp("site-filled")
    build_site.build(root, build_date=date(2026, 9, 9), artifact_path=artifact)
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.html"))
    }


def test_index_shows_trends_immediately(filled_site):
    # эпик #1: готовые тренды на первом экране, не пустота вокруг hero
    assert "Растущие запросы" in filled_site["index.html"]
    assert "增长的查询" in filled_site["zh/index.html"]
    assert "синтетический рост" in filled_site["index.html"]


def test_index_sections_in_class_order(filled_site):
    # CLASS_ORDER: рост раньше сезонного, сезонное раньше остального
    page = filled_site["index.html"]
    assert page.index("Растущие запросы") < page.index("Сезонное")


def test_trends_cards_render_artifact(filled_site):
    page = filled_site["trends.html"]
    assert "новогодние подарки" in page
    assert "Скор" in page
    assert "Данные от" in page and "9 сентября 2026 г." in page
    zh = filled_site["zh/trends.html"]
    assert "数据截至" in zh and "2026年9月9日" in zh


def test_trends_card_score_localized(filled_site):
    # ru: десятичная запятая; zh: точка (i18n-формат чисел, #21 п.6)
    assert "," in filled_site["trends.html"].split("Скор", 1)[1][:400]
    zh_tail = filled_site["zh/trends.html"].split("评分", 1)[1][:400]
    assert re.search(r"\d\.\d", zh_tail)


def test_trends_non_growing_score_is_dash(filled_site):
    # вне «растёт» скора нет — тире, а не 0,00 (объяснимость, #85)
    assert "Скор: —" in filled_site["trends.html"]


def test_phrase_html_escaped(tmp_path):
    # фраза — данные из Вордстата: разрыв разметки через < недопустим
    evil = tmp_path / "evil.json"
    entry = {
        "schema": build_site.SHOWCASE_SCHEMA,
        "generated_at": "2026-09-09",
        "phrases": [
            {"phrase": "<script>alert(1)</script>", "class": "GROWING",
             "rank": 1, "score": 0.5, "components": None}
        ],
    }
    evil.write_text(json.dumps(entry), encoding="utf-8")
    root = tmp_path / "site"
    build_site.build(root, artifact_path=evil)
    page = (root / "trends.html").read_text(encoding="utf-8")
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


# --- пересборка витрины из артефакта в репо (issue #107) --------------------


def test_default_artifact_path_in_tracked_site_data():
    # путь артефакта зафиксирован в репо: site_data/showcase.json; каталог
    # не должен попадать под паттерн .gitignore — иначе витрина навсегда
    # останется в пустом состоянии
    assert build_site.DEFAULT_ARTIFACT_PATH.parent.name == "site_data"
    assert build_site.DEFAULT_ARTIFACT_PATH.name == "showcase.json"
    gitignore = (Path(build_site.SITE_DIR).parent / ".gitignore").read_text(encoding="utf-8")
    for line in gitignore.splitlines():
        pattern = line.split("#", 1)[0].strip().rstrip("/")
        assert pattern != "site_data", ".gitignore исключает каталог артефактов"


def test_missing_default_artifact_builds_empty_state(tmp_path, monkeypatch):
    # до первого коммита результатов файлом site_data/showcase.json нет —
    # Pages-сборка не падает, витрина в пустом состоянии (#21 п.7)
    monkeypatch.setattr(build_site, "DEFAULT_ARTIFACT_PATH", tmp_path / "missing.json")
    out = tmp_path / "site"
    assert build_site.main(["build_site.py", str(out)]) == 0
    assert "Данных пока нет" in (out / "trends.html").read_text(encoding="utf-8")


def test_default_artifact_used_when_present(tmp_path, monkeypatch):
    # артефакт по умолчанию читается без явного --artifact: дата данных —
    # generated_at артефакта через i18n-форматтер, не дата сборки
    artifact = _artifact_file(tmp_path)
    monkeypatch.setattr(build_site, "DEFAULT_ARTIFACT_PATH", artifact)
    out = tmp_path / "site"
    assert build_site.main(["build_site.py", str(out)]) == 0
    page = (out / "trends.html").read_text(encoding="utf-8")
    assert "синтетический рост" in page
    assert "Данные от 9 сентября 2026 г." in page


def test_placeholder_braces_in_data_neutralized(tmp_path):
    # фраза-данные вида {{ключ}} не должна занимать позицию шаблона:
    # иначе подставится локализованная строка или сборка упадёт на
    # неизвестном ключе
    evil = tmp_path / "braces.json"
    entry = {
        "schema": build_site.SHOWCASE_SCHEMA,
        "generated_at": None,
        "phrases": [
            {"phrase": "{{trends.empty.text}}", "class": "GROWING",
             "rank": 1, "score": 0.5, "components": None}
        ],
    }
    evil.write_text(json.dumps(entry), encoding="utf-8")
    root = tmp_path / "site"
    build_site.build(root, artifact_path=evil)
    page = (root / "trends.html").read_text(encoding="utf-8")
    assert "Витрина в разработке" not in page
    assert "{ {trends.empty.text}" in page


def test_string_rank_from_corrupted_artifact_does_not_break_markup(tmp_path):
    # load_artifact типы не валидирует — строковый rank экранируется,
    # как фраза и класс
    bad = tmp_path / "rank.json"
    entry = {
        "schema": build_site.SHOWCASE_SCHEMA,
        "generated_at": None,
        "phrases": [
            {"phrase": "x", "class": "GROWING",
             "rank": "<b>1</b>", "score": 0.5, "components": None}
        ],
    }
    bad.write_text(json.dumps(entry), encoding="utf-8")
    root = tmp_path / "site"
    build_site.build(root, artifact_path=bad)
    page = (root / "trends.html").read_text(encoding="utf-8")
    assert "<b>1</b>" not in page
    assert "&lt;b&gt;1&lt;/b&gt;" in page


def test_bad_artifact_schema_fails_build(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope", "phrases": []}), encoding="utf-8")
    with pytest.raises(build_site.BuildError, match="не артефакт"):
        build_site.build(tmp_path / "site", artifact_path=bad)


def test_missing_artifact_keeps_empty_state(site_root):
    # без --artifact сборка не падает и показывает пустое состояние (#21 п.7)
    page = (site_root / "trends.html").read_text(encoding="utf-8")
    assert "Данных пока нет" in page

"""Тесты каркаса витрины (issue #21).

Главный критерий приёмки: строку интерфейса нельзя добавить в обход
механизма локализации. Здесь это проверяется напрямую: в шаблон с
буквальным текстом сборка обязана упасть, чистые шаблоны — собраться.
"""

from __future__ import annotations

import csv
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

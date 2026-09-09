"""Тесты карточек трендов витрины (issue #105, фаза 5).

Конвейер по issue: фикстуры ``dynamics_*.csv`` → ``phrase_record_from_frame``
→ сериализация (showcase/v2 с рядом) → сборка сайта. Проверяется карточка
(фраза, класс, скор), объяснение «почему в списке» с числами по локали и
SVG-график истории: парсится XML и точки соответствуют ряду.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_site
from tests.test_showcase import _growing_record
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.showcase import save_showcase
from wordstat_trends.trends.ranking import phrase_record_from_frame

FIXTURES = Path(__file__).parent / "fixtures"


def _record(csv_name: str, phrase: str):
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return phrase_record_from_frame(frame, phrase=phrase)


def _records() -> list:
    return [
        _growing_record(),
        _record("dynamics_seasonal.csv", "новогодние подарки"),
        _record("dynamics_high_freq.csv", "купить телефон"),
    ]


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("artifact") / "showcase.json"
    return save_showcase(_records(), target, generated_at=date(2026, 9, 9))


@pytest.fixture(scope="module")
def site(tmp_path_factory, artifact_path) -> dict[str, str]:
    root = tmp_path_factory.mktemp("site-trends")
    build_site.build(root, build_date=date(2026, 9, 9), artifact_path=artifact_path)
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.html"))
    }


def _artifact(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- карточка: фраза, класс, скор --------------------------------------------


def test_card_shows_phrase_class_and_score(site):
    page = site["trends.html"]
    assert "синтетический рост" in page
    assert 'class="trend-card"' in page
    assert "растёт" in page  # метка класса в карточке (trend.class.growing)
    assert "Скор: " in page
    assert re.search(r"Скор: \d+,\d\d", page)  # ru: десятичная запятая, 2 знака (format_score)
    assert "№ 1" in page  # ранг — ключ trends.col.rank + число


def test_card_count_matches_artifact(site, artifact_path):
    page = site["trends.html"]
    assert page.count('class="trend-card"') == len(_artifact(artifact_path)["phrases"])


def test_zh_card_renders_chinese(site):
    zh = site["zh/trends.html"]
    assert "增长" in zh  # trend.class.growing
    assert "为何上榜" in zh  # trend.why
    assert 'lang="zh"' in zh


# --- объяснение «почему в списке» --------------------------------------------


def test_growing_explanation_substitutes_numbers(site):
    page = site["trends.html"]
    # шаблон ru: «рост к прогнозу в {ratio} раза, рост идёт {months} мес.»
    assert "Почему в списке" in page
    m = re.search(r"рост к прогнозу в (\d,\d) раза, рост идёт (\d+) мес.", page)
    assert m, "объяснение роста с подставленными числами не найдено"
    assert m.group(1) == "2,0"  # ratio синтетического роста (удвоение окна)


def test_falling_or_seasonal_explanation_present(site):
    page = site["trends.html"]
    # сезонная/стабильная фикстуры идут без {months}: шаблон без плейсхолдеров
    assert "сезон" in page or "стабильный спрос" in page


def test_zh_explanation_localized(site):
    zh = site["zh/trends.html"]
    m = re.search(r"较预测增长(\d\.\d)倍", zh)
    assert m, "zh-объяснение роста не найдено"
    assert m.group(1) == "2.0"


# --- SVG-график истории -------------------------------------------------------


def _svg_element(page: str, index: int) -> ET.Element:
    matches = re.findall(r"<svg\b.*?</svg>", page, flags=re.DOTALL)
    assert matches, "inline SVG не найден на странице"
    return ET.fromstring(matches[index])


def test_svg_parses_and_point_count_matches_series(site, artifact_path):
    page = site["trends.html"]
    for i, entry in enumerate(_artifact(artifact_path)["phrases"]):
        svg = _svg_element(page, i)
        # SVG-неймспейс сохранён при fromstring корневого <svg>
        history = svg.find(".//{*}polyline[@class='chart-history']")
        window = svg.find(".//{*}polyline[@class='chart-window']")
        assert history is not None and window is not None
        n = len(entry["series"]["values"])
        w = entry["window_months"]
        assert len(history.attrib["points"].split()) == n - w + 1
        assert len(window.attrib["points"].split()) == w


def test_svg_points_correspond_to_series(site, artifact_path):
    page = site["trends.html"]
    for i, entry in enumerate(_artifact(artifact_path)["phrases"]):
        svg = _svg_element(page, i)
        expected = build_site.chart_points([float(v) for v in entry["series"]["values"]])
        lines = svg.findall(".//{*}polyline")
        assert len(lines) == 2
        got = [
            tuple(float(c) for c in pair.split(","))
            for line in lines
            for pair in line.attrib["points"].split()
        ]
        # склейка history+window дублирует стыковую точку — она и должна совпасть;
        # атрибут points округлён до одного знака
        assert got[0] == pytest.approx(expected[0], abs=0.05)
        assert got[-1] == pytest.approx(expected[-1], abs=0.05)
        # монотонность по x: месяцы идут в порядке ряда
        xs = [x for x, _ in got]
        assert xs == sorted(xs)


def test_chart_points_scale_from_zero_to_max():
    # ось Y — от нуля до максимума: максимум наверху (минимальный y), ноль ниже
    values = [100.0, 200.0, 300.0]
    points = build_site.chart_points(values)
    highest = min(y for _, y in points)
    lowest = max(y for _, y in points)
    assert points[2][1] == pytest.approx(highest)  # максимум ряда — последняя точка
    assert points[0][1] == pytest.approx(lowest)  # ноль шкалы (100/300 от максимума)
    assert lowest > highest
    assert all(build_site._CHART_PAD_X <= x <= build_site._CHART_W - build_site._CHART_PAD_X for x, _ in points)


def test_svg_month_labels_localized(site):
    ru = site["trends.html"]
    zh = site["zh/trends.html"]
    assert "январь 2024" in ru or re.search(r"[а-я]+ 20\d\d", ru)
    assert "2024年1月" in zh or re.search(r"20\d\d年\d+月", zh)


def test_zh_svg_same_points(site):
    # график одинаков в обеих локалях (переводятся только подписи)
    def points(page: str) -> list[str]:
        return re.findall(r"<polyline[^>]*points=\"([^\"]+)\"", page)

    assert points(site["trends.html"]) == points(site["zh/trends.html"])


# --- битый ряд в артефакте ------------------------------------------------------


def test_broken_series_raises_build_error():
    # отсутствие ключей / кривые значения / кривой период — BuildError,
    # а не сырой KeyError/ValueError (контракт load_artifact)
    with pytest.raises(build_site.BuildError, match="ряд истории"):
        build_site.render_series_svg({"periods": ["2024-01"]}, 3, "ru")  # нет values
    with pytest.raises(build_site.BuildError, match="ряд истории"):
        build_site.render_series_svg({"periods": ["2024-01"], "values": ["x"]}, 3, "ru")
    with pytest.raises(build_site.BuildError, match="ряд истории"):
        build_site.render_series_svg({"periods": ["январь"], "values": [1.0]}, 3, "ru")


# --- обратная совместимость v1 -------------------------------------------------


def test_v1_artifact_builds_cards_without_chart_and_explanation(tmp_path):
    v1 = tmp_path / "v1.json"
    v1.write_text(
        json.dumps({
            "schema": "showcase/v1",
            "generated_at": "2026-09-09",
            "phrases": [
                {"phrase": "старая фраза", "class": "GROWING", "rank": 1,
                 "score": 0.5, "components": None}
            ],
        }),
        encoding="utf-8",
    )
    root = tmp_path / "site"
    build_site.build(root, artifact_path=v1)
    page = (root / "trends.html").read_text(encoding="utf-8")
    assert "старая фраза" in page
    assert 'class="trend-card"' in page
    assert "<svg" not in page
    assert "Почему в списке" not in page

"""Тесты секции приоритизации закупок витрины (issue #116, фаза 6.3).

Фикстуры и сборка артефакта — по образцу ``test_site_niches.py``: полный
конвейер записей → кластеризация → ``serialize_showcase`` (v3 с блоком
``sourcing``) → ``build_site.build``. Формулы скора закупки покрыты в
``test_sourcing_scoring.py``, здесь — сериализация в артефакт и рендер:
объяснимость (скор + компоненты + окно на странице), обратная
совместимость v2 (без sourcing) и пустой sourcing-блок.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_site
from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.nlp.clustering import cluster_phrases
from wordstat_trends.showcase import (
    NICHE_TOP_PHRASES,
    READABLE_SCHEMAS,
    SCHEMA,
    load_showcase,
    save_showcase,
    serialize_showcase,
)
from wordstat_trends.trends.growth import SCORE_WINDOW
from wordstat_trends.trends.ranking import TrendClass, phrase_record, rank_showcase

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _synthetic_record(phrase: str, multiplier: float):
    pattern = _fixture_series("dynamics_seasonal.csv").iloc[:12].to_numpy()
    flat = pd.Series(
        np.tile(pattern.astype(float), 3),
        index=pd.PeriodIndex(pd.period_range("2024-01", periods=36, freq="M"), freq="M"),
        name="queries",
    )
    flat.iloc[-SCORE_WINDOW:] = flat.iloc[-SCORE_WINDOW:] * multiplier
    return phrase_record(flat, phrase=phrase)


def _records() -> list:
    return [
        _synthetic_record("курсы английского", 3.0),
        _synthetic_record("выучить английский", 2.5),
        _synthetic_record("репетитор английского", 2.0),
        _synthetic_record("английский для детей", 0.5),
        phrase_record(_fixture_series("dynamics_seasonal.csv"), phrase="новогодние подарки"),
        phrase_record(_fixture_series("dynamics_high_freq.csv"), phrase="купить телефон"),
    ]


def _artifact() -> dict:
    records = _records()
    return serialize_showcase(
        records,
        generated_at=date(2026, 9, 10),
        result=cluster_phrases([row.phrase for row in records], pipeline="tfidf"),
    )


def _build(tmp_path: Path, artifact: dict | None) -> dict[str, str]:
    artifact_path = None
    if artifact is not None:
        artifact_path = tmp_path / "artifact.json"
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    root = tmp_path / "site"
    build_site.build(root, build_date=date(2026, 9, 10), artifact_path=artifact_path)
    return {str(p.relative_to(root)): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.html"))}


# --- сериализация: артефакт v3 с sourcing -------------------------------------


def test_serialize_schema_is_v3_with_sourcing():
    artifact = _artifact()
    assert artifact["schema"] == SCHEMA == "showcase/v3"
    assert artifact["sourcing"], "с кластеризацией sourcing должен быть непустым"
    assert "showcase/v2" in READABLE_SCHEMAS  # обратная совместимость чтения


def test_serialize_sourcing_shape_and_explainability():
    artifact = _artifact()
    ranked = {row.phrase: row.rank for row in rank_showcase(_records())}
    for entry in artifact["sourcing"]:
        assert entry["class"] in {c.name for c in TrendClass}
        assert entry["topic"]
        # объяснимость: скор и все четыре компоненты видны в разложении
        assert set(entry["components"]) == {"trend", "seasonal", "window", "stability"}
        assert all(0.0 <= value <= 1.0 for value in entry["components"].values())
        # окно: календарь или честный None (несезонная ниша)
        if entry["window"] is not None:
            assert set(entry["window"]) == {"order_month", "peak_month", "on_time"}
            assert 1 <= entry["window"]["order_month"] <= 12
            assert 1 <= entry["window"]["peak_month"] <= 12
        # топ-фразы состава — с рангами витрины (ссылки на карточки #105)
        assert 0 < len(entry["phrases"]) <= NICHE_TOP_PHRASES
        for item in entry["phrases"]:
            assert item["rank"] == ranked[item["phrase"]]


def test_serialize_sourcing_score_is_weighted_sum():
    import wordstat_trends.sourcing.scoring as scoring

    for entry in _artifact()["sourcing"]:
        c = entry["components"]
        expected = (
            scoring.TREND_WEIGHT * c["trend"]
            + scoring.SEASONAL_WEIGHT * c["seasonal"]
            + scoring.WINDOW_WEIGHT * c["window"]
            + scoring.STABILITY_WEIGHT * c["stability"]
        )
        assert abs(entry["score"] - expected) < 1e-6


def test_serialize_without_clustering_gives_empty_sourcing():
    artifact = serialize_showcase(_records())
    assert artifact["sourcing"] == []
    assert artifact["niches"] == []


def test_sourcing_round_trip_through_file(tmp_path):
    records = _records()
    result = cluster_phrases([row.phrase for row in records], pipeline="tfidf")
    path = save_showcase(records, tmp_path / "showcase.json", generated_at=date(2026, 9, 10), result=result)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == serialize_showcase(records, generated_at=date(2026, 9, 10), result=result)
    assert load_showcase(path) == data


def test_load_reads_v2_without_sourcing(tmp_path):
    # v2 (#106) не содержит sourcing — читается, витрина соберётся без секции
    v2 = tmp_path / "v2.json"
    v2.write_text(
        json.dumps(
            {
                "schema": "showcase/v2",
                "generated_at": "2026-09-10",
                "niches": [],
                "phrases": [{"phrase": "x", "class": "GROWING", "rank": 1, "score": 0.5, "components": None}],
            }
        ),
        encoding="utf-8",
    )
    data = load_showcase(v2)
    assert data["schema"] == "showcase/v2"
    assert "sourcing" not in data


def test_load_rejects_corrupted_sourcing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "generated_at": "2026-09-10",
                "sourcing": "не список",
                "phrases": [{"phrase": "x", "class": "GROWING", "rank": 1, "score": 0.5, "components": None}],
            }
        ),
        encoding="utf-8",
    )
    from wordstat_trends.showcase import ShowcaseError

    with pytest.raises(ShowcaseError, match="sourcing"):
        load_showcase(bad)


# --- рендер секции на витрине ----------------------------------------------------


def test_index_renders_sourcing_table_with_explanation(tmp_path):
    pages = _build(tmp_path, _artifact())
    index = pages["index.html"]
    assert "Товарные ниши" in index
    # формула весов — под заголовком: каждое число объяснимо
    assert "0,4 × тренд" in index
    assert "Скор закупки" in index
    assert "Сезонность" in index and "Устойчивость" in index
    # компоненты в разложении: локализованные числа в ячейках таблицы
    assert 'class="sourcing-table"' in index
    assert "сезонного пика нет — решение по тренду" in index
    # топ-фразы состава — ссылки на карточки трендов (#105)
    assert 'href="trends.html#phrase-' in index
    assert pages["trends.html"].count('id="phrase-') >= 3


def test_index_sourcing_window_calendar_ru(tmp_path):
    # «заказывать в … → пик в …» — месяцы окном, локализованные
    index = _build(tmp_path, _artifact())["index.html"]
    assert "заказывать в" in index and "пик в" in index


def test_zh_index_renders_sourcing_localized(tmp_path):
    zh = _build(tmp_path, _artifact())["zh/index.html"]
    assert "商品利基" in zh
    assert "采购评分" in zh
    assert "0.4 × 趋势" in zh
    # фразы состава — перевод + оригинал кириллицей (#20, #120)
    assert "новогодние подарки" in zh


def test_v2_artifact_renders_without_sourcing_section(tmp_path):
    # v2 без sourcing: сборка не падает, секции нет, карточки трендов на месте
    v2 = {
        "schema": "showcase/v2",
        "generated_at": "2026-09-10",
        "niches": [],
        "phrases": [{"phrase": "x", "class": "GROWING", "rank": 1, "score": 0.5, "components": None}],
    }
    pages = _build(tmp_path, v2)
    assert "Товарные ниши" not in pages["index.html"]
    assert 'id="phrase-1"' in pages["trends.html"]


def test_empty_sourcing_renders_meaningful_section(tmp_path):
    # v3 с пустым sourcing (продакшн-путь непустых niches ещё едет): секция
    # рендерится осмысленно — заголовок и честное пустое состояние
    artifact = _artifact()
    artifact["sourcing"] = []
    pages = _build(tmp_path, artifact)
    index = pages["index.html"]
    assert "Товарные ниши" in index
    assert "Приоритизация закупок пока не посчитана" in index
    zh = pages["zh/index.html"]
    assert "采购优先级尚未计算" in zh


def test_sourcing_data_is_escaped(tmp_path):
    artifact = _artifact()
    artifact["sourcing"][0]["topic"] = "<script>вредитель</script>"
    pages = _build(tmp_path, artifact)
    assert "<script>" not in pages["index.html"]

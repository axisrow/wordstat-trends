"""Тесты ниш витрины (issue #106): связка кластеров с классами трендов.

Фикстуры — по образцу ``test_showcase.py`` / ``test_clustering.py``: три
«английские» фразы с общими леммами (синтетический рост с разными
множителями + падающая), плюс сезонная и стабильная фразы фикстур Фазы 1.
Кластеризация — настоящий TF-IDF-конвейер (основные зависимости, без
extra `nlp` и без фейков); качество группировки фраз с общими леммами
покрыто в ``test_clustering.py``, здесь — формулы ниши и рендер секции.
"""

from __future__ import annotations

import json
import statistics
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import build_site
from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.nlp.clustering import cluster_phrases
from wordstat_trends.showcase import (
    NICHE_TOP_PHRASES,
    SCHEMA,
    build_niches,
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


def _ranked() -> list:
    records = [
        _synthetic_record("курсы английского", 3.0),
        _synthetic_record("выучить английский", 2.5),
        _synthetic_record("репетитор английского", 2.0),
        _synthetic_record("английский для детей", 0.5),
        phrase_record(_fixture_series("dynamics_seasonal.csv"), phrase="новогодние подарки"),
        phrase_record(_fixture_series("dynamics_high_freq.csv"), phrase="купить телефон"),
    ]
    return rank_showcase(records)


def _niches() -> list:
    ranked = _ranked()
    result = cluster_phrases([row.phrase for row in ranked], pipeline="tfidf")
    return build_niches(ranked, result)


# --- формулы ниши (предрегистрация в docs/TRENDS.md) ------------------------


def test_english_niche_topic_and_dominant_class():
    english = [niche for niche in _niches() if "английский" in niche.topic]
    assert len(english) == 1
    niche = english[0]
    # три растущие фразы против одной падающей в составе — мода «растёт»
    assert niche.klass is TrendClass.GROWING
    # метка-тема — топ-леммы кластера
    assert "английский" in niche.top_lemmas


def test_english_niche_members_ranked_and_score_is_median():
    ranked = _ranked()
    by_phrase = {row.phrase: row for row in ranked}
    niche = next(n for n in _niches() if "английский" in n.topic)
    ranks = [row.rank for row in niche.members]
    assert ranks == sorted(ranks)
    dominant_scores = [
        by_phrase[row.phrase].score for row in niche.members if row.klass is TrendClass.GROWING
    ]
    assert niche.score == statistics.median(dominant_scores)


def test_seasonal_niche_score_is_zero_outside_growing():
    seasonal = [niche for niche in _niches() if "подарок" in " ".join(niche.top_lemmas)]
    assert len(seasonal) == 1
    assert seasonal[0].klass is TrendClass.SEASONAL
    # вне «растёт» скоры фраз нулевые — скор ниши 0.0 (docs/TRENDS.md)
    assert seasonal[0].score == 0.0


def test_niches_ordered_by_class_then_score():
    niches = _niches()
    keys = [
        (list(TrendClass).index(niche.klass), -niche.score, niche.topic) for niche in niches
    ]
    assert keys == sorted(keys)


# --- артефакт -----------------------------------------------------------------


def test_serialize_niches_shape():
    artifact = serialize_showcase(_ranked(), cluster_phrases(
        [row.phrase for row in _ranked()], pipeline="tfidf"
    ), generated_at=date(2026, 9, 10))
    assert artifact["schema"] == SCHEMA
    assert artifact["niches"], "ниши должны быть непустыми"
    for niche in artifact["niches"]:
        assert niche["class"] in {c.name for c in TrendClass}
        assert niche["topic"]
        assert 0 < len(niche["phrases"]) <= NICHE_TOP_PHRASES
        for entry in niche["phrases"]:
            assert set(entry) == {"phrase", "rank"}


def test_serialize_without_clustering_gives_empty_niches():
    artifact = serialize_showcase(_ranked())
    assert artifact["niches"] == []


def test_niche_round_trip_through_file(tmp_path):
    ranked = _ranked()
    result = cluster_phrases([row.phrase for row in ranked], pipeline="tfidf")
    path = save_showcase(ranked, tmp_path / "showcase.json", result=result, generated_at=date(2026, 9, 10))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == serialize_showcase(ranked, result, generated_at=date(2026, 9, 10))


# --- рендер секции на витрине ---------------------------------------------------


def _build(tmp_path: Path, artifact: dict | None) -> dict[str, str]:
    artifact_path = None
    if artifact is not None:
        artifact_path = tmp_path / "artifact.json"
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    root = tmp_path / "site"
    build_site.build(root, build_date=date(2026, 9, 10), artifact_path=artifact_path)
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.html"))
    }


def test_index_renders_niche_section_with_anchors(tmp_path):
    ranked = _ranked()
    result = cluster_phrases([row.phrase for row in ranked], pipeline="tfidf")
    pages = _build(tmp_path, serialize_showcase(ranked, result))
    index = pages["index.html"]
    assert "Темы и ниши" in index
    niche = next(n for n in result.clusters if "курсы английского" in n.phrases)
    for lemma in niche.top_lemmas:
        assert lemma in index
    # класс ниши переведён ключом локали, фразы — ссылки на строки таблицы
    assert "Растущие запросы" in index
    assert 'href="trends.html#phrase-' in index
    # цели ссылок существуют: строки таблицы трендов несут якоря phrase-N
    trends = pages["trends.html"]
    assert 'id="phrase-1"' in trends
    assert "主题与利基" in pages["zh/index.html"]


def test_no_niches_no_section(tmp_path):
    # артефакт с фразами, но без кластеризации — секции ниш нет
    pages = _build(tmp_path, serialize_showcase(_ranked()))
    assert "Темы и ниши" not in pages["index.html"]
    assert 'id="phrase-1"' in pages["trends.html"]


def test_empty_artifact_keeps_empty_state(tmp_path):
    # пустой артефакт (нет файла) — прежнее пустое состояние, без ниш
    pages = _build(tmp_path, None)
    assert "Темы и ниши" not in pages["index.html"]
    assert "Данных пока нет" in pages["trends.html"]

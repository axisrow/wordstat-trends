"""Тесты сериализации витрины (issue #104, фаза 5).

Фикстуры фраз — по образцу ``test_trends_ranking.py``: сезонная
(«новогодние подарки»), стабильная высокочастотная («купить телефон»),
синтетический рост. Проверяется форма артефакта и round-trip через JSON,
параметры детекции не подбираются.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.showcase import (
    SCHEMA,
    ShowcaseError,
    load_showcase,
    save_showcase,
    serialize_showcase,
)
from wordstat_trends.trends.growth import SCORE_WINDOW
from wordstat_trends.trends.ranking import (
    TrendClass,
    phrase_record,
    rank_showcase,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _growing_record(multiplier: float = 2.0):
    pattern = _fixture_series("dynamics_seasonal.csv").iloc[:12].to_numpy()
    flat = pd.Series(
        np.tile(pattern.astype(float), 3),
        index=pd.PeriodIndex(pd.period_range("2024-01", periods=36, freq="M"), freq="M"),
        name="queries",
    )
    flat.iloc[-SCORE_WINDOW:] = flat.iloc[-SCORE_WINDOW:] * multiplier
    return phrase_record(flat, phrase="синтетический рост")


def _ranked() -> list:
    records = [
        _growing_record(),
        phrase_record(_fixture_series("dynamics_seasonal.csv"), phrase="новогодние подарки"),
        phrase_record(_fixture_series("dynamics_high_freq.csv"), phrase="купить телефон"),
    ]
    return rank_showcase(records)


def test_serialize_schema_and_generated_at():
    artifact = serialize_showcase(_ranked(), generated_at=date(2026, 9, 9))
    assert artifact["schema"] == SCHEMA
    assert artifact["generated_at"] == "2026-09-09"
    assert len(artifact["phrases"]) == 3


def test_serialize_class_names_are_machine_keys():
    artifact = serialize_showcase(_ranked())
    classes = {p["class"] for p in artifact["phrases"]}
    assert classes <= {c.name for c in TrendClass}
    # синтетический рост классифицируется как рост и идёт первым
    assert artifact["phrases"][0]["class"] == "GROWING"


def test_serialize_components_only_for_growing():
    artifact = serialize_showcase(_ranked())
    for entry in artifact["phrases"]:
        if entry["class"] == "GROWING":
            assert entry["components"] is not None
            assert entry["score"] > 0
        else:
            assert entry["components"] is None
            assert entry["score"] == 0.0


def test_serialize_ranks_are_dense():
    artifact = serialize_showcase(_ranked())
    assert [p["rank"] for p in artifact["phrases"]] == [1, 2, 3]


def test_json_round_trip(tmp_path):
    ranked = _ranked()
    path = save_showcase(ranked, tmp_path / "showcase.json", generated_at=date(2026, 9, 9))
    assert json.loads(path.read_text(encoding="utf-8")) == serialize_showcase(
        ranked, generated_at=date(2026, 9, 9)
    )
    assert load_showcase(path) == serialize_showcase(
        ranked, generated_at=date(2026, 9, 9)
    )


def test_load_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "phrases": []}), encoding="utf-8")
    with pytest.raises(ShowcaseError):
        load_showcase(bad)

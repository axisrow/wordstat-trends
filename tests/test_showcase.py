"""Тесты сериализации витрины (issue #104, фаза 5; ряд в артефакте — #105).

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
    READABLE_SCHEMAS,
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


def _records() -> list:
    return [
        _growing_record(),
        phrase_record(_fixture_series("dynamics_seasonal.csv"), phrase="новогодние подарки"),
        phrase_record(_fixture_series("dynamics_high_freq.csv"), phrase="купить телефон"),
    ]


def test_serialize_schema_and_generated_at():
    artifact = serialize_showcase(_records(), generated_at=date(2026, 9, 9))
    assert artifact["schema"] == SCHEMA
    assert artifact["generated_at"] == "2026-09-09"
    assert len(artifact["phrases"]) == 3


def test_serialize_class_names_are_machine_keys():
    artifact = serialize_showcase(_records())
    classes = {p["class"] for p in artifact["phrases"]}
    assert classes <= {c.name for c in TrendClass}
    # синтетический рост классифицируется как рост и идёт первым
    assert artifact["phrases"][0]["class"] == "GROWING"


def test_serialize_components_only_for_growing():
    artifact = serialize_showcase(_records())
    for entry in artifact["phrases"]:
        if entry["class"] == "GROWING":
            assert entry["components"] is not None
            assert entry["score"] > 0
        else:
            assert entry["components"] is None
            assert entry["score"] == 0.0


def test_serialize_ranks_are_dense():
    artifact = serialize_showcase(_records())
    assert [p["rank"] for p in artifact["phrases"]] == [1, 2, 3]


def test_serialize_carries_full_series():
    # issue #105: ряд кладётся в артефакт целиком — и история, и окно скоринга
    records = {r.phrase: r for r in _records()}
    artifact = serialize_showcase(_records())
    for entry in artifact["phrases"]:
        record = records[entry["phrase"]]
        series = entry["series"]
        assert len(series["values"]) == len(record.series)
        assert series["periods"][0] == str(record.series.index[0])
        assert series["periods"][-1] == str(record.series.index[-1])
        # все периоды — календарные месяцы YYYY-MM
        assert all(len(p) == 7 and p[4] == "-" for p in series["periods"])
        # окно скоринга — те же последние месяцы, что в детекции
        assert entry["window_months"] == len(record.growth.actual)
        # отношение факт/прогноз — из детекции, а не взвешенный скор витрины
        assert entry["ratio"] == pytest.approx(record.growth.score, abs=1e-6)


def test_serialize_filtered_out_records_excluded():
    artifact = serialize_showcase(
        _records()[:1] + [phrase_record(_fixture_series("dynamics_seasonal.csv"), phrase="мусор", filtered_out=True)]
    )
    assert [p["phrase"] for p in artifact["phrases"]] == ["синтетический рост"]


def test_json_round_trip(tmp_path):
    records = _records()
    path = save_showcase(records, tmp_path / "showcase.json", generated_at=date(2026, 9, 9))
    assert json.loads(path.read_text(encoding="utf-8")) == serialize_showcase(
        records, generated_at=date(2026, 9, 9)
    )
    assert load_showcase(path) == serialize_showcase(
        records, generated_at=date(2026, 9, 9)
    )


def test_load_reads_v1_for_backward_compatibility(tmp_path):
    # v1 (#104) не содержит ряда и ratio — читается, витрина соберётся без графиков
    v1 = tmp_path / "v1.json"
    v1.write_text(
        json.dumps({
            "schema": "showcase/v1",
            "generated_at": "2026-09-09",
            "phrases": [{"phrase": "x", "class": "GROWING", "rank": 1, "score": 0.5, "components": None}],
        }),
        encoding="utf-8",
    )
    assert load_showcase(v1)["schema"] == "showcase/v1"


def test_load_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "phrases": []}), encoding="utf-8")
    with pytest.raises(ShowcaseError):
        load_showcase(bad)
    assert "showcase/v1" in READABLE_SCHEMAS

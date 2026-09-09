"""Тесты скоринга ниш для закупки (issue #115, фаза 6.2).

Фикстуры — по образцу ``test_site_niches.py``: синтетическая растущая
ниша (три растущие + одна падающая фраза), сезонная «новогодние
подарки» (пик в декабре, ряд кончается 2026-07), ниша однодневок (весь
состав отсеян ступенью #84) и пустой состав. Формулы и веса
предрегистрированы в docs/TRENDS.md, тесты их проверяют, параметры не
подбирают.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.nlp.clustering import cluster_phrases
from wordstat_trends.showcase import build_niches
from wordstat_trends.sourcing.scoring import (
    INTENT_MARKER_LEMMAS,
    SEASONAL_RATIO_TOP,
    SEASONAL_WEIGHT,
    STABILITY_WEIGHT,
    TREND_WEIGHT,
    WINDOW_WEIGHT,
    intent_share,
    score_sourcing,
    seasonal_component,
    stability_component,
    window_component,
)
from wordstat_trends.sourcing.timing import DEFAULT_LEAD_TIME_WEEKS, lead_time_months
from wordstat_trends.trends.growth import SCORE_WINDOW
from wordstat_trends.trends.ranking import (
    SEASONAL_MAX_MIN_RATIO,
    TrendClass,
    phrase_record,
    rank_showcase,
    seasonal_max_min_ratio,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _synthetic_record(phrase: str, multiplier: float, filtered_out: bool = False):
    pattern = _fixture_series("dynamics_seasonal.csv").iloc[:12].to_numpy()
    flat = pd.Series(
        np.tile(pattern.astype(float), 3),
        index=pd.PeriodIndex(pd.period_range("2024-01", periods=36, freq="M"), freq="M"),
        name="queries",
    )
    flat.iloc[-SCORE_WINDOW:] = flat.iloc[-SCORE_WINDOW:] * multiplier
    return phrase_record(flat, phrase=phrase, filtered_out=filtered_out)


def _records() -> list:
    return [
        _synthetic_record("курсы английского", 3.0),
        _synthetic_record("выучить английский", 2.5),
        _synthetic_record("репетитор английского", 2.0),
        _synthetic_record("английский для детей", 0.5),
        _synthetic_record("английский по скайпу недорого", 3.0, filtered_out=True),
        phrase_record(_fixture_series("dynamics_seasonal.csv"), phrase="новогодние подарки"),
        phrase_record(_fixture_series("dynamics_high_freq.csv"), phrase="купить телефон"),
        # однодневки: отсеяны ступенью #84 целиком, леммы общие — кластер свой
        phrase_record(
            _fixture_series("dynamics_seasonal.csv"),
            phrase="микрокракен портал",
            filtered_out=True,
        ),
        phrase_record(
            _fixture_series("dynamics_seasonal.csv"),
            phrase="микрокракен буст",
            filtered_out=True,
        ),
    ]


def _niches(records=None) -> list:
    records = records if records is not None else _records()
    ranked = rank_showcase(records)
    result = cluster_phrases([record.phrase for record in records], pipeline="tfidf")
    return build_niches(ranked, result, records)


def _sourcing() -> list:
    records = _records()
    return score_sourcing(_niches(records), records)


# --- метрики устойчивости ниши (шов в showcase.build_niches) -------------------


def test_growing_niche_stability_metrics():
    niche = next(n for n in _niches() if "английский" in n.topic)
    assert niche.klass is TrendClass.GROWING
    # 4 фразы кластера в витрине + 1 отсеянная ступенью #84 → доля 1/5
    assert niche.filtered_share == 1 / 5
    # все растущие фразы синтетики держат порог ровно окно скоринга
    assert niche.median_growth_age_months == SCORE_WINDOW


def test_metrics_default_without_records():
    # без записей до отсева конвейер #106 не меняется: «отсева нет»
    records = _records()
    ranked = rank_showcase(records)
    result = cluster_phrases([record.phrase for record in records], pipeline="tfidf")
    niche = next(
        n for n in build_niches(ranked, result) if "английский" in n.topic
    )
    assert niche.filtered_share == 0.0
    assert niche.median_growth_age_months == SCORE_WINDOW


def test_oneday_niche_not_built():
    # весь состав кластера отсеян → членов витрины нет → ниши нет
    topics = [niche.topic for niche in _niches()]
    assert all("микрокракен" not in topic for topic in topics)


def test_empty_composition_gives_empty_sourcing():
    assert score_sourcing([], []) == []


# --- компоненты скора (предрегистрация в docs/TRENDS.md) -----------------------


def test_seasonal_component_bounds():
    flat = _fixture_series("dynamics_high_freq.csv")
    assert seasonal_max_min_ratio(flat) < SEASONAL_MAX_MIN_RATIO
    assert seasonal_component(flat) == 0.0
    seasonal = _fixture_series("dynamics_seasonal.csv")
    expected = min(
        math.log10(seasonal_max_min_ratio(seasonal)) / math.log10(SEASONAL_RATIO_TOP),
        1.0,
    )
    assert seasonal_component(seasonal) == expected


def test_window_component_binary():
    assert window_component(None) == 0.0
    off = SimpleNamespace(on_time=False)
    assert window_component(off) == 0.0
    on = SimpleNamespace(on_time=True)
    assert window_component(on) == 1.0


def test_stability_component_formula():
    growing = SimpleNamespace(
        filtered_share=0.2,
        klass=TrendClass.GROWING,
        median_growth_age_months=6,
    )
    assert stability_component(growing) == 0.8 / 6
    seasonal = SimpleNamespace(
        filtered_share=0.2, klass=TrendClass.SEASONAL, median_growth_age_months=None
    )
    # вне «растёт» возраста роста нет — фактор 1.0 без штрафа
    assert stability_component(seasonal) == 0.8


# --- скор и приоритизация -------------------------------------------------------


def test_seasonal_niche_window_on_time():
    sourcing = {row.topic: row for row in _sourcing()}
    seasonal = next(
        row for topic, row in sourcing.items() if "новогодн" in topic or "подарок" in topic
    )
    window = seasonal.window
    assert window is not None
    # пик декабря, ряд кончается 2026-07: заказ к пику 2026-12 успеваем
    assert window.peak_month == 12
    assert window.on_time
    assert seasonal.components.window == 1.0
    assert seasonal.components.seasonal > 0.0


def test_growing_niche_score_is_weighted_composition():
    records = _records()
    niche = next(n for n in _niches(records) if "английский" in n.topic)
    rows = score_sourcing([niche], records)
    row = rows[0]
    # арифметика композиции: скор = веса × компоненты (docs/TRENDS.md)
    assert row.score == (
        TREND_WEIGHT * row.components.trend
        + SEASONAL_WEIGHT * row.components.seasonal
        + WINDOW_WEIGHT * row.components.window
        + STABILITY_WEIGHT * row.components.stability
    )
    assert row.components.trend == niche.score
    assert row.components.stability == stability_component(niche)
    assert 0.0 <= row.score <= 1.0


def test_all_components_in_unit_interval():
    for row in _sourcing():
        for value in (
            row.components.trend,
            row.components.seasonal,
            row.components.window,
            row.components.stability,
            row.score,
        ):
            assert 0.0 <= value <= 1.0


def test_sourcing_sorted_deterministically():
    rows = _sourcing()
    keys = [
        (-row.score, list(TrendClass).index(row.klass), row.topic) for row in rows
    ]
    assert keys == sorted(keys)
    assert [k for k in keys] == sorted(
        keys
    )  # повторный прогон — тот же порядок (стабильность сортировки)


def test_sourcing_carries_structure_not_render():
    row = _sourcing()[0]
    # выход — структура: состав фраз, окно #114, метрики устойчивости
    assert row.phrases
    assert row.window is None or row.window.lead_time_months == lead_time_months(
        DEFAULT_LEAD_TIME_WEEKS
    )
    assert row.filtered_share >= 0.0


# --- гипотеза purchase-intent (механизм, не вывод о пользе) --------------------


def test_intent_share_by_lemma_markers():
    members = [
        SimpleNamespace(phrase="купить телефон"),
        SimpleNamespace(phrase="телефон цена"),
        SimpleNamespace(phrase="сколько стоит телефон"),
        SimpleNamespace(phrase="телефон"),
    ]
    # «купить», «цена», «стоимость» — маркеры; «сколько/стоит» тоже лемма
    # «стоить»? нет: маркер — «стоимость», а не «стоить» → доля 2/4.
    assert intent_share(SimpleNamespace(members=members)) == 0.5


def test_intent_share_empty_members():
    assert intent_share(SimpleNamespace(members=[])) == 0.0


def test_intent_markers_are_lemmas():
    # словарь хранит леммы (не словоформы) — сравнение идёт по леммам фраз
    assert "купить" in INTENT_MARKER_LEMMAS
    assert "дешёвый" in INTENT_MARKER_LEMMAS
    assert intent_share(
        SimpleNamespace(members=[SimpleNamespace(phrase="телефон дешёвый")])
    ) == 1.0


def test_intent_share_not_in_score():
    # гипотеза помечена и в скор НЕ входит: состав компонент фиксирован
    records = _records()
    niche = next(n for n in _niches(records) if "английский" in n.topic)
    rows = score_sourcing([niche], records)
    assert set(rows[0].components.__dataclass_fields__) == {
        "trend",
        "seasonal",
        "window",
        "stability",
    }

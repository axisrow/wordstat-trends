"""Тесты ранжирования для витрины (issue #85, фаза 3 Б3).

Составные фикстуры по issue: сезонная («новогодние подарки») +
высокочастотная стабильная («купить телефон») + синтетический рост.
Пороги и веса предрегистрированы в docs/TRENDS.md; тесты проверяют
классификацию, порядок и стабильность, параметры не подбирают.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.trends.growth import SCORE_WINDOW, score_growth
from wordstat_trends.trends.ranking import (
    ANOMALY_WEIGHT,
    FRESHNESS_WEIGHT,
    VOLUME_WEIGHT,
    PhraseRecord,
    RankedPhrase,
    TrendClass,
    classify,
    growth_age_months,
    growth_score_components,
    phrase_record,
    phrase_record_from_frame,
    rank_showcase,
    seasonal_max_min_ratio,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _flat_seasonal_series() -> pd.Series:
    """Синтетика: сезонный профиль, повторённый год в год без роста."""

    pattern = _fixture_series("dynamics_seasonal.csv").iloc[:12].to_numpy()
    return pd.Series(
        np.tile(pattern.astype(float), 3),
        index=pd.PeriodIndex(
            pd.period_range("2024-01", periods=36, freq="M"), freq="M"
        ),
        name="queries",
    )


def _growing_series(multiplier: float = 2.0, months: int = SCORE_WINDOW) -> pd.Series:
    boosted = _flat_seasonal_series()
    boosted.iloc[-months:] = boosted.iloc[-months:] * multiplier
    return boosted


def _record(series: pd.Series, phrase: str) -> PhraseRecord:
    return phrase_record(series, phrase=phrase)


def _composite_records() -> list[PhraseRecord]:
    """Составная фикстура issue: сезонная + высокочастотная + синтетический рост."""

    return [
        _record(_fixture_series("dynamics_seasonal.csv"), "новогодние подарки"),
        _record(_fixture_series("dynamics_high_freq.csv"), "купить телефон"),
        _record(_growing_series(), "растущая синтетика"),
    ]


def test_composite_fixture_classification():
    showcase = rank_showcase(_composite_records())
    by_phrase = {row.phrase: row for row in showcase}
    assert by_phrase["растущая синтетика"].klass is TrendClass.GROWING
    assert by_phrase["новогодние подарки"].klass is TrendClass.SEASONAL
    assert by_phrase["купить телефон"].klass is TrendClass.STABLE


def test_showcase_order_classes_then_ranks():
    showcase = rank_showcase(_composite_records() + [_record(_falling_series(), "падающая")])
    assert [row.klass for row in showcase] == [
        TrendClass.GROWING,
        TrendClass.SEASONAL,
        TrendClass.FALLING,
        TrendClass.STABLE,
    ]
    assert [row.rank for row in showcase] == [1, 2, 3, 4]


def _falling_series() -> pd.Series:
    dampened = _flat_seasonal_series()
    dampened.iloc[-SCORE_WINDOW:] = dampened.iloc[-SCORE_WINDOW:] * 0.5
    return dampened


def test_falling_detected():
    # Падение — устойчивое отклонение вниз от сезонного ожидания: скор 0.5
    # ниже DECLINE_RATIO_THRESHOLD, класс — «падает».
    row = rank_showcase([_record(_falling_series(), "падающая")])[0]
    assert row.klass is TrendClass.FALLING
    assert row.components is None


def test_seasonal_amplitude_separates_fixtures():
    # Сезонная фикстура — амплитуда десятки, высокочастотная стабильная
    # ≈ 1.2: порог 2.0 разделяет их с запасом в обе стороны.
    assert seasonal_max_min_ratio(_fixture_series("dynamics_seasonal.csv")) > 2.0
    assert seasonal_max_min_ratio(_fixture_series("dynamics_high_freq.csv")) < 2.0


def test_growing_score_components_bounded():
    components = growth_score_components(_record(_growing_series(), "растущая"))
    for value in (components.anomaly, components.volume, components.freshness):
        assert 0.0 <= value <= 1.0
    # Хвост роста ровно SCORE_WINDOW месяцев ×2 → возраст роста = окну.
    assert components.growth_age_months == SCORE_WINDOW
    assert components.freshness == pytest.approx(1.0 / SCORE_WINDOW)


def test_growth_age_months_walks_back():
    # Рост длиной 1 месяц против 3 месяцев: возраст считается по хвосту
    # месяцев, держащих порог, а не по окну скоринга.
    assert growth_age_months(_growing_series(months=1)) == 1
    assert growth_age_months(_growing_series(months=3)) == 3
    # Без роста (плоская сезонная база) возраст не определён.
    assert growth_age_months(_flat_seasonal_series()) is None


def test_volume_breaks_ties_between_equal_growth():
    # Звезда с 10 запросами не интересна закупке: при равной силе роста
    # выше встаёт фраза с большей частотностью.
    low_volume = _scaled(_growing_series(), window_scale=1.0, base_scale=0.001)
    high_volume = _scaled(_growing_series(), window_scale=1.0, base_scale=1.0)
    showcase = rank_showcase(
        [_record(low_volume, "мелкая растущая"), _record(high_volume, "крупная растущая")]
    )
    assert showcase[0].phrase == "крупная растущая"
    first_c = showcase[0].components
    second_c = showcase[1].components
    assert first_c is not None and second_c is not None
    assert first_c.volume > second_c.volume


def _scaled(series: pd.Series, window_scale: float, base_scale: float) -> pd.Series:
    """Сила роста (окно × window_scale) и частотность (база × base_scale)
    регулируются независимо — компоненты скора разведены."""

    scaled = series.astype(float) * base_scale
    scaled.iloc[-SCORE_WINDOW:] = series.iloc[-SCORE_WINDOW:] * base_scale * window_scale
    return scaled


def test_freshness_component_prefers_recent_growth():
    # Свежесть — компонента скора: рост, начавшийся в последний месяц,
    # свежее роста, тянущегося 6 месяцев.
    # Рост длиной 1 месяц: чтобы окно скоринга (3 месяца) подтвердило рост,
    # единственный растущий месяц утраивается — (3+1+1)/3 = 1.67 >= порога.
    fresh = _record(_growing_series(months=1, multiplier=3.0), "свежий рост")
    old = _record(_growing_series(months=6), "старый рост")
    rows = {row.phrase: row for row in rank_showcase([old, fresh])}
    fresh_c = rows["свежий рост"].components
    old_c = rows["старый рост"].components
    assert fresh_c is not None and old_c is not None
    assert fresh_c.freshness > old_c.freshness
    assert fresh_c.growth_age_months == 1
    assert old_c.growth_age_months == 6


def test_determinism_same_input_same_order():
    records = _composite_records()
    assert rank_showcase(records) == rank_showcase(records)


def test_adding_phrase_preserves_order_of_others():
    # Требование issue #85: добавление одной фразы не переворачивает
    # порядок остальных без причины. Скор фразы зависит только от её
    # собственных данных (нормировка «относительно себя»), поэтому
    # относительный порядок исходных фраз сохраняется как подпоследовательность.
    base = _composite_records() + [
        _record(_falling_series(), "падающая"),
        _record(_growing_series(months=6), "долгий рост"),
    ]
    before = rank_showcase(base)
    added = base + [_record(_scaled(_growing_series(), 3.0, 10.0), "новая звезда")]
    after = rank_showcase(added)

    before_order = [row.phrase for row in before]
    after_order = [row.phrase for row in after if row.phrase != "новая звезда"]
    # after_order — перестановка before_order, индуцированная только
    # вставкой, порядок старых фраз не меняется.
    assert after_order == before_order
    # Скоры старых фраз не изменились вовсе.
    scores_before = {row.phrase: row.score for row in before}
    for row in after:
        if row.phrase != "новая звезда":
            assert row.score == pytest.approx(scores_before[row.phrase])


def test_filtered_out_records_excluded():
    # Ступень отсева (#84) выше по течению: отфильтрованная запись
    # помечается флагом и в витрину не попадает.
    records = _composite_records()
    filtered = [
        PhraseRecord(
            phrase=r.phrase,
            series=r.series,
            growth=r.growth,
            filtered_out=True,
        )
        for r in records[:1]
    ]
    showcase = rank_showcase(filtered + records[1:])
    assert [row.phrase for row in showcase] == ["растущая синтетика", "купить телефон"]


def test_tie_break_by_phrase():
    # Ровно одинаковые записи: порядок по фразе — детерминированность.
    first = _record(_growing_series(), "бета")
    second = _record(_growing_series(), "альфа")
    showcase = rank_showcase([first, second])
    assert [row.phrase for row in showcase] == ["альфа", "бета"]
    assert showcase[0].score == showcase[1].score


def test_weights_sum_to_one():
    # Веса — явные константы, в сумме единица (скор остаётся в [0, 1]).
    assert ANOMALY_WEIGHT + VOLUME_WEIGHT + FRESHNESS_WEIGHT == pytest.approx(1.0)


def test_phrase_record_from_frame_attrs():
    rows = parse_dynamics_rows(
        (FIXTURES / "dynamics_seasonal.csv").read_text(encoding="utf-8-sig")
    )
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    frame.attrs["phrase"] = "новогодние подарки"
    record = phrase_record_from_frame(frame)
    assert record.phrase == "новогодние подарки"
    assert record.growth.score == pytest.approx(
        score_growth(record.series, phrase="новогодние подарки").score
    )
    assert isinstance(rank_showcase([record])[0], RankedPhrase)
    assert classify(record) is TrendClass.SEASONAL

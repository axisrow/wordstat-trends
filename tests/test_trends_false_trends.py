"""Тесты отсева ложных трендов (issue #84, фаза 3 Б2).

Пороги предрегистрированы в docs/TRENDS.md до прогона; тесты проверяют
свойства фильтров на синтетике и обязательные ложноположительные случаи:
инфоповод, праздничная фраза без роста амплитуды, праздничная фраза с
растущей амплитудой (новые фикстуры — dynamics_seasonal.csv рост
амплитуды не покрывает). Параметры не подгоняются.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.trends.false_trends import (
    MIN_CONFIRMED_MONTHS,
    MONTH_RATIO_THRESHOLD,
    PEAK_GROWTH_RATIO,
    REASON_GROWTH_CONFIRMED,
    REASON_NO_GROWTH,
    REASON_PEAK_AMPLITUDE_GROWING,
    REASON_SEASONAL_PEAK_FLAT,
    REASON_SINGLE_SPIKE,
    TrendVerdict,
    classify_growth,
    month_ratios,
)
from wordstat_trends.trends.growth import (
    GROWTH_RATIO_THRESHOLD,
    SCORE_WINDOW,
    score_growth,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _tiled_seasonal(n_years: int, end: str) -> pd.Series:
    """Синтетика: сезонный профиль «новогодних подарков», повторённый без
    роста (та же конструкция, что в test_trends_growth)."""

    pattern = _fixture_series("dynamics_seasonal.csv").iloc[:12].to_numpy()
    periods = pd.period_range(end=end, periods=12 * n_years, freq="M")
    return pd.Series(
        np.tile(pattern.astype(float), n_years),
        index=pd.PeriodIndex(periods, freq="M"),
        name="queries",
    )


# --- Фильтр 1: разовый всплеск (инфоповод) -------------------------------


def test_info_surge_rejected_by_filter_one():
    # Ложноположительный случай «инфоповод»: один месяц ×6 утащил среднее
    # окна за порог детекции (скор > 1.5 — детекция сказала бы «рост»),
    # но подтверждён лишь один месяц — не тренд. Месячная грануляция уже
    # погасила дневные всплески; этот фильтр — про месячный инфоповод.
    spiked = _tiled_seasonal(3, "2026-12")
    spiked.iloc[-1] = spiked.iloc[-1] * 6.0
    score = score_growth(spiked, phrase="инфоповод")
    assert score.is_growth  # детекция без отсева дала бы ложный тренд
    verdict = classify_growth(spiked, phrase="инфоповод")
    assert isinstance(verdict, TrendVerdict)
    assert not verdict.is_trend
    assert verdict.reason == REASON_SINGLE_SPIKE
    assert len(verdict.confirmed_months) == 1
    assert len(verdict.confirmed_months) < MIN_CONFIRMED_MONTHS


def test_sustained_growth_confirmed():
    # Рост ×2 во всём окне из трёх месяцев (несезонные май–июль) —
    # устойчивый: подтверждены все месяцы, месяцы вне праздничной дуги.
    grown = _tiled_seasonal(3, "2026-07")
    grown.iloc[-SCORE_WINDOW:] = grown.iloc[-SCORE_WINDOW:] * 2.0
    verdict = classify_growth(grown, phrase="растущая")
    assert verdict.is_trend
    assert verdict.reason == REASON_GROWTH_CONFIRMED
    assert len(verdict.confirmed_months) == SCORE_WINDOW
    assert all(r == pytest.approx(2.0) for r in verdict.month_ratios)


def test_two_of_three_months_confirm_enough():
    # Два из трёх месяцев выше порога — минимальное подтверждение по
    # предрегистрации (MIN_CONFIRMED_MONTHS = 2), третий месяц — шум.
    mixed = _tiled_seasonal(3, "2026-07")
    mixed.iloc[-2:] = mixed.iloc[-2:] * 2.0
    verdict = classify_growth(mixed, phrase="частично растущая")
    assert len(verdict.confirmed_months) == 2
    assert verdict.is_trend


def test_month_ratios_match_score_window():
    y = _fixture_series("dynamics_seasonal.csv")
    score = score_growth(y, phrase="новогодние подарки")
    ratios = month_ratios(score)
    assert len(ratios) == len(score.window)
    expected = tuple(
        a / f for a, f in zip(score.actual.to_numpy(), score.forecast.to_numpy())
    )
    assert ratios == pytest.approx(expected)


def test_zero_month_forecast_rejected():
    # Нулевой месяц в train (низкочастотная фраза) → нулевой прогноз для
    # одного месяца окна: среднее прогноза остаётся положительным и
    # score_growth его пропускает, но помесячное отношение ушло бы в inf и
    # месяц молча «подтвердил» бы рост. Громкий отказ вместо тихого
    # ложноположительного вердикта (по духу NaN-гварда #83).
    y = _tiled_seasonal(3, "2026-07").astype(float)
    y.iloc[-13] = 0.0  # нулевой месяц ровно 12 мес назад от последнего окна
    with pytest.raises(ValueError, match="неположительный прогноз"):
        classify_growth(y, phrase="низкочастотная")


# --- Фильтр 2: праздничная сезонность против тренда ----------------------


def test_holiday_flat_peak_rejected():
    # Ложноположительный случай «праздничная без роста»: пик декабрь
    # 1.2M год в год стабилен, но текущий сезон в окне ×2 против
    # прошлого года — скор детекции 2.0, подтверждены все месяцы дуги.
    # Амплитуда пика год к году ≈ 1.0 — сезонность без роста, отсев.
    y = _fixture_series("dynamics_holiday_flat.csv")
    score = score_growth(y, phrase="новогодние подарки оптом")
    assert score.is_growth
    assert score.score == pytest.approx(2.0)
    verdict = classify_growth(y, phrase="новогодние подарки оптом")
    assert not verdict.is_trend
    assert verdict.reason == REASON_SEASONAL_PEAK_FLAT
    assert verdict.peak_ratio is not None
    assert verdict.peak_ratio == pytest.approx(1.0)
    assert verdict.peak_ratio < PEAK_GROWTH_RATIO
    assert verdict.ets_structure is not None
    assert verdict.ets_structure["has_seasonal"] is True


def test_holiday_growing_amplitude_is_trend():
    # «Праздничная с ростом амплитуды»: весь профиль растёт ×1.7 в год,
    # амплитуда пика последнего полного года / предыдущего = 1.7 —
    # растущий пик это тренд, не сезонность.
    y = _fixture_series("dynamics_holiday_growing.csv")
    score = score_growth(y, phrase="гирлянды на елку")
    assert score.is_growth
    verdict = classify_growth(y, phrase="гирлянды на елку")
    assert verdict.is_trend
    assert verdict.reason == REASON_PEAK_AMPLITUDE_GROWING
    assert verdict.peak_ratio == pytest.approx(1.7)
    assert 1.7 >= PEAK_GROWTH_RATIO


def test_seasonal_fixture_below_detection_is_no_growth():
    # dynamics_seasonal.csv: скор 0.82 < порога детекции — фильтрам
    # нечего отсеивать, вердикт no_growth.
    y = _fixture_series("dynamics_seasonal.csv")
    verdict = classify_growth(y, phrase="новогодние подарки")
    assert not verdict.is_trend
    assert verdict.reason == REASON_NO_GROWTH
    assert not verdict.score.is_growth
    assert verdict.score.score < GROWTH_RATIO_THRESHOLD


# --- Механика фильтра 2 ---------------------------------------------------


def test_growth_outside_peak_arc_skips_peak_analysis():
    # Подтверждённый рост в месяцы вне праздничной дуги (май–июль)
    # сезонного объяснения не имеет: пик-анализ не применяется,
    # peak_ratio/ets_structure остаются None.
    grown = _tiled_seasonal(3, "2026-07")
    grown.iloc[-SCORE_WINDOW:] = grown.iloc[-SCORE_WINDOW:] * 2.0
    verdict = classify_growth(grown)
    assert verdict.reason == REASON_GROWTH_CONFIRMED
    assert verdict.peak_ratio is None
    assert verdict.ets_structure is None


def test_confirmed_month_threshold_is_registered_value():
    # Пороги в тестах сверяются с предрегистрацией, не подбираются.
    assert MONTH_RATIO_THRESHOLD == 1.25
    assert MIN_CONFIRMED_MONTHS == 2
    assert PEAK_GROWTH_RATIO == 1.3

"""Тесты сезонного тайминга закупки (issue #114, фаза 6.1).

Фикстуры: сезонная («новогодние подарки», пик — декабрь),
высокочастотная стабильная («купить телефон», сезонности нет) и
синтетический несезонный ряд. Формулы предрегистрированы в
docs/TRENDS.md; тесты проверяют механику окна, параметры не подбирают.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.sourcing import (
    DEFAULT_LEAD_WEEKS,
    LEAD_TIME_PRESETS,
    lead_weeks_to_months,
    peak_month,
    sourcing_timing,
)
from wordstat_trends.trends.ranking import seasonal_max_min_ratio, seasonal_month_profile

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _flat_series() -> pd.Series:
    """Синтетика: несезонный плоский ряд (амплитуда профиля = 1.0)."""

    return pd.Series(
        np.full(36, 1000.0),
        index=pd.PeriodIndex(pd.period_range("2024-01", periods=36, freq="M"), freq="M"),
        name="queries",
    )


def test_seasonal_fixture_peak_month_december():
    y = _fixture_series("dynamics_seasonal.csv")
    assert peak_month(y) == 12


def test_seasonal_fixture_window_and_on_time():
    # Последняя точка — июль 2026; пик — декабрь; 8 недель → 2 месяца:
    # заказ в октябре, люфт до ноября, до пика 5 месяцев, успеваем.
    y = _fixture_series("dynamics_seasonal.csv")
    timing = sourcing_timing(y, phrase="новогодние подарки")
    assert timing.last_month == pd.Period("2026-07", freq="M")
    window = timing.window
    assert window is not None
    assert window.peak_month == 12
    assert window.order_month == 10
    assert window.latest_order_month == 11
    assert window.lead_months == 2
    assert timing.months_to_peak == 5
    assert timing.slack_months == 3
    assert timing.on_time is True


def test_seasonal_fixture_not_on_time_with_sea_preset():
    # Море 12 недель → 3 месяца lead: заказ в сентябре, люфт до октября,
    # до пика 5 месяцев — успеваем лишь впритык (slack 2); ещё длиннее
    # lead — уже не успеваем.
    y = _fixture_series("dynamics_seasonal.csv")
    timing = sourcing_timing(y, lead_weeks=LEAD_TIME_PRESETS["sea"])
    window = timing.window
    assert window is not None
    assert window.order_month == 9
    assert timing.on_time is True
    late = sourcing_timing(y, lead_weeks=26)  # 6 месяцев
    assert late.on_time is False
    assert late.slack_months == -1


def test_peak_in_last_month_wraps_to_next_year():
    # Последняя точка данных — сам месяц пика: ближайший пик строго
    # впереди, через 12 месяцев.
    idx = pd.PeriodIndex(pd.period_range("2025-01", periods=24, freq="M"), freq="M")
    values = np.where(idx.month == 12, 500.0, 10.0)
    y = pd.Series(values, index=idx, name="queries")
    assert y.index[-1].month == 12
    assert peak_month(y) == 12
    timing = sourcing_timing(y)
    assert timing.months_to_peak == 12


def test_high_freq_fixture_no_seasonality_none_window():
    y = _fixture_series("dynamics_high_freq.csv")
    assert seasonal_max_min_ratio(y) < 2.0
    timing = sourcing_timing(y, phrase="купить телефон")
    assert timing.window is None
    assert timing.months_to_peak is None
    assert timing.slack_months is None
    assert timing.on_time is None


def test_flat_series_none_window():
    y = _flat_series()
    assert seasonal_month_profile(y).nunique() == 1
    timing = sourcing_timing(y)
    assert timing.window is None
    assert timing.on_time is None


def test_peak_month_tie_break_lowest_month():
    # Два месяца с равным максимумом профиля — берётся меньший номер.
    idx = pd.PeriodIndex(pd.period_range("2024-01", periods=36, freq="M"), freq="M")
    values = np.full(36, 100.0)
    for i, p in enumerate(idx):
        if p.month in (3, 9):
            values[i] = 500.0
    y = pd.Series(values, index=idx, name="queries")
    assert peak_month(y) == 3


def test_lead_weeks_rounds_up():
    assert lead_weeks_to_months(4) == 1  # 0.92 → 1
    assert lead_weeks_to_months(8) == 2  # 1.84 → 2
    assert lead_weeks_to_months(13) == 3  # 2.99 → 3


def test_default_lead_weeks_matches_rail_preset():
    assert DEFAULT_LEAD_WEEKS == LEAD_TIME_PRESETS["rail"]

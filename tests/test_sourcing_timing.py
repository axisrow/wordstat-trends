"""Тесты сезонного тайминга закупки (issue #114, фаза 6.1).

Фикстуры — те же, что у ранжирования (#85): сезонная «новогодние
подарки» (пик в декабре, ряд кончается 2026-07) и несезонная
«купить телефон» (амплитуда ниже порога — окна нет). Правила
предрегистрированы в docs/TRENDS.md, тесты их проверяют, параметры
не подбирают.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.sourcing.timing import (
    DEFAULT_LEAD_TIME_WEEKS,
    LEAD_TIME_PRESETS,
    lead_time_months,
    monthly_profile,
    peak_month,
    purchase_window,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _seasonal() -> pd.Series:
    return _fixture_series("dynamics_seasonal.csv")


def test_peak_month_seasonal_fixture_is_december():
    assert peak_month(_seasonal()) == 12


def test_peak_month_flat_series_is_none():
    # Несезонный ряд: пик «найдётся» у любого ряда, поэтому его нет.
    assert peak_month(_fixture_series("dynamics_high_freq.csv")) is None


def test_peak_month_tie_break_smallest_month():
    # Два равных максимума профиля (март и сентябрь) — берётся март.
    pattern = np.array([1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0])
    y = pd.Series(
        np.tile(pattern, 3),
        index=pd.PeriodIndex(pd.period_range("2023-01", periods=36, freq="M"), freq="M"),
        name="queries",
    )
    assert peak_month(y) == 3


def test_peak_month_short_history_is_none():
    # Меньше двух проходов календаря — профиль шумит, окно не выдаётся.
    assert peak_month(_seasonal().iloc[:12]) is None


def test_monthly_profile_indexed_by_calendar_month():
    profile = monthly_profile(_seasonal())
    assert set(profile.index) == set(range(1, 13))
    assert profile.idxmax() == 12


@pytest.mark.parametrize(
    ("weeks", "months"),
    [(4, 1), (8, 2), (12, 3), (52, 12), (1, 1)],  # округление вверх
)
def test_lead_time_months_rounds_up(weeks, months):
    assert lead_time_months(weeks) == months


def test_purchase_window_seasonal_fixture_default_lead():
    # Ряд кончается 2026-07; пик — декабрь; дефолт 8 недель = 2 месяца
    # доставки → заказ в октябре, к пику 2026-12 успеваем.
    window = purchase_window(_seasonal())
    assert window is not None
    assert window.peak_month == 12
    assert window.order_month == 10
    assert window.order_period == pd.Period("2026-10", freq="M")
    assert window.peak_period == pd.Period("2026-12", freq="M")
    assert window.months_to_order == 3
    assert window.months_to_peak == 5
    assert window.lead_time_months == 2
    assert window.on_time is True


def test_purchase_window_long_lead_shifts_order_month():
    # 12 недель = 3 месяца: заказ в сентябре.
    window = purchase_window(_seasonal(), lead_time_weeks=12)
    assert window is not None
    assert window.order_month == 9
    assert window.order_period == pd.Period("2026-09", freq="M")


def test_purchase_window_wrap_around_year():
    # Годовая намотка: пик в феврале, поставка 3 месяца → заказ в ноябре.
    pattern = np.array([1.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = pd.Series(
        np.tile(pattern, 3),
        index=pd.PeriodIndex(pd.period_range("2023-01", periods=36, freq="M"), freq="M"),
        name="queries",
    )
    window = purchase_window(y, lead_time_weeks=13)  # 13 недель = 3 месяца
    assert window is not None
    assert window.order_month == 11
    assert window.peak_month == 2


def _series_ending(end: str, months: int, peak: int) -> pd.Series:
    """Синтетика: значения по номеру месяца, пик `peak`, конец ряда `end`."""

    index = pd.PeriodIndex(pd.period_range(end=pd.Period(end, freq="M"), periods=months, freq="M"), freq="M")
    values = [20.0 if m == peak else 1.0 for m in index.month]
    return pd.Series(values, index=index, name="queries")


def test_purchase_window_late_for_nearest_peak():
    # Ряд кончается ноябрём 2025: до пика 2025-12 один месяц, поставка
    # 2 месяца — к ближайшему пику не успеваем.
    window = purchase_window(_series_ending("2025-11", 25, peak=12))
    assert window is not None
    assert window.months_to_peak == 1
    assert window.on_time is False


def test_purchase_window_order_month_already_here():
    # Ряд кончается октябрём 2025 — месяц заказа настал: months_to_order=0.
    window = purchase_window(_series_ending("2025-10", 26, peak=12))
    assert window is not None
    assert window.months_to_order == 0
    assert window.order_period == pd.Period("2025-10", freq="M")


def test_purchase_window_flat_series_is_none():
    assert purchase_window(_fixture_series("dynamics_high_freq.csv")) is None


def test_lead_time_presets_cover_weeks_to_months_range():
    # Пресеты — недели и месяцы эпика: 4..12 недель = 1..3 месяца.
    assert set(LEAD_TIME_PRESETS) == {"air", "rail", "sea"}
    assert DEFAULT_LEAD_TIME_WEEKS == LEAD_TIME_PRESETS["rail"]
    assert {lead_time_months(w) for w in LEAD_TIME_PRESETS.values()} == {1, 2, 3}

"""Окно сбора по умолчанию (issue #5): пять лет вместо платформенных 24 мес.

Проверяются рамки интерфейса Вордстата (валидация до браузера) и раскладка
дефолтного окна на полные прошедшие месяцы.
"""

from datetime import date

import pytest

from wordstat_trends.collect.config import (
    DEFAULT_GRANULARITY,
    HISTORY_FLOOR,
    MAX_WINDOW_MONTHS,
    DynamicsWindow,
    default_window,
)
from wordstat_trends.collect.errors import WindowRangeError


def test_default_window_is_five_full_months():
    """60 полных прошедших месяцев: конец — последний день предыдущего месяца."""
    window = default_window(today=date(2026, 9, 8))
    assert window.date_from == date(2021, 9, 1)
    assert window.date_to == date(2026, 8, 31)
    assert window.granularity == DEFAULT_GRANULARITY == "monthly"
    window.validate(today=date(2026, 9, 8))  # дефолт обязан проходить собственную валидацию


def test_default_window_spans_five_seasonal_cycles():
    """Смысл дефолта (#5): 60 точек ≈ 5 годовых циклов — не 24 месяца и 2 цикла."""
    window = default_window(today=date(2026, 9, 8))
    months = (window.date_to.year - window.date_from.year) * 12 + window.date_to.month - window.date_from.month + 1
    assert months == 60
    assert months > 24  # платформенный дефолт, недостаточный для sp=12


def test_cli_flags_reachable_wordstat_collect():
    window = DynamicsWindow(date_from=date(2021, 9, 1), date_to=date(2026, 8, 31))
    assert window.cli_flags() == [
        "--granularity",
        "monthly",
        "--date-from",
        "2021-09-01",
        "--date-to",
        "2026-08-31",
    ]


def test_rejects_before_history_floor():
    with pytest.raises(WindowRangeError, match="границы истории"):
        DynamicsWindow(date_from=date(2017, 12, 1), date_to=date(2022, 11, 30)).validate()


def test_rejects_window_shorter_than_three_months():
    with pytest.raises(WindowRangeError, match="минимум интерфейса"):
        DynamicsWindow(date_from=HISTORY_FLOOR, date_to=date(2018, 2, 28)).validate()


def test_rejects_window_longer_than_five_years():
    with pytest.raises(WindowRangeError, match="максимум интерфейса"):
        DynamicsWindow(date_from=HISTORY_FLOOR, date_to=date(2024, 2, 29)).validate()


def test_rejects_end_before_start():
    with pytest.raises(WindowRangeError, match="позже"):
        DynamicsWindow(date_from=date(2024, 1, 1), date_to=date(2023, 1, 1)).validate()


def test_rejects_end_in_future():
    with pytest.raises(WindowRangeError, match="в будущем"):
        DynamicsWindow(date_from=date(2020, 1, 1), date_to=date(2999, 1, 1)).validate(today=date(2026, 9, 8))


def test_max_window_month_constant_matches_interface_limit():
    assert MAX_WINDOW_MONTHS == 60  # «до пяти лет» по справке Яндекса

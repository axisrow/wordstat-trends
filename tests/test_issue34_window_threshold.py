"""Тесты для scripts/issue34_window_threshold.py (issue #34)."""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_vector_d import check_weekday_balance, load_daily_dynamics_csv
from scripts.issue34_window_threshold import (
    BASE_WINDOW_DAYS,
    CANDIDATE_SP,
    FIXTURES,
    WINDOWS,
    calibration_synth,
    detect_sktime_detector,
    suffix_window,
    verdict_from_sp,
)


@pytest.fixture(scope="module")
def seasonal_base() -> pd.Series:
    full = load_daily_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    return full.iloc[:BASE_WINDOW_DAYS]


@pytest.mark.parametrize("days", WINDOWS)
def test_suffix_windows_share_end_date_and_are_balanced(seasonal_base: pd.Series, days: int):
    window = suffix_window(seasonal_base, days)

    # Все окна заканчиваются одной датой (якорение суффиксами, issue #34) и
    # вкладываются друг в друга.
    assert len(window) == days
    assert window.index[-1] == seasonal_base.index[-1]
    assert window.index[0] == seasonal_base.index[len(seasonal_base) - days]
    # Кратность 7 -> каждый день недели ровно days/7 раз (check_weekday_balance
    # дополнительно требует непрерывности и уникальности дат).
    balance = check_weekday_balance(window)
    assert balance["balanced"], balance


def test_suffix_window_rejects_non_multiple_of_sp(seasonal_base: pd.Series):
    with pytest.raises(ValueError, match="кратность"):
        suffix_window(seasonal_base, 27)


def test_suffix_window_rejects_too_long_window(seasonal_base: pd.Series):
    with pytest.raises(ValueError, match="некорректно"):
        suffix_window(seasonal_base, len(seasonal_base) + 7)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (7, "да"),
        (4, "нет"),  # детектор вернул sp=4 — сезонность с периодом 7 НЕ обнаружена
        (1, "нет"),
        (8, "нет"),  # соседний период 8 — тоже «нет» (разрешение периодограммы)
        (None, "нет"),  # fail-closed: отсутствие значения не молчаливый пропуск
    ],
)
def test_verdict_from_sp(raw, expected):
    verdict, _ = verdict_from_sp(raw)
    assert verdict == expected


def test_verdict_from_sp_nan_is_fail_closed():
    # docs/MODELING.md: NaN в вердиктной ячейке обязан читаться как «нет»,
    # а не проваливаться сквозь сравнение (nan == 7 -> False молча).
    verdict, raw = verdict_from_sp(float("nan"))
    assert verdict == "нет"
    assert isinstance(raw, float) and math.isnan(raw)


def test_candidate_sp_is_exactly_seven():
    # Предрегистрированный параметр прогонов (issue #34): вопрос — вердикт по
    # sp=7, широкий список кандидатов размывает вердикт и обрезается nlags.
    assert CANDIDATE_SP == [7]


def test_qstat_detects_s7_on_ideal_sine_56():
    # Согласование с калибровкой issue: на 56-дневной идеальной синусоиде
    # периода 7 Qstat обязан вернуть sp=7 (детектор, не платящий параметрами).
    t = np.arange(56)
    y = pd.Series(10.0 + 4.0 * np.sin(2 * np.pi * t / 7))
    result = detect_sktime_detector(y, "qstat")
    assert result["verdict"] == "да"
    assert result["raw_sp"] == 7


def test_calibration_synth_reproduces_reference_thresholds():
    # Воспроизведение калибровки issue #34 (расхождение порогов ACF/Qstat на
    # одно окно зафиксировано в docs/ISSUE_34_WINDOW_THRESHOLD.md): Qstat = 7
    # на всех длинах, ACF = 7 с 42, Periodogram чередует 7 и соседние периоды.
    rows = calibration_synth()
    assert sorted(int(k) for k in rows) == WINDOWS
    for n, cell in rows.items():
        assert cell["qstat"]["verdict"] == "да"
        assert cell["acf"]["verdict"] == ("нет" if int(n) < 42 else "да")

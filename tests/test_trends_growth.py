"""Тесты детекции аномального роста (issue #83, фаза 3 Б1).

Пороги и окно предрегистрированы в docs/TRENDS.md; тесты проверяют
синтетические свойства (стабильный ряд ≈ 1, инъекция роста детектируется)
и поведение на фикстурах, но не подбирают параметры.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.forecasting.baseline import SP, to_monthly_series
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.trends.growth import (
    GROWTH_RATIO_THRESHOLD,
    MIN_HISTORY,
    SCORE_WINDOW,
    GrowthScore,
    InsufficientHistoryError,
    score_frame,
    score_growth,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def test_seasonal_fixture_stable_no_growth():
    # «новогодние подарки»: жёсткая сезонность повторяется год в год,
    # спад 2026 против 2025 — скор < 1, не рост.
    result = score_growth(_fixture_series("dynamics_seasonal.csv"), phrase="новогодние подарки")
    assert isinstance(result, GrowthScore)
    assert result.score == pytest.approx(0.82, abs=0.03)
    assert not result.is_growth


def test_high_freq_stable_is_not_trend():
    # «купить телефон»: миллионные объёмы, но относительно собственной
    # истории — лишь умеренный подъём (июнь–июль 2026 выше прошлого года),
    # скор ≈ 1.2 < порога, тренда нет. Именно этот случай охраняет
    # нормировка «относительно себя», а не популяции: абсолютная
    # частотность в скор не входит.
    result = score_growth(_fixture_series("dynamics_high_freq.csv"), phrase="купить телефон")
    assert result.score == pytest.approx(1.19, abs=0.03)
    assert not result.is_growth


def _flat_seasonal_series() -> pd.Series:
    """Синтетика: сезонный профиль, повторённый год в год без роста —
    сезонный наив на таком ряде точен, скор базы строго 1.0."""

    pattern = _fixture_series("dynamics_seasonal.csv").iloc[:12].to_numpy()
    return pd.Series(
        np.tile(pattern.astype(float), 3),
        index=pd.PeriodIndex(
            pd.period_range("2024-01", periods=36, freq="M"), freq="M"
        ),
        name="queries",
    )


def test_flat_seasonal_base_score_is_one():
    result = score_growth(_flat_seasonal_series(), phrase="плоская сезонная")
    assert result.score == pytest.approx(1.0)
    assert not result.is_growth


def test_injected_growth_detected():
    # Синтетика: устойчивый рост ×2 во всём окне скоринга → скор строго 2
    # (прогноз берётся из неискажённой истории год назад).
    boosted = _flat_seasonal_series()
    boosted.iloc[-SCORE_WINDOW:] = boosted.iloc[-SCORE_WINDOW:] * 2.0
    result = score_growth(boosted, phrase="растущая")
    assert result.score == pytest.approx(2.0)
    assert result.is_growth
    assert result.score >= GROWTH_RATIO_THRESHOLD


def test_window_fields_match_tail():
    y = _fixture_series("dynamics_high_freq.csv")
    result = score_growth(y, phrase="купить телефон", window=4)
    assert result.window == tuple(y.index[-4:])
    assert list(result.actual.index) == [p for p in result.window]
    # Прогноз окна — значения ровно 12 месяцев назад (сезонный наив).
    assert list(result.forecast.to_numpy()) == list(y.iloc[-16:-12].astype(float).to_numpy())
    assert result.score == pytest.approx(
        float(result.actual.mean() / result.forecast.mean()), rel=1e-9
    )


def test_single_month_spike_not_growth():
    # Разовый всплеск одного месяца (сигнал #22) на стабильной базе окно
    # 3 месяца отсекает: +один удвоенный месяц к сумме окна держит скор
    # (отношение сумм) ниже порога.
    spiked = _flat_seasonal_series()
    last_month = float(spiked.iloc[-1])
    window_sum = float(spiked.iloc[-SCORE_WINDOW:].sum())
    spiked.iloc[-1] = spiked.iloc[-1] * 2.0
    expected = (window_sum + last_month) / window_sum
    result = score_growth(spiked, phrase="всплеск")
    assert result.score == pytest.approx(expected)
    assert result.score < GROWTH_RATIO_THRESHOLD
    assert not result.is_growth


def test_short_history_rejected():
    y = _fixture_series("dynamics_seasonal.csv").iloc[: MIN_HISTORY - 1]
    with pytest.raises(InsufficientHistoryError, match="MIN_HISTORY"):
        score_growth(y)


def test_zero_forecast_rejected():
    y = _fixture_series("dynamics_seasonal.csv").astype(float)
    y.iloc[-SCORE_WINDOW - 12 : -SCORE_WINDOW] = 0.0  # нулевой год назад
    with pytest.raises(ValueError, match="прогноз"):
        score_growth(y)


def test_negative_window_rejected():
    y = _fixture_series("dynamics_seasonal.csv")
    with pytest.raises(ValueError, match="окно скоринга"):
        score_growth(y, window=0)


def test_window_eating_season_rejected():
    # Окно, съедающее train до короче сезона: NaiveForecaster(sp=12) молча
    # взял бы весь train как «сезонный профиль» — громкий отказ вместо
    # тихого прогноза не из «12 месяцев назад».
    y = _fixture_series("dynamics_seasonal.csv")
    with pytest.raises(ValueError, match="короче сезона"):
        score_growth(y, window=len(y) - SP + 1)
    # Максимально допустимое окно (train ровно в один сезон) — валидно.
    score_growth(y, window=len(y) - SP)


def test_nan_values_rejected():
    y = _fixture_series("dynamics_seasonal.csv").astype(float)
    y.iloc[5] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        score_growth(y)


def test_actual_series_are_floats_and_finite():
    result = score_growth(_fixture_series("dynamics_seasonal.csv"))
    assert np.isfinite(result.score)
    assert result.forecast.dtype == float
    assert result.actual.dtype == float


def test_score_frame_takes_phrase_from_attrs():
    rows = parse_dynamics_rows((FIXTURES / "dynamics_seasonal.csv").read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    frame.attrs["phrase"] = "новогодние подарки"
    via_frame = score_frame(frame)
    via_series = score_growth(to_monthly_series(frame), phrase="новогодние подарки")
    assert via_frame.phrase == "новогодние подарки"
    assert via_frame.score == pytest.approx(via_series.score)
    assert via_frame.is_growth == via_series.is_growth

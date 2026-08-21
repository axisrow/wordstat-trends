"""Вектор недельной сезонности `sp=7` двумя независимыми бэкендами.

Гипотеза #23 просит вектор параметров модели: уровень, тренд и сезонный
профиль. Здесь профиль недельный, снимается с дневного ряда, и снимается
**дважды** — sktime и statsforecast, — потому что на первой итерации
(месячной, `sp=12`) именно расхождение бэкендов оказалось главным
результатом: statsforecast отвергал сезонность по AIC там, где sktime её
показывал. Согласие двух независимых реализаций — часть доказательства, а
не формальность.

Спецификация модели нигде не задаётся руками: оба бэкенда выбирают её сами
(`auto=True` / автоперебор), выбранная фиксируется и попадает в отчёт.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .daily_window import SP

WEEKDAY_NAMES = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


@dataclass(frozen=True)
class WeeklyVector:
    """Вектор параметров: уровень, тренд и недельный профиль."""

    level: float
    trend: float
    seasonal: np.ndarray
    """7 мультипликативных коэффициентов, индекс 0 — понедельник."""

    spec: str
    """Спецификация, выбранная бэкендом самостоятельно."""

    aic: float | None = None
    seasonal_chosen: bool = True
    """Включил ли бэкенд сезонную компоненту вообще."""

    def as_array(self) -> np.ndarray:
        return np.concatenate(([self.level, self.trend], self.seasonal))

    @property
    def amplitude(self) -> float:
        """СКО профиля — сколько в нём вообще формы."""
        return float(self.seasonal.std())

    @property
    def peak_day(self) -> str:
        return WEEKDAY_NAMES[int(np.argmax(self.seasonal))]


def _sktime_to_calendar(values: np.ndarray, last_day: pd.Timestamp) -> np.ndarray:
    """sktime: состояния по одному на наблюдение, последнее — последний день окна."""
    calendar = np.empty(SP, dtype=float)
    for offset, value in enumerate(values):
        day = last_day - pd.Timedelta(days=len(values) - 1 - offset)
        calendar[day.dayofweek] = value
    return calendar


def _statsforecast_to_calendar(values: np.ndarray, first_day: pd.Timestamp) -> np.ndarray:
    """statsforecast: начальные сезонные состояния в ОБРАТНОМ порядке.

    `values[0]` относится к дню, непосредственно предшествующему первому
    наблюдению, `values[1]` — к предыдущему, и так далее. Порядок и якорь
    установлены эмпирически: сверкой с эмпирическими средними по дням недели
    на реальных фикстурах (корреляция 0.990 и 1.000 против −0.45 при
    «прямом» прочтении).

    Без этого профили двух бэкендов сравнивались бы повёрнутыми друг
    относительно друга, и мнимое «расхождение бэкендов» оказалось бы
    артефактом кода, а не измеренным результатом.
    """
    calendar = np.empty(SP, dtype=float)
    for offset, value in enumerate(values):
        day = first_day - pd.Timedelta(days=offset + 1)
        calendar[day.dayofweek] = value
    return calendar


def _normalise(seasonal: np.ndarray, *, multiplicative: bool, level: float) -> np.ndarray:
    """Приводит профиль к сопоставимому мультипликативному виду со средним 1.

    ETS не нормирует состояния сам: уровень и сезонность делят общий масштаб.
    Мультипликативный профиль достаточно поделить на среднее.

    Аддитивный делить на среднее **нельзя**: его коэффициенты в сумме около
    нуля, и деление на почти нулевое среднее даёт бессмысленный взрыв
    (замерено: амплитуда 5588 у заведомо плоской фразы). Аддитивный профиль
    переводится в мультипликативный через уровень: `1 + (s − s̄) / level`.
    Это делает профили двух бэкендов сравнимыми, даже когда они выбрали
    разную форму сезонности.
    """
    if multiplicative:
        return seasonal / seasonal.mean()
    return 1.0 + (seasonal - seasonal.mean()) / abs(level)


def sktime_vector(series: pd.Series) -> WeeklyVector:
    """`AutoETS(sp=7, auto=True)` из sktime — спецификацию выбирает сам."""
    from sktime.forecasting.ets import AutoETS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster = AutoETS(auto=True, sp=SP, n_jobs=1)
        forecaster.fit(series.astype(float))

    fitted = forecaster._fitted_forecaster
    seasonal_chosen = fitted.seasonal is not None
    if seasonal_chosen:
        raw = np.asarray(fitted.states["seasonal"][-SP:], dtype=float)
        seasonal = _normalise(
            _sktime_to_calendar(raw, series.index[-1]),
            multiplicative=fitted.seasonal == "mul",
            level=float(fitted.states["level"].iloc[-1]),
        )
    else:
        # Сезонности нет — профиль плоский по построению, и это факт о модели,
        # а не повод подставить что-то похожее на сезонность.
        seasonal = np.ones(SP)

    trend_states = fitted.states.get("trend")
    return WeeklyVector(
        level=float(fitted.states["level"].iloc[-1]),
        trend=float(trend_states.iloc[-1]) if trend_states is not None else 0.0,
        seasonal=seasonal,
        spec=_spec_letters(fitted),
        aic=float(fitted.aic),
        seasonal_chosen=seasonal_chosen,
    )


def _spec_letters(fitted) -> str:
    error = fitted.error[0].upper()
    if fitted.trend is None:
        trend = "N"
    else:
        trend = fitted.trend[0].upper() + ("d" if fitted.damped_trend else "")
    seasonal = "N" if fitted.seasonal is None else fitted.seasonal[0].upper()
    return f"ETS({error},{trend},{seasonal})"


def statsforecast_vector(series: pd.Series) -> WeeklyVector:
    """`AutoETS(season_length=7)` из statsforecast — независимая реализация."""
    from statsforecast.models import AutoETS as StatsForecastAutoETS

    values = series.astype(float).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = StatsForecastAutoETS(season_length=SP)
        model.fit(values)

    fitted = model.model_
    # `components` — четыре буквы: ошибка, тренд, сезонность, damped.
    components = str(fitted["components"])
    error, trend, seasonal_kind = components[0], components[1], components[2]
    seasonal_chosen = seasonal_kind != "N"

    states = np.asarray(fitted["states"], dtype=float)
    if seasonal_chosen:
        seasonal = _normalise(
            _statsforecast_to_calendar(states[0, -SP:], series.index[0]),
            multiplicative=seasonal_kind == "M",
            level=float(states[-1, 0]),
        )
    else:
        seasonal = np.ones(SP)

    level = float(states[-1, 0])
    trend_value = float(states[-1, 1]) if trend != "N" else 0.0
    return WeeklyVector(
        level=level,
        trend=trend_value,
        seasonal=seasonal,
        spec=f"ETS({error},{trend},{seasonal_kind})",
        aic=float(np.ravel(fitted["aic"])[0]),
        seasonal_chosen=seasonal_chosen,
    )


BACKENDS = {"sktime": sktime_vector, "statsforecast": statsforecast_vector}

"""Вектор параметров сезонной модели как представление спроса (гипотеза #23).

Два оценивателя сезонного профиля, сознательно разные:

`autoets_vector`
    То, что просит гипотеза: `AutoETS(sp=12)` из sktime — уровень, тренд и
    12 сезонных коэффициентов. Работает **только** на полных 24 точках:
    statsmodels инициализирует сезонность эвристикой, которой нужно два
    полных цикла, и на 21/18 точках падает с `ValueError`. Это измеренный
    факт, см. `docs/EXPERIMENT_VECTOR_C.md`.

`seasonal_index`
    Детерминированный сезонный профиль: средние по календарному месяцу на
    детрендированном логарифме ряда. Та же величина по смыслу
    (мультипликативный профиль из 12 коэффициентов со средним 1), но
    определена на любом ряде от года длиной и — в отличие от скользящего
    среднего — покрывает все 12 месяцев и на 21, и на 18 точках. Нужен,
    чтобы проверка устойчивости из пункта 2 гипотезы вообще была исполнима.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

SP = 12
"""Годовой период на месячном ряде."""


@dataclass(frozen=True)
class DemandVector:
    """Вектор параметров: уровень, тренд и сезонный профиль."""

    level: float
    trend: float
    seasonal: np.ndarray
    """12 мультипликативных коэффициентов, индекс 0 — январь."""

    spec: str
    """Спецификация оценивателя (для AutoETS — выбранная модель ETS)."""

    def as_array(self) -> np.ndarray:
        return np.concatenate(([self.level, self.trend], self.seasonal))


def autoets_vector(series: pd.Series, sp: int = SP) -> DemandVector:
    """Снимает вектор через `AutoETS` sktime.

    Падает `ValueError`, если в ряде меньше двух полных сезонных циклов —
    это поведение statsmodels, и мы его намеренно не глушим.
    """
    from sktime.forecasting.ets import AutoETS

    forecaster = AutoETS(auto=True, sp=sp, n_jobs=1)
    with warnings.catch_warnings():
        # Оптимизатор шумит на коротком ряде; сам факт короткости мы
        # фиксируем отдельно, предупреждения здесь ничего не добавляют.
        warnings.simplefilter("ignore")
        forecaster.fit(series.astype(float))

    fitted = forecaster._fitted_forecaster
    spec = f"ETS({fitted.error[0].upper()},{_trend_letter(fitted)},{_seasonal_letter(fitted)})"

    seasonal = np.asarray(fitted.states["seasonal"][-sp:], dtype=float)
    # Состояния идут в порядке ряда; переставляем в календарный (январь = 0).
    seasonal = _to_calendar_order(seasonal, last_period=series.index[-1], sp=sp)
    # ETS не нормирует сезонные состояния: уровень и сезонность делят между
    # собой общий масштаб, и среднее профиля у разных фраз получается разным
    # (замерено: 1.22 / 0.85 / 1.02). Нормируем к среднему 1, иначе профили
    # несопоставимы ни между фразами, ни между пересчётами.
    seasonal = seasonal / seasonal.mean()

    level = float(fitted.states["level"].iloc[-1])
    trend_states = fitted.states.get("trend")
    trend = float(trend_states.iloc[-1]) if trend_states is not None else 0.0

    return DemandVector(level=level, trend=trend, seasonal=seasonal, spec=spec)


def _trend_letter(fitted) -> str:
    if fitted.trend is None:
        return "N"
    letter = fitted.trend[0].upper()
    return f"{letter}d" if fitted.damped_trend else letter


def _seasonal_letter(fitted) -> str:
    return "N" if fitted.seasonal is None else fitted.seasonal[0].upper()


def _to_calendar_order(values: np.ndarray, last_period: pd.Period, sp: int) -> np.ndarray:
    """Переставляет sp коэффициентов так, чтобы индекс 0 был январём."""
    calendar = np.empty(sp, dtype=float)
    for offset, value in enumerate(values):
        # values[-1] относится к последнему периоду ряда.
        month = (last_period - (len(values) - 1 - offset)).month
        calendar[month - 1] = value
    return calendar


def seasonal_index(series: pd.Series, sp: int = SP) -> np.ndarray:
    """Мультипликативный сезонный профиль через средние по календарному месяцу.

    Ряд логарифмируется, из него вычитается глобальный линейный тренд, остатки
    усредняются по календарному месяцу и возвращаются в исходный масштаб
    (`exp`) с нормировкой к среднему 1.

    Почему не отношение к центрированному скользящему среднему — классический
    сезонный индекс: центрированное СС длиной `sp` съедает по полгода с каждого
    конца ряда. На 24 точках это оставляет ровно 12 отношений (все 12 месяцев),
    но на 21 точке — 9 месяцев, а на 18 — шесть, причём выпадают ноябрь и
    декабрь, то есть **ровно те месяцы, которые несут сигнал** у сезонной фразы.
    Сравнение профилей после такого укорочения меряет не устойчивость спроса, а
    то, чем заполнены дыры. Здесь же вклад даёт каждое наблюдение, и все 12
    месяцев покрыты и на 24, и на 21, и на 18 точках (проверено, см. отчёт).

    Требует хотя бы `sp` точек. Возвращает `sp` коэффициентов, индекс 0 —
    январь.
    """
    if len(series) < sp:
        raise ValueError(f"Нужно минимум {sp} точек, получено {len(series)}")

    profile, counts = _month_means(series, sp)
    empty = int(np.isnan(profile).sum())
    if empty:
        raise ValueError(f"{empty} календарных месяцев без наблюдений — профиль не определён")
    return profile / profile.mean()


def month_counts(series: pd.Series, sp: int = SP) -> np.ndarray:
    """Сколько наблюдений пришлось на каждый календарный месяц (январь = 0).

    Профиль, у которого месяц опирается на одно наблюдение, слабее того, где
    их два, — и отчёт обязан показывать это рядом с числами.
    """
    return _month_means(series, sp)[1]


def _month_means(series: pd.Series, sp: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.log(series.astype(float).to_numpy())
    positions = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(positions, values, 1)
    residuals = values - (intercept + slope * positions)

    months = np.array([period.month for period in series.index])
    profile = np.full(sp, np.nan)
    counts = np.zeros(sp, dtype=int)
    for month in range(1, sp + 1):
        selected = residuals[months == month]
        counts[month - 1] = selected.size
        if selected.size:
            profile[month - 1] = np.exp(selected.mean())
    return profile, counts

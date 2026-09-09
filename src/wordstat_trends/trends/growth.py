"""Детекция аномального роста месячного ряда (issue #83, фаза 3 Б1).

Скор аномальности = средний факт окна скоринга / средний прогноз
сезонного наива на то же окно. Сезонный наив берётся из
:mod:`wordstat_trends.forecasting.baseline` — тот же бейзлайн, что и в
фазе 2, без второй реализации.

Границы задачи (issue #83):

- Отдельная библиотека детекции аномалий не нужна (docs/REFERENCES.md,
  «Позиция проекта»): скор из прогноза интерпретируем и не тянет
  зависимость — принципиальное решение, не экономия.
- Скор нормирован относительно собственной истории фразы, а не популяции:
  высокочастотная стабильная фраза — не тренд, низкочастотная растущая —
  кандидат.
- Выход — структура данных без рендера: витрина — фаза #14.

Параметры ``SCORE_WINDOW``, ``GROWTH_RATIO_THRESHOLD``, ``MIN_HISTORY``
предрегистрированы в docs/TRENDS.md до прогона на данных.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from sktime.forecasting.base import BaseForecaster

from wordstat_trends.forecasting.baseline import SP, seasonal_naive, to_monthly_series

#: Окно скоринга: сколько последних месяцев сравниваются с прогнозом.
#: 1 месяц шумен (разовый всплеск #22 даёт ложный сигнал), длинное окно
#: размывает старт роста. Предрегистрировано в docs/TRENDS.md.
SCORE_WINDOW = 3

#: Порог «рост есть»: средний факт окна / средний прогноз >= 1.5 —
#: «в полтора раза чаще, чем год назад». Предрегистрирован в docs/TRENDS.md.
GROWTH_RATIO_THRESHOLD = 1.5

#: Минимальная длина ряда: два полных сезонных цикла — меньше сезонному
#: наиву не с чем сравнивать (обучающая история + само окно).
MIN_HISTORY = 2 * SP


class InsufficientHistoryError(ValueError):
    """Ряд короче MIN_HISTORY — сезонному наиву не хватает истории."""


@dataclass(frozen=True)
class GrowthScore:
    """Скор аномального роста одной фразы.

    ``forecast``/``actual`` — помесячные ряды окна скоринга (прогноз
    сезонного наива и факт), ``score`` — отношение их средних,
    ``window`` — календарные месяцы окна.
    """

    phrase: str
    score: float  # средний факт окна / средний прогноз окна
    is_growth: bool  # score >= GROWTH_RATIO_THRESHOLD
    window: tuple[pd.Period, ...]
    forecast: pd.Series
    actual: pd.Series


def score_growth(
    y: pd.Series,
    phrase: str = "",
    window: int = SCORE_WINDOW,
    forecaster: BaseForecaster | None = None,
) -> GrowthScore:
    """Месячный ряд фразы → скор аномального роста.

    ``y`` — месячный ряд из :func:`to_monthly_series` (непрерывность уже
    проверена). Хвост ``window`` месяцев — окно скоринга, всё раньше —
    обучающая история; сезонный наив прогнозирует окно, скор — отношение
    среднего факта к среднему прогнозу.
    """

    if window < 1:
        raise ValueError(f"окно скоринга должно быть >= 1, получено {window}")
    if len(y) < MIN_HISTORY:
        raise InsufficientHistoryError(
            f"ряд {phrase!r}: {len(y)} месяцев < MIN_HISTORY={MIN_HISTORY} "
            f"(нужны два полных сезонных цикла)"
        )

    train, actual = y.iloc[:-window], y.iloc[-window:]
    if forecaster is None:
        forecaster = seasonal_naive()
    forecaster.fit(train)
    # Стаб sktime типизирует predict как union с кортежем — на месячном
    # Series с fh он всегда возвращает Series.
    forecast = cast(pd.Series, forecaster.predict(list(range(1, window + 1)))).astype(float)

    forecast_mean = float(forecast.mean())
    actual_mean = float(actual.astype(float).mean())
    if forecast_mean <= 0:
        raise ValueError(
            f"ряд {phrase!r}: нулевой или отрицательный прогноз окна — "
            f"скор факт/прогноз не определён"
        )
    score = actual_mean / forecast_mean

    return GrowthScore(
        phrase=phrase,
        score=score,
        is_growth=score >= GROWTH_RATIO_THRESHOLD,
        window=tuple(y.index[-window:]),
        forecast=forecast,
        actual=actual.astype(float),
    )


def score_frame(frame: pd.DataFrame, phrase: str | None = None) -> GrowthScore:
    """DataFrame загрузчика (``period``/``queries``) → GrowthScore.

    Обёртка «одним вызовом»: ``load_run`` → ``to_monthly_series`` →
    :func:`score_growth`. Если фраза не передана, берётся из
    ``frame.attrs["phrase"]`` (заполняется загрузчиком), иначе пустая.
    """

    resolved = frame.attrs.get("phrase") if phrase is None else phrase
    return score_growth(to_monthly_series(frame), phrase=resolved if resolved is not None else "")


__all__ = [
    "GROWTH_RATIO_THRESHOLD",
    "MIN_HISTORY",
    "SCORE_WINDOW",
    "GrowthScore",
    "InsufficientHistoryError",
    "score_frame",
    "score_growth",
]

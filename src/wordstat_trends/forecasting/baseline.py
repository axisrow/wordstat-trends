"""Сезонный наивный бейзлайн и каркас валидации (issue #11, фаза 2).

Порядок фазы принципиален (см. #11): бейзлайн и валидация раньше любых
моделей. Пока нет честного сезонного наива и фиксированного протокола
оценки, любые числа моделей неинтерпретируемы — не с чем сравнивать.

Состав:

- :func:`seasonal_naive` — ``NaiveForecaster(strategy="last", sp=12)``:
  прогноз на следующий месяц = значение ровно 12 месяцев назад;
- :func:`make_expanding_splitter` — ``ExpandingWindowSplitter`` **из
  ``sktime.split``** (не из ``sktime.forecasting.model_selection`` — тот
  путь дублирует сплиттеры до sktime 1.x и считается легаси);
- :func:`evaluate_forecaster` — обёртка над
  ``sktime.forecasting.model_evaluation.evaluate`` с метрикой MASE
  (``MeanAbsoluteScaledError``): читается как «в N раз точнее наивной»,
  MASE > 1 — модель проигрывает наивному бейзлайну.

Никакого shuffle: сплиттер расширяющимся окном сохраняет хронологию,
обучение всегда строго раньше теста. Горизонт и шаг кратны 12, чтобы фолды
не смешивали месяцы разных сезонов.
"""

from __future__ import annotations

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sktime.split import ExpandingWindowSplitter

#: Годовая сезонность месячного ряда (полный ряд после склейки #6 — от
#: января 2018, ~104 точки ≈ 8 циклов).
SP = 12

#: Сколько точек занимает тестовый хвост при бэктесте: один полный цикл.
TEST_LENGTH = 12

#: Начальное обучающее окно: 6 полных циклов — меньше некуда для sp=12.
INITIAL_WINDOW = 72


class SeriesContinuityError(ValueError):
    """В ряду есть пропуск или дубль месяца — сезонный sp=12 нельзя доверять."""


def to_monthly_series(frame: pd.DataFrame) -> pd.Series:
    """DataFrame загрузчика → месячный ``pd.Series`` с ``PeriodIndex``.

    ``load_run`` отдаёт колонки ``period`` (``YYYY-MM``) и ``queries``;
    sktime требует индекс периода — иначе сплиттер примет целочисленную
    ось за позиционную и ``sp=12`` потеряет привязку к календарю.

    Непрерывность обязательна (как и в склейке #6): дыра или дубль месяца
    молча ломают сезонный наив — ``sp=12`` арифметически привязан к
    индексу, прогноз возьмётся не за тот календарный месяц. Поэтому здесь
    ``SeriesGapError``-образная явная ошибка, а не тихий сдвиг.
    """

    index = pd.PeriodIndex(frame["period"], freq="M")
    # is_monotonic_increasing нестрогий — дубль месяца ловится отдельно.
    if not index.is_full or not index.is_monotonic_increasing or not index.is_unique:
        raise SeriesContinuityError(
            f"ряд не является непрерывным месячным (пропуск/дубль месяца): "
            f"{frame['period'].iloc[0]}..{frame['period'].iloc[-1]}, строк {len(index)}"
        )
    return pd.Series(frame["queries"].to_numpy(), index=index, name="queries")


def seasonal_naive() -> NaiveForecaster:
    """Сезонный наивный бейзлайн: значение ровно ``sp=12`` месяцев назад."""

    return NaiveForecaster(strategy="last", sp=SP)


def make_expanding_splitter(
    initial_window: int = INITIAL_WINDOW,
    test_length: int = TEST_LENGTH,
    step_length: int = SP,
) -> ExpandingWindowSplitter:
    """Расширяющееся окно валидации: хронология, без shuffle.

    ``step_length`` по умолчанию равен ``SP``: фолды сдвигаются на целый
    сезонный цикл, каждый фолд тестирует один и тот же календарный срез
    относительно длины истории.
    """

    # fh списком 1..test_length: sktime сам соберёт ForecastingHorizon.
    return ExpandingWindowSplitter(  # type: ignore[call-arg] — стаб sktime зажимает fh в int
        fh=list(range(1, test_length + 1)),
        initial_window=initial_window,
        step_length=step_length,
    )


def evaluate_forecaster(
    forecaster: BaseForecaster,
    y: pd.Series,
    splitter: ExpandingWindowSplitter | None = None,
) -> pd.DataFrame:
    """Прогнать форкастер по фолдам и посчитать MASE на каждом.

    Возвращает сырой результат ``sktime...evaluate``: по строке на фолд
    (``cutoff`` — последняя обучающая точка) с колонкой
    ``test_MeanAbsoluteScaledError``. Агрегация (медиана по фолдам) —
    дело вызывающего кода, здесь только честный перфолдовый лог.
    """

    if splitter is None:
        splitter = make_expanding_splitter()
    return evaluate(
        forecaster=forecaster,
        y=y,
        cv=splitter,
        scoring=MeanAbsoluteScaledError(),
        strategy="refit",
        error_score="raise",
    )


__all__ = [
    "INITIAL_WINDOW",
    "SP",
    "SeriesContinuityError",
    "TEST_LENGTH",
    "evaluate_forecaster",
    "make_expanding_splitter",
    "seasonal_naive",
    "to_monthly_series",
]

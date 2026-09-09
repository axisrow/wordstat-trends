"""Модели поверх каркаса валидации: Theta, AutoETS и AutoARIMA-бенчмарк.

Обе модели фазы 2.2 (#74) и бенчмарк фазы 2.3 (#75) проходят через каркас
из ``baseline.py`` — тот же ``ExpandingWindowSplitter`` из ``sktime.split``,
тот же ``evaluate``, та же метрика MASE. Никаких отдельных сплитов под
модель: сравнение с сезонным наивным бейзлайном честно только на общих
фолдах.

Учёт накопленного в #32/#34 (там — дневные ряды, sp=7):

- #32: AutoETS(auto=True) может менять саму структуру модели
  (``has_seasonal``) между окнами 56/42 — на коротких дневных окнах выбор
  спецификации по AIC/AICc неустойчив. Здесь обучающие окна 72 и 84 месяца
  (6 и 7 полных сезонных циклов) — существенно длиннее, но вывод #32
  обязывает не доверять структуре молча: :func:`ets_full_series_structure`
  фиксирует выбранную на полном ряду спецификацию явно.
- #34: сезонность детектируется только начиная с некоторой пороговой длины
  окна. Начальное окно каркаса (72 месяца = 6 циклов) выбрано выше любого
  порога из #34, поэтому сезонная компонента доступна с первого фолда.

AutoARIMA (issue #75) — бенчмарк для сравнения, НЕ кандидат в прод.
Реализация — ``StatsForecastAutoARIMA`` из sktime: это обёртка над уже
закреплённой зависимостью ``statsforecast``, новых пакетов не требуется
(pmdarima-обёртка ``sktime.forecasting.arima.AutoARIMA`` потребовала бы
отдельную тяжёлую зависимость — ради бенчмарка не тащим). Гипотеза #75:
на ~100 точках ARIMA-семейство проигрывает Theta/ETS — здесь она получает
число.

Состав:

- :func:`theta_model` — ``ThetaForecaster(sp=12)``;
- :func:`auto_ets` — ``AutoETS(sp=12, auto=True)`` с фиксированным
  ``random_state`` и ``n_jobs=1`` (детерминированность сетки спецификаций);
- :func:`auto_arima` — ``StatsForecastAutoARIMA(sp=12)`` (бенчмарк #75);
- :func:`ets_full_series_structure` — какая структура ETS выбрана
  AutoETS(auto=True) на полном ряду;
- :func:`evaluate_models` — все модели и наивный бейзлайн на ОДНИХ фолдах,
  MASE по каждому фолду + медиана.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.split import ExpandingWindowSplitter

from wordstat_trends.forecasting.baseline import (
    SP,
    evaluate_forecaster,
    make_expanding_splitter,
    seasonal_naive,
)

#: Фиксированный seed для моделей, где есть стохастика (init-розыгрыш
#: AutoETS). ThetaForecaster стохастики не имеет и детерминирован по
#: построению — seed там неприменим.
RANDOM_STATE = 42

#: Порядок моделей в отчёте :func:`evaluate_models`: бейзлайн первым —
#: с ним сравниваются остальные. Определяется лениво (после фабрик), см.
#: конец модуля.
MODEL_FACTORIES: dict[str, Callable[[], BaseForecaster]]


def theta_model() -> ThetaForecaster:
    """``ThetaForecaster(sp=12)`` — классический Theta-метод sktime."""

    return ThetaForecaster(sp=SP)


def auto_ets() -> AutoETS:
    """``AutoETS(sp=12, auto=True)`` — сетка спецификаций ETS по AICc.

    ``n_jobs=1`` и фиксированный ``random_state`` — детерминированность
    выбора спецификации между прогонами (см. тесты).
    """

    return AutoETS(sp=SP, auto=True, n_jobs=1, random_state=RANDOM_STATE)


def ets_full_series_structure(y: pd.Series) -> dict:
    """Структура ETS, выбранная AutoETS(auto=True) на ПОЛНОМ ряду.

    Возвращает ``{"spec": "AAdA", "has_seasonal": True, "error": "add", ...}``.
    Прямой аналог ``sktime_has_seasonal`` из ``scripts/issue32_seasonal_structure.py``,
    но для месячного sp=12: фиксация структуры — требование
    issue #74 после вывода #32 (смена ``has_seasonal`` между окнами).

    Атрибут ``_fitted_forecaster`` приватный, но стабильный в sktime 1.x:
    это обёрнутый ``statsmodels`` ETSResults. Строка спецификации —
    классические три буквы ETS: ошибка / тренд (``d`` — демпфированный) /
    сезонность.
    """

    forecaster = auto_ets()
    forecaster.fit(y)
    fitted = forecaster._fitted_forecaster  # type: ignore[attr-defined]  # statsmodels ETSResults
    model = fitted.model  # type: ignore[attr-defined]
    trend_letter = {"add": "A", "mul": "M"}.get(model.trend, "")
    if trend_letter and model.damped_trend:
        trend_letter += "d"
    seasonal_letter = {"add": "A", "mul": "M"}.get(model.seasonal, "N")
    spec = f"{model.error[0].upper()}{trend_letter or 'N'}{seasonal_letter}"
    return {
        "spec": spec,
        "has_seasonal": model.seasonal is not None,
        "error": model.error,
        "trend": model.trend,
        "damped_trend": bool(model.damped_trend),
        "seasonal": model.seasonal,
    }


def evaluate_models(
    y: pd.Series,
    splitter: ExpandingWindowSplitter | None = None,
) -> pd.DataFrame:
    """Бейзлайн, Theta и AutoETS на ОДНИХ фолдах → MASE по фолдам.

    Один и тот же сплиттер (по умолчанию — из
    :func:`~wordstat_trends.forecasting.baseline.make_expanding_splitter`)
    прогоняется через все модели: никаких отдельных сплитов под модель.
    Возвращает DataFrame: строка на модель, колонки ``fold_0..fold_n`` с
    перфолдовым MASE и ``mase_median`` для сводного сравнения. MASE < 1 у
    модели — она точнее наивного бейзлайна на тех же фолдах.
    """

    if splitter is None:
        splitter = make_expanding_splitter()
    rows: dict[str, dict[str, float]] = {}
    for name, factory in MODEL_FACTORIES.items():
        forecaster: BaseForecaster = factory()
        result = evaluate_forecaster(forecaster, y, splitter)
        mase = result["test_MeanAbsoluteScaledError"]
        row: dict[str, float] = {f"fold_{i}": v for i, v in enumerate(mase)}
        row["mase_median"] = mase.median()
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def auto_arima() -> StatsForecastAutoARIMA:
    """``StatsForecastAutoARIMA(sp=12)`` — бенчмарк, НЕ кандидат в прод (#75).

    sktime-обёртка над ``statsforecast.models.AutoARIMA`` (порт Hyndman
    ``forecast::auto.arima``): stepwise-поиск порядка по AICc. Именно эта
    обёртка, а не ``sktime.forecasting.arima.AutoARIMA``: последняя требует
    ``pmdarima`` — отдельную тяжёлую зависимость, которую ради одного
    бенчмарка не тащим; ``statsforecast`` уже закреплён в проекте.
    """

    return StatsForecastAutoARIMA(sp=SP)


MODEL_FACTORIES = {
    "seasonal_naive": seasonal_naive,
    "theta": theta_model,
    "auto_ets": auto_ets,
    "auto_arima": auto_arima,
}

__all__ = [
    "RANDOM_STATE",
    "auto_arima",
    "auto_ets",
    "ets_full_series_structure",
    "evaluate_models",
    "theta_model",
]

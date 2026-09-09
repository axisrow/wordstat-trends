"""Отсев ложных трендов поверх скора детекции (issue #84, фаза 3 Б2).

Витрина показывает долгосрочные тренды (решение 2026-08-21 в #12):
аудитория — закупка товаров из Китая с циклом поставки недели и месяцы.
Инфоповод, попавший в витрину, — не шум, а неверный ответ: пользователь
закупит товар под всплеск, которого через неделю не будет. Отсев —
определяющее требование, ослаблять его ради краткосрочных всплесков #22
нельзя: у тех другой сигнал, другой ряд (дневной) и другой срок жизни —
граница зафиксирована в докстринге :mod:`wordstat_trends.short_term`.

Роль месячной грануляции — сознательная, а не молчаливая: месячное значение
Вордстата агрегирует дневные запросы, поэтому дневной всплеск (#22) уже
погашен агрегацией и до скора детекции не доходит. Оба фильтра ниже работают
с **месячными** эффектами:

1. **Разовый всплеск против устойчивого роста** — скор детекции есть
   отношение средних по окну, один аномальный месяц с большой амплитудой
   поднимает его за порог. Требуется подтверждение несколькими месяцами:
   месяц подтверждает рост, если его факт / сезонный наив для него
   >= ``MONTH_RATIO_THRESHOLD``; нужно ``MIN_CONFIRMED_MONTHS`` из окна.
2. **Праздничная сезонность против тренда** — наив sp=12 считает
   декабрьский пик нормой, но амплитуда пика, растущая год к году, — тренд.
   Критерий — амплитуда пика последнего полного календарного года против
   предыдущего (год к году, не месяц к месяцу). Применяется только к рядам
   с сезонной компонентой по :func:`ets_full_series_structure` — вывод #32:
   вердикту о сезонности нельзя доверять молча, структура фиксируется
   явно (docs/ISSUE_32_SEASONAL_STRUCTURE.md).

Пороги ``MONTH_RATIO_THRESHOLD``, ``MIN_CONFIRMED_MONTHS``,
``PEAK_ARC_SHARE``, ``PEAK_GROWTH_RATIO`` предрегистрированы в
docs/TRENDS.md до прогона на данных и на фикстурах не подбирались.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wordstat_trends.forecasting.baseline import SP
from wordstat_trends.forecasting.models import ets_full_series_structure
from wordstat_trends.trends.growth import GrowthScore, score_growth

#: Месячный факт / сезонный наив для этого месяца >= 1.25 — месяц
#: «подтверждает» рост. Ниже порога окна (1.5): месяц шумнее среднего окна,
#: но подтверждение должно быть содержательным. Предрегистрировано в
#: docs/TRENDS.md.
MONTH_RATIO_THRESHOLD = 1.25

#: Сколько подтверждённых месяцев из SCORE_WINDOW нужно для тренда:
#: один месяц — не тренд (инфоповод), два — минимальное подтверждение.
#: Предрегистрировано в docs/TRENDS.md.
MIN_CONFIRMED_MONTHS = 2

#: «Праздничная дуга»: календарные месяцы с историческим уровнем
#: >= 0.5 × уровень пика. Рост в месяцы вне дуги сезонного объяснения не
#: имеет — пик-анализ к нему не применяется. Предрегистрировано в
#: docs/TRENDS.md.
PEAK_ARC_SHARE = 0.5

#: Амплитуда пика последнего полного года / предыдущего >= 1.3 —
#: амплитуда растёт, это тренд. Ниже порога детекции 1.5: рост амплитуды
#: компаундится на двух полных годах. Предрегистрировано в docs/TRENDS.md.
PEAK_GROWTH_RATIO = 1.3

#: Вердикты отсева. «seasonal_peak_flat» — праздничная сезонность без
#: роста амплитуды (отсев), «peak_amplitude_growing» — растущий пик
#: (тренд), «single_spike» — инфоповод, «no_growth» — скор ниже порога
#: детекции.
REASON_NO_GROWTH = "no_growth"
REASON_SINGLE_SPIKE = "single_spike"
REASON_SEASONAL_PEAK_FLAT = "seasonal_peak_flat"
REASON_PEAK_AMPLITUDE_GROWING = "peak_amplitude_growing"
REASON_GROWTH_CONFIRMED = "growth_confirmed"


@dataclass(frozen=True)
class TrendVerdict:
    """Вердикт отсева ложных трендов одной фразы.

    ``score`` — скор детекции (:class:`GrowthScore`), на котором основан
    вердикт; ``month_ratios`` — помесячные факт/сезонный наив окна;
    ``confirmed_months`` — месяцы, прошедшие ``MONTH_RATIO_THRESHOLD``;
    ``peak_ratio`` — амплитуда пика последнего полного года / предыдущего
    (None, если пик-анализ не применялся); ``ets_structure`` — структура
    ETS на полном ряду (None, если фильтр 2 не дошёл до неё).
    """

    phrase: str
    score: GrowthScore
    is_trend: bool
    reason: str
    month_ratios: tuple[float, ...]
    confirmed_months: tuple[pd.Period, ...]
    peak_ratio: float | None
    ets_structure: dict | None


def month_ratios(score: GrowthScore) -> tuple[float, ...]:
    """Помесячные отношения факт/прогноз окна скоринга.

    Прогноз сезонного наива positivity гарантирована :func:`score_growth`
    (нулевой прогноз там — громкий отказ).
    """

    return tuple(
        float(a) / float(f) for a, f in zip(score.actual.to_numpy(), score.forecast.to_numpy())
    )


def _seasonal_profile(history: pd.Series) -> pd.Series:
    """Средний уровень каждого календарного месяца по истории вне окна."""

    return history.groupby(history.index.month).mean()


def _yearly_peak_ratio(history: pd.Series) -> float | None:
    """Амплитуда пика последнего полного календарного года / предыдущего.

    Полные годы — 12 месяцев истории вне окна. Меньше двух полных лет —
    None (пик-анализ неприменим, правило предегистрировано в
    docs/TRENDS.md).
    """

    months_per_year = history.groupby(history.index.year).size()
    full_years = months_per_year[months_per_year == SP]
    if len(full_years) < 2:
        return None
    last_two = full_years.index[-2:]
    peaks = [float(history[history.index.year == year].max()) for year in last_two]
    if peaks[0] <= 0:
        return None
    return peaks[1] / peaks[0]


def classify_growth(
    y: pd.Series,
    phrase: str = "",
    score: GrowthScore | None = None,
) -> TrendVerdict:
    """Месячный ряд (+ скор детекции) → вердикт «тренд или ложный сигнал».

    ``y`` — месячный ряд из :func:`to_monthly_series`; ``score`` — скор
    детекции :func:`score_growth` (если None, считается здесь на том же
    окне ``SCORE_WINDOW``).
    """

    if score is None:
        score = score_growth(y, phrase=phrase)
    ratios = month_ratios(score)
    confirmed = tuple(
        p for p, r in zip(score.window, ratios) if r >= MONTH_RATIO_THRESHOLD
    )

    def verdict(
        is_trend: bool,
        reason: str,
        peak_ratio: float | None = None,
        ets_structure: dict | None = None,
    ) -> TrendVerdict:
        return TrendVerdict(
            phrase=phrase or score.phrase,
            score=score,
            is_trend=is_trend,
            reason=reason,
            month_ratios=ratios,
            confirmed_months=confirmed,
            peak_ratio=peak_ratio,
            ets_structure=ets_structure,
        )

    # Порог детекции не пройден — фильтрам нечего отсеивать.
    if not score.is_growth:
        return verdict(False, REASON_NO_GROWTH)

    # Фильтр 1: подтверждение роста несколькими месяцами против сезонного
    # ожидания. Один аномальный месяц (инфоповод), даже утащивший среднее
    # окна за GROWTH_RATIO_THRESHOLD, — не тренд.
    if len(confirmed) < MIN_CONFIRMED_MONTHS:
        return verdict(False, REASON_SINGLE_SPIKE)

    # Фильтр 2: праздничная сезонность против тренда. Применяется только
    # когда все подтверждённые месяцы лежат в праздничной дуге — рост в
    # несезонный месяц сезонного объяснения не имеет.
    history = y[~y.index.isin(score.window)]
    profile = _seasonal_profile(history)
    peak_level = float(profile.max())
    arc_months = {m for m, v in profile.items() if v >= PEAK_ARC_SHARE * peak_level}
    if not {p.month for p in confirmed} <= arc_months:
        return verdict(True, REASON_GROWTH_CONFIRMED)

    # Вывод #32: сезонность фиксируется структурой ETS на полном ряду
    # явно, а не молчаливым предположением.
    structure = ets_full_series_structure(y)
    if not structure["has_seasonal"]:
        return verdict(True, REASON_GROWTH_CONFIRMED, ets_structure=structure)

    ratio = _yearly_peak_ratio(history)
    if ratio is None:
        return verdict(True, REASON_GROWTH_CONFIRMED, ets_structure=structure)
    if ratio >= PEAK_GROWTH_RATIO:
        return verdict(True, REASON_PEAK_AMPLITUDE_GROWING, ratio, structure)
    return verdict(False, REASON_SEASONAL_PEAK_FLAT, ratio, structure)


__all__ = [
    "MIN_CONFIRMED_MONTHS",
    "MONTH_RATIO_THRESHOLD",
    "PEAK_ARC_SHARE",
    "PEAK_GROWTH_RATIO",
    "REASON_GROWTH_CONFIRMED",
    "REASON_NO_GROWTH",
    "REASON_PEAK_AMPLITUDE_GROWING",
    "REASON_SEASONAL_PEAK_FLAT",
    "REASON_SINGLE_SPIKE",
    "TrendVerdict",
    "classify_growth",
    "month_ratios",
]

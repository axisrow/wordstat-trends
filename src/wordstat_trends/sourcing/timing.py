"""Сезонный тайминг закупки (issue #114, фаза 6.1).

Переупаковка фаз 2–4 в календарь закупки, без нового ML: месячный профиль
ряда (общая функция :func:`seasonal_month_profile` из ``trends.ranking``)
→ месяц пика → окно заказа «пик минус срок поставки». Lead time —
параметр закупки (недели), не прогноз.

Выход — структура данных без рендера. Вордстат меряет спрос, не маржу:
окно отвечает «когда заказывать, чтобы попасть в пик», не «выгодно ли»
(граница эпика #15). Ряд без сезонности
(``seasonal_max_min_ratio < SEASONAL_MAX_MIN_RATIO``) — честный ``None``:
пик не выдумывается, решение остаётся «по тренду».

Формулы предрегистрированы в docs/TRENDS.md до прогона на фикстурах.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from wordstat_trends.trends.ranking import (
    SEASONAL_MAX_MIN_RATIO,
    seasonal_max_min_ratio,
    seasonal_month_profile,
)

#: Дефолтный lead time: ж/д доставка Китай→РФ — типовой путь
#: маркетплейс-закупок. Предрегистрирован в docs/TRENDS.md.
DEFAULT_LEAD_WEEKS = 8

#: Пресеты lead time (недели) по способу доставки Китай→РФ:
#: авиа-экспресс (дорого, малые партии), авиа, ж/д (дефолт), море
#: (дёшево, большие партии). Обоснование — docs/TRENDS.md.
LEAD_TIME_PRESETS: dict[str, int] = {
    "air_express": 4,
    "air": 6,
    "rail": 8,
    "sea": 12,
}

#: Средний месяц в неделях: 365.25/12/7 ≈ 4.348. Предрегистрировано
#: в docs/TRENDS.md.
WEEKS_PER_MONTH = 4.348

#: Люфт на задержку поставки: крайний безопасный месяц заказа — месяц
#: заказа + 1. Предрегистрирован в docs/TRENDS.md.
BUFFER_MONTHS = 1


def lead_weeks_to_months(lead_weeks: int) -> int:
    """Lead time в неделях → месяцы, округление вверх: лучше заказать
    раньше, чем опоздать к пику."""

    return math.ceil(lead_weeks / WEEKS_PER_MONTH)


def peak_month(y: pd.Series) -> int:
    """Месяц пика спроса: месяц с максимальным средним профиля (1..12).

    Тай-брейк при равных max — меньший номер месяца (детерминированность;
    месяц — не дата, календарь один). Предрегистрирован в docs/TRENDS.md.
    """

    profile = seasonal_month_profile(y)
    return int(profile.idxmax())


@dataclass(frozen=True)
class OrderWindow:
    """Окно заказа: пик, месяц заказа и крайний безопасный месяц."""

    peak_month: int  # месяц пика спроса, 1..12
    order_month: int  # месяц заказа = пик − lead_months, 1..12
    latest_order_month: int  # крайний месяц с люфтом = order_month + BUFFER_MONTHS
    lead_weeks: int
    lead_months: int
    buffer_months: int


@dataclass(frozen=True)
class SourcingTiming:
    """Календарь закупки фразы. ``window=None`` — ряд без сезонности,
    окно не определяется («по тренду»), поля пика/успеваемости — None."""

    phrase: str
    last_month: pd.Period  # последняя точка данных ряда
    window: OrderWindow | None
    months_to_peak: int | None  # до ближайшего будущего вхождения пика
    slack_months: int | None  # months_to_peak − lead_months
    on_time: bool | None  # успеваем ли к ближайшему пику; None без окна


def sourcing_timing(
    y: pd.Series, phrase: str = "", lead_weeks: int = DEFAULT_LEAD_WEEKS
) -> SourcingTiming:
    """Месячный ряд → календарь закупки (месяц заказа, люфт, успеваемость).

    Ряд без сезонности (амплитуда профиля ниже порога) → ``window=None``:
    пик не выдумывается. ``months_to_peak`` — строго вперёд от последней
    точки данных: если последний месяц — пик, ближайший через 12.
    """

    last = pd.PeriodIndex(y.index)[-1]
    if seasonal_max_min_ratio(y) < SEASONAL_MAX_MIN_RATIO:
        return SourcingTiming(
            phrase=phrase, last_month=last, window=None,
            months_to_peak=None, slack_months=None, on_time=None,
        )

    top = peak_month(y)
    lead_months = lead_weeks_to_months(lead_weeks)
    order_month = (top - lead_months - 1) % 12 + 1
    window = OrderWindow(
        peak_month=top,
        order_month=order_month,
        latest_order_month=(order_month + BUFFER_MONTHS - 1) % 12 + 1,
        lead_weeks=lead_weeks,
        lead_months=lead_months,
        buffer_months=BUFFER_MONTHS,
    )
    months_to_peak = (top - last.month - 1) % 12 + 1
    slack = months_to_peak - lead_months
    return SourcingTiming(
        phrase=phrase,
        last_month=last,
        window=window,
        months_to_peak=months_to_peak,
        slack_months=slack,
        on_time=slack >= 0,
    )


__all__ = [
    "BUFFER_MONTHS",
    "DEFAULT_LEAD_WEEKS",
    "LEAD_TIME_PRESETS",
    "OrderWindow",
    "SourcingTiming",
    "WEEKS_PER_MONTH",
    "lead_weeks_to_months",
    "peak_month",
    "sourcing_timing",
]

"""Сезонный тайминг закупки (issue #114, фаза 6.1).

Эпик #15: «сезонная компонента → когда закупать (пик минус срок
доставки)». Модуль не добавляет ML — переупаковывает месячный профиль,
который ранжирование (#85) уже считает для классификации «сезонное», в
календарь заказа: месяц пика спроса, месяц заказа (пик минус срок
поставки из Китая), успеваем к ближайшему пику или нет.

Правила предрегистрированы в docs/TRENDS.md (раздел фазы 6.1) и здесь
только воспроизводятся:

- пик — месяц с наибольшим средним месячного профиля; ряд без сезонной
  амплитуды (``seasonal_max_min_ratio`` ниже порога классификации)
  пика не имеет — окно не определяется, честный ``None``, пик не
  выдумывается;
- тай-брейк равных максимумов — наименьший номер месяца;
- срок поставки — параметр в неделях (недели–месяцы эпика), в месяцы
  переводится округлением вверх: ``ceil(weeks * 12 / 52)``;
- «успеваем» — до ближайшего будущего пика не меньше месяцев, чем
  занимает поставка: заказ в этом цикле доставится к пику, иначе
  следующий пик через год.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from wordstat_trends.forecasting.baseline import SP
from wordstat_trends.trends.ranking import SEASONAL_MAX_MIN_RATIO, seasonal_max_min_ratio

#: Сроки поставки Китай → РФ по типам логистики, недели, округлённые
#: вверх от типичных диапазонов (авиа ~2–3, ж/д ~5–7, море ~8–10).
#: Ориентир для пресетов, не прогноз: реальный срок — параметр вызова.
LEAD_TIME_PRESETS: dict[str, int] = {"air": 4, "rail": 8, "sea": 12}

#: Дефолт — ж/д (смешанная доставка): середина диапазона эпика
#: «недели–месяцы» и самый массовый способ для товарки из Китая.
DEFAULT_LEAD_TIME_WEEKS = LEAD_TIME_PRESETS["rail"]

#: Минимум истории для профиля: два полных календарных года не нужно,
#: но меньше двух проходов каждого месяца профиль шумит — берём тот же
#: горизонт, что и детекция роста (MIN_HISTORY=24 в growth.py проверяется
#: выше по течению, здесь только страховка от вырожденного входа).
MIN_PROFILE_MONTHS = SP * 2


@dataclass(frozen=True)
class PurchaseWindow:
    """Календарь закупки одной фразы/ниши: когда заказывать к какому пику.

    ``order_period``/``peak_period`` — ближайшие будущие (или текущий
    месяц для заказа) месяцы от последней точки ряда.
    """

    peak_month: int  # 1..12 — месяц пика спроса
    order_month: int  # 1..12 — месяц заказа: пик минус срок поставки
    order_period: pd.Period  # ближайший месяц заказа от конца ряда
    peak_period: pd.Period  # ближайший будущий пик от конца ряда
    months_to_order: int  # месяцев от конца ряда до order_period
    months_to_peak: int  # месяцев от конца ряда до peak_period
    lead_time_months: int  # срок поставки в месяцах (округление вверх)
    on_time: bool  # заказ сейчас успевает к ближайшему пику


def monthly_profile(y: pd.Series) -> pd.Series:
    """Средние по календарным месяцам за всю историю ряда.

    Тот же профиль, что внутри :func:`seasonal_max_min_ratio` (#85):
    усреднение по годам сглаживает разовые всплески, остаётся устойчивая
    сезонная компонента. Индекс результата — номер месяца 1..12.
    """

    month_means = y.groupby(pd.PeriodIndex(y.index).month).mean().astype(float)
    return month_means


def peak_month(y: pd.Series) -> int | None:
    """Месяц пика спроса или ``None`` для ряда без сезонной амплитуды.

    Порог — тот же, что у классификации «сезонное» (#85): ниже
    ``SEASONAL_MAX_MIN_RATIO`` колебания неотличимы от шума, пик
    «найдётся» у любого ряда — поэтому его нет. Равные максимумы —
    наименьший номер месяца (детерминированность, winter-пики
    декабрь/январь дают декабрь).
    """

    if len(y) < MIN_PROFILE_MONTHS or seasonal_max_min_ratio(y) < SEASONAL_MAX_MIN_RATIO:
        return None
    profile = monthly_profile(y)
    return int(profile.idxmax())  # idxmax у Series с числовым индексом — первый максимум


def lead_time_months(weeks: int) -> int:
    """Срок поставки в неделях → месяцы, округление вверх.

    Округление вверх — ошибка в меньшую сторону стоит пропущенного пика.
    """

    return math.ceil(weeks * 12 / 52)


def purchase_window(
    y: pd.Series,
    lead_time_weeks: int = DEFAULT_LEAD_TIME_WEEKS,
) -> PurchaseWindow | None:
    """Ряд → окно закупки: месяц заказа к ближайшему пику или ``None``.

    ``None`` — пик не определён (несезонный ряд): календарь не
    выдумывается, решение по такому ряду — «по тренду», не по сезону.
    """

    peak = peak_month(y)
    if peak is None:
        return None
    lead = lead_time_months(lead_time_weeks)
    last = pd.PeriodIndex(y.index)[-1]

    def next_offset(month: int) -> int:
        # Смещение до ближайшего месяца `month` от конца ряда:
        # 0 — месяц заказа уже настал (заказывать сейчас), 1..11 — впереди.
        return (month - last.month) % 12

    order_month = (peak - lead - 1) % 12 + 1  # месяц заказа в 1..12
    months_to_order = next_offset(order_month)
    months_to_peak = next_offset(peak)
    return PurchaseWindow(
        peak_month=peak,
        order_month=order_month,
        order_period=last + months_to_order,
        peak_period=last + months_to_peak,
        months_to_order=months_to_order,
        months_to_peak=months_to_peak,
        lead_time_months=lead,
        on_time=months_to_peak >= lead,
    )


__all__ = [
    "DEFAULT_LEAD_TIME_WEEKS",
    "LEAD_TIME_PRESETS",
    "MIN_PROFILE_MONTHS",
    "PurchaseWindow",
    "lead_time_months",
    "monthly_profile",
    "peak_month",
    "purchase_window",
]

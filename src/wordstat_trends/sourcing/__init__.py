"""Сезонный тайминг закупки (issue #114, фаза 6.1).

Пакет «когда закупать»: месяц пика спроса, lead time поставки, окно
заказа. Формулы предрегистрированы в docs/TRENDS.md.
"""

from wordstat_trends.sourcing.timing import (
    BUFFER_MONTHS,
    DEFAULT_LEAD_WEEKS,
    LEAD_TIME_PRESETS,
    WEEKS_PER_MONTH,
    OrderWindow,
    SourcingTiming,
    lead_weeks_to_months,
    peak_month,
    sourcing_timing,
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

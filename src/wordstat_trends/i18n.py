"""Локализация витрины: форматы чисел и дат по локали (ru по умолчанию, zh).

Значения форматируются на этапе сборки статического сайта (issue #21);
те же функции использует рендеринг данных (#12, #14).

Форматы сверены с реальными значениями tests/fixtures/dynamics_*.csv:
«Число запросов» там приходит как «997 977» (ru, неразрывный пробел),
периоды — как «август 2024». Zh-закупщик привык к «997,977» и «2024年8月».
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

DEFAULT_LOCALE = "ru"

# Русские названия месяцев — Вордстат отдаёт их в периодах («август 2024»),
# поэтому держим свой полный список, а не локаль системы: сборка детерминирована
# и не зависит от установленных locale на раннере Actions.
# Именительный — в периодах Вордстата («август 2024»), родительный — в полных
# датах («9 сентября 2026 г.»). Склонения нерегулярны, поэтому оба списка явные.
_RU_MONTHS = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
]

_RU_MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]

_ZH_MONTHS = [
    "1月", "2月", "3月", "4月", "5月", "6月",
    "7月", "8月", "9月", "10月", "11月", "12月",
]

class Locale(StrEnum):
    RU = "ru"
    ZH = "zh"


def normalize_locale(value: str) -> Locale:
    """«ru-RU»/«zh_CN» → Locale; неизвестное → ru (дефолт проекта)."""
    code = value.replace("_", "-").split("-")[0].lower()
    return Locale.ZH if code == "zh" else Locale.RU


def format_number(value: int | float, locale: Locale | str = Locale.RU) -> str:
    """Целое число с разделителем групп по локали.

    ru: 997977 → «997 977» (узкий неразрывный пробел, как в Вордстате).
    zh: 997977 → «997,977» (запятая, стандарт zh-CN).
    """
    locale = normalize_locale(str(locale))
    grouped = f"{value:,}"
    if locale is Locale.ZH:
        return grouped
    return grouped.replace(",", " ")


def format_share(value: float, locale: Locale | str = Locale.RU) -> str:
    """Доля («Доля от всех запросов, %» из выгрузки Вордстата).

    В CSV она приходит как «0,0115» — десятичная запятая в обеих локалях,
    различается пробел перед знаком процента: ru «0,0115 %», zh «0.0115%».
    """
    locale = normalize_locale(str(locale))
    if locale is Locale.ZH:
        return f"{value:.4f}" + "%"
    return f"{value:.4f}".replace(".", ",") + " %"


def format_month(month: date, locale: Locale | str = Locale.RU) -> str:
    """Месячный период («август 2024» / «2024年8月»)."""
    locale = normalize_locale(str(locale))
    if locale is Locale.ZH:
        return f"{month.year}年{_ZH_MONTHS[month.month - 1]}"
    return f"{_RU_MONTHS[month.month - 1]} {month.year}"


def format_ratio(value: float, locale: Locale | str = Locale.RU) -> str:
    """Отношение «факт/прогноз» (issue #20: «рост к прогнозу в 2,3 раза»).

    Один десятичный знак — достаточная точность для подписи витрины;
    разделитель по локали: ru «2,3», zh «2.3». Значения — реальные
    ``GrowthScore.score`` из tests/fixtures (например 0.5 у падающей
    синтетики, 2.0 у растущей), а не абстрактные примеры.
    """
    locale = normalize_locale(str(locale))
    rendered = f"{value:.1f}"
    return rendered if locale is Locale.ZH else rendered.replace(".", ",")


def format_date(day: date, locale: Locale | str = Locale.RU) -> str:
    """Полная дата: ru «9 сентября 2026 г.», zh «2026年9月9日»."""
    locale = normalize_locale(str(locale))
    if locale is Locale.ZH:
        return f"{day.year}年{day.month}月{day.day}日"
    genitive = _RU_MONTHS_GENITIVE[day.month - 1]
    return f"{day.day} {genitive} {day.year} г."

"""Шаблоны объяснений витрины с подстановкой (issue #20, фаза 5).

Объяснения генерируются из данных («рост к прогнозу в 2,3 раза»), поэтому
переводятся **шаблоны с плейсхолдерами**, а не готовые строки — иначе числа
и падежи разъедутся. Шаблоны живут в существующем механизме локализации
``site/locales/{ru,zh}.json`` (ключи ``explain.*`` и ``trend.class.*``),
второго механизма не заводится.

Подставляемые структуры — фактические выходы детекции и ранжирования:
``GrowthScore`` (:mod:`wordstat_trends.trends.growth`) даёт отношение
факт/прогноз, ``RankedPhrase`` (:mod:`wordstat_trends.trends.ranking`) —
класс и возраст роста. Числа подставляются уже отформатированными по
локали (:func:`wordstat_trends.i18n.format_ratio`) — в шаблон не ездит
сырой float.

Граница принципа «без LLM»: объяснение собирается подстановкой в
детерминированный шаблон, никакого генеративного вызова; все числа уже
посчитаны классическим пайплайном до перевода.
"""

from __future__ import annotations

import json
from pathlib import Path

from wordstat_trends.i18n import Locale, format_ratio, normalize_locale
from wordstat_trends.trends.growth import GrowthScore
from wordstat_trends.trends.ranking import RankedPhrase, TrendClass

#: Каталог локалей механизма i18n (site/locales, issue #21).
LOCALES_DIR = Path(__file__).resolve().parents[2] / "site" / "locales"

#: Ключ шаблона для каждого класса фразы (TrendClass → ключ локали).
_CLASS_TEMPLATE_KEY = {
    TrendClass.GROWING: "explain.growing",
    TrendClass.SEASONAL: "explain.seasonal",
    TrendClass.FALLING: "explain.falling",
    TrendClass.STABLE: "explain.stable",
}

#: Ключ метки класса (растёт/сезонное/... — русские значения TrendClass
#: локализуются тем же механизмом ключей, хардкод второй таблицы не нужен).
_CLASS_LABEL_KEY = {
    TrendClass.GROWING: "trend.class.growing",
    TrendClass.SEASONAL: "trend.class.seasonal",
    TrendClass.FALLING: "trend.class.falling",
    TrendClass.STABLE: "trend.class.stable",
}


class ExplainError(Exception):
    """Ключа объяснения нет в локали — опечатка не проходит молча."""


def load_messages(locale: Locale | str, locales_dir: Path = LOCALES_DIR) -> dict[str, str]:
    """Сообщения локали из site/locales/<код>.json (существующий механизм)."""

    code = str(normalize_locale(str(locale)))
    return json.loads((locales_dir / f"{code}.json").read_text(encoding="utf-8"))


def _render(messages: dict[str, str], key: str, params: dict[str, str]) -> str:
    if key not in messages:
        raise ExplainError(f"нет ключа {key!r} в локали — проверьте site/locales/*.json")
    try:
        return messages[key].format(**params)
    except KeyError as exc:
        # Плейсхолдер шаблона без значения (например {months} при
        # components=None) — ошибка контракта, а не сырой KeyError.
        raise ExplainError(
            f"ключ {key!r}: плейсхолдер {exc} не получил значения"
        ) from exc


def explain_phrase(
    ranked: RankedPhrase,
    growth: GrowthScore,
    locale: Locale | str = Locale.RU,
    messages: dict[str, str] | None = None,
) -> str:
    """Объяснение строки витрины: шаблон класса + подстановка чисел.

    ``ranked``/``growth`` — фактические структуры выхода ranking/growth
    одной фразы: класс и возраст роста — из ``RankedPhrase``, отношение
    факт/прогноз — из ``GrowthScore.score`` (в ``RankedPhrase`` попадает
    только взвешенный скор витрины, исходное отношение живёт в детекции).
    """

    messages = messages if messages is not None else load_messages(locale)
    params: dict[str, str] = {"ratio": format_ratio(growth.score, locale)}
    if ranked.components is not None:
        params["months"] = str(ranked.components.growth_age_months)
    return _render(messages, _CLASS_TEMPLATE_KEY[ranked.klass], params)


def localized_class(klass: TrendClass, locale: Locale | str = Locale.RU) -> str:
    """Метка класса по локали («растёт» / «增长»)."""

    messages = load_messages(locale)
    label = _render(messages, _CLASS_LABEL_KEY[klass], {})
    # ru-значение ключа обязано совпадать с TrendClass.value — иначе
    # рассинхрон метки и логики классификации пройдёт молча.
    if normalize_locale(str(locale)) is Locale.RU and label != klass.value:
        raise ExplainError(
            f"trend.class.*: ru-метка {label!r} != TrendClass.value {klass.value!r}"
        )
    return label


__all__ = [
    "ExplainError",
    "LOCALES_DIR",
    "explain_phrase",
    "load_messages",
    "localized_class",
]

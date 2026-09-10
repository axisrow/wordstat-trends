"""Скоринг ниш для закупки (issue #115, фаза 6.2).

Композиция уже существующих кусков, нового ML нет: агрегированный скор
ниши (#106, медиана скоров состава) + сезонная амплитуда профиля (#85) +
окно закупки (#114) + устойчивость (доля отфильтрованного состава и
возраст роста — шов в ``showcase.build_niches``). Выход — структура
``list[NicheSourcing]`` без рендера (граница «ядро выдаёт структуру,
рендер — витрина»).

Формулы и веса предрегистрированы в docs/TRENDS.md (раздел фазы 6.2) и
здесь только воспроизводятся. Все компоненты в [0, 1] и считаются
относительно самой ниши — скор ниши не зависит от соседей по популяции
(принцип #85 сохраняется).

Гипотеза purchase-intent (п. 4 issue #115) — механизм разметки по
словарю интент-маркеров, в скор **не входит**: польза сигнала
валидируется только живыми данными, которых нет до деплоя.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from wordstat_trends.nlp.lemmatize import lemmatize
from wordstat_trends.sourcing.timing import (
    DEFAULT_LEAD_TIME_WEEKS,
    PurchaseWindow,
    purchase_window,
)
from wordstat_trends.trends.ranking import (
    CLASS_ORDER,
    SEASONAL_MAX_MIN_RATIO,
    PhraseRecord,
    TrendClass,
    seasonal_max_min_ratio,
)

#: Верхняя планка сезонной амплитуды для нормировки компоненты: профиль
#: с max/min = 100 («новогодние подарки» ≈ 50, фикстура фазы 1) считается
#: предельно выраженным. Предрегистрирована в docs/TRENDS.md.
SEASONAL_RATIO_TOP = 100.0

#: Веса компонент скора закупки. Тренд первичен — без роста/сезона
#: закупки нет; амплитуда, окно и устойчивость — равные поправки
#: «какой», «когда» и «надёжно ли». АПРИОРНЫЕ, НЕПОДТВЕРЖДЁННЫЕ:
#: живых данных до деплоя нет, калибровка — после (триггер issue #115).
#: Предрегистрированы в docs/TRENDS.md.
TREND_WEIGHT = 0.4
SEASONAL_WEIGHT = 0.2
WINDOW_WEIGHT = 0.2
STABILITY_WEIGHT = 0.2

#: Словарь интент-маркеров (леммы) для гипотезы purchase-intent:
#: уточняющие запросы о покупке как сигнал готовности покупать.
#: ГИПОТЕЗА: сигнал не валидирован (живых данных нет), в скор не входит.
INTENT_MARKER_LEMMAS = frozenset(
    {
        "купить",
        "цена",
        "стоимость",
        "заказать",
        "продажа",
        "недорого",
        "дешёвый",
    }
)


@dataclass(frozen=True)
class NicheSourcingComponents:
    """Компоненты скора закупки ниши, все в [0, 1] — для объяснимости."""

    trend: float  # агрегированный скор ниши #106 (медиана скоров состава)
    seasonal: float  # амплитуда профиля ниши, лог-нормировка до SEASONAL_RATIO_TOP
    window: float  # окно закупки #114: 1 — сезонная и успеваем к пику, иначе 0
    stability: float  # выживший состав × свежесть роста (для «растёт»)


@dataclass(frozen=True)
class NicheSourcing:
    """Строка приоритизации закупки: ниша, скор, компоненты и окно.

    ``window`` — окно закупки #114 по агрегированному ряду ниши или
    ``None`` (несезонная ниша: решение «по тренду»). ``intent_share`` —
    гипотеза purchase-intent, в скор не входит. ``phrases`` — состав
    ниши в порядке витрины.
    """

    topic: str
    klass: TrendClass
    score: float  # взвешенная сумма компонент, [0, 1]
    components: NicheSourcingComponents
    window: PurchaseWindow | None
    filtered_share: float  # доля состава, отсеянного ступенью #84
    median_growth_age_months: int | None  # медиана возраста роста (для «растёт»)
    intent_share: float  # доля интент-фраз состава (ГИПОТЕЗА, не в скоре)
    phrases: list[str]


def niche_series(niche, records: list[PhraseRecord]) -> pd.Series:
    """Агрегированный ряд ниши: среднее рядов состава по месяцам.

    Ряды живут в :class:`PhraseRecord`, а не в :class:`RankedPhrase`
    (как и в ``serialize_showcase``), поэтому состав ниши отображается
    обратно на записи. Среднее — та же агрегация «кластер → один ряд»,
    что и медиана скоров в #106: одна фраза состава не тянет профиль
    ниши целиком.
    """

    by_phrase = {record.phrase: record for record in records}
    frame = pd.concat(
        [by_phrase[row.phrase].series for row in niche.members], axis=1
    )
    return frame.mean(axis=1)


def seasonal_component(y: pd.Series) -> float:
    """Амплитуда профиля ряда → [0, 1].

    Ниже порога сезонности (#85) амплитуда — шум: компонента 0. Выше —
    логарифмическая нормировка до ``SEASONAL_RATIO_TOP``: различие
    амплитуды 10 и 20 важнее, чем 100 и 200 (та же логика, что у
    нормировки объёма в #85).
    """

    ratio = seasonal_max_min_ratio(y)
    if ratio < SEASONAL_MAX_MIN_RATIO:
        return 0.0
    return min(math.log10(ratio) / math.log10(SEASONAL_RATIO_TOP), 1.0)


def window_component(window: PurchaseWindow | None) -> float:
    """Окно закупки #114 → [0, 1]: бинарно.

    1 — пик есть и заказ сейчас успевает к нему; 0 — пик пропущен этого
    цикла или ряда сезонности нет (окно ``None``). Успеваем/не успеваем —
    бинарный вердикт #114; сглаживать его выдуманной шкалой нет данных.
    """

    return 1.0 if window is not None and window.on_time else 0.0


def stability_component(niche) -> float:
    """Устойчивость ниши → [0, 1]: выживший состав × свежесть роста.

    ``1 - filtered_share`` — доля состава, прошедшая отсев однодневок
    (#84). Для «растёт»-ниш дополнительно свежесть ``1 / медиана возраста
    роста`` (та же форма, что freshness в #85); вне «растёт» возраста нет
    — фактор 1.0 без штрафа.
    """

    survived = 1.0 - niche.filtered_share
    if niche.klass is TrendClass.GROWING and niche.median_growth_age_months:
        return survived / niche.median_growth_age_months
    return survived


def intent_share(niche) -> float:
    """Доля интент-фраз в составе ниши (ГИПОТЕЗА purchase-intent).

    Фраза интентная, если среди её лемм есть маркер из
    ``INTENT_MARKER_LEMMAS``. Сигнал НЕ валидирован живыми данными и в
    скор закупки не входит — механизм разметки для будущей валидации.
    """

    lemmas = [set(lemmatize(row.phrase)) for row in niche.members]
    marked = sum(1 for phrase_lemmas in lemmas if phrase_lemmas & INTENT_MARKER_LEMMAS)
    return marked / len(lemmas) if lemmas else 0.0


def score_sourcing(
    niches: list,
    records: list[PhraseRecord],
    lead_time_weeks: int = DEFAULT_LEAD_TIME_WEEKS,
) -> list[NicheSourcing]:
    """Ниши витрины + записи с рядами → приоритизация закупки.

    Скор = веса × компоненты (предрегистрация в docs/TRENDS.md), каждая
    компонента — относительно самой ниши, не популяции. Порядок: скор
    убыванием, затем ``CLASS_ORDER``, затем тема по алфавиту
    (детерминированность, как порядок ниш в #106).
    """

    rows: list[NicheSourcing] = []
    for niche in niches:
        y = niche_series(niche, records)
        window = purchase_window(y, lead_time_weeks=lead_time_weeks)
        components = NicheSourcingComponents(
            trend=niche.score,
            seasonal=seasonal_component(y),
            window=window_component(window),
            stability=stability_component(niche),
        )
        score = (
            TREND_WEIGHT * components.trend
            + SEASONAL_WEIGHT * components.seasonal
            + WINDOW_WEIGHT * components.window
            + STABILITY_WEIGHT * components.stability
        )
        rows.append(
            NicheSourcing(
                topic=niche.topic,
                klass=niche.klass,
                score=score,
                components=components,
                window=window,
                filtered_share=niche.filtered_share,
                median_growth_age_months=niche.median_growth_age_months,
                intent_share=intent_share(niche),
                phrases=[row.phrase for row in niche.members],
            )
        )
    rows.sort(
        key=lambda row: (
            -row.score,
            CLASS_ORDER.index(row.klass),
            row.topic,
        )
    )
    return rows


__all__ = [
    "INTENT_MARKER_LEMMAS",
    "NicheSourcing",
    "NicheSourcingComponents",
    "SEASONAL_RATIO_TOP",
    "SEASONAL_WEIGHT",
    "STABILITY_WEIGHT",
    "TREND_WEIGHT",
    "WINDOW_WEIGHT",
    "intent_share",
    "niche_series",
    "score_sourcing",
    "seasonal_component",
    "stability_component",
    "window_component",
]

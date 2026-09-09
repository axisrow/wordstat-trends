"""Ранжирование фраз для витрины (issue #85, фаза 3 Б3).

Потребитель детекции (:mod:`wordstat_trends.trends.growth`, #83): вход —
фразы со скорами факт/прогноз; ступень отсева (#84) стоит выше по
течению, её интерфейс не воспроизводится — отфильтрованность принимается
опциональным флагом ``filtered_out`` и такие записи в витрину не попадают.

Что делает модуль:

- Классификация: **растёт / сезонное / падает / стабильное**. Растёт и
  падает — тот же скор факт/прогноз против симметричных порогов;
  сезонное — устойчивая сезонная компонента без тренда (амплитуда
  месячного профиля); остальное — стабильное.
- Ранжирование внутри «растёт»: скор = веса × компоненты (аномальность,
  абсолютная частотность, свежесть начала роста). Все компоненты
  нормируются относительно самой фразы, а не популяции — поэтому скор
  фразы не зависит от соседей и ранжирование стабильно к добавлению
  одной фразы (требование issue #85).
- Выход — отсортированная структура данных без рендера: витрина —
  фаза #14, граница по образцу ``short_term.py`` («ядро выдаёт структуру
  данных, а не рендер»).

Параметры ``DECLINE_RATIO_THRESHOLD``, ``SEASONAL_MAX_MIN_RATIO``,
веса и ``VOLUME_TOP`` предрегистрированы в docs/TRENDS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from wordstat_trends.forecasting.baseline import SP, to_monthly_series
from wordstat_trends.trends.growth import (
    GROWTH_RATIO_THRESHOLD,
    GrowthScore,
    score_growth,
)

#: Порог «падает»: средний факт окна / средний прогноз <= 1/1.5 ≈ 0.67 —
#: симметрично порогу роста: «в полтора раза реже, чем год назад».
#: Предрегистрирован в docs/TRENDS.md.
DECLINE_RATIO_THRESHOLD = 1.0 / GROWTH_RATIO_THRESHOLD

#: Порог сезонности: самый сильный месяц месячного профиля минимум вдвое
#: чаще самого слабого. Меньше — колебания неотличимы от шума стабильного
#: ряда (у фикстуры «купить телефон» отношение ≈ 1.2, у «новогодние
#: подарки» — порядка 50). Предрегистрирован в docs/TRENDS.md.
SEASONAL_MAX_MIN_RATIO = 2.0

#: Верхняя планка абсолютной частотности для нормировки компоненты
#: объёма: звезда с миллионом запросов в месяц. Выше — компонента
#: насыщена (1.0): закупку интересует «достаточно ли людей», а не
#: различие миллиона от двух. Предрегистрирована в docs/TRENDS.md.
VOLUME_TOP = 1_000_000.0

#: Веса компонент скора роста. Аномальность — первичный сигнал: без
#: роста кандидата нет, поэтому половина веса. Частотность — треть
#: аномальности: звезда с 10 запросами не интересна закупке, но масштаб
#: не главнее самого роста. Свежесть — бонус: начавшийся раньше рост
#: всё ещё рост, поэтому минимальный вес. Предрегистрированы в
#: docs/TRENDS.md.
ANOMALY_WEIGHT = 0.5
VOLUME_WEIGHT = 0.3
FRESHNESS_WEIGHT = 0.2


class TrendClass(Enum):
    """Класс фразы для витрины (issue #85)."""

    GROWING = "растёт"
    SEASONAL = "сезонное"
    FALLING = "падает"
    STABLE = "стабильное"


#: Порядок классов в витрине: рост — главная ценность, сезонное —
#: планирование закупки по календарю, падение — сигнал сворачивать
#: позицию, стабильное — фон.
CLASS_ORDER = (
    TrendClass.GROWING,
    TrendClass.SEASONAL,
    TrendClass.FALLING,
    TrendClass.STABLE,
)


@dataclass(frozen=True)
class PhraseRecord:
    """Вход ранжирования: ряд фразы + её скор детекции.

    ``filtered_out`` — необязательная пометка от ступени отсева (#84,
    едет отдельным модулем выше по течению): отфильтрованные записи в
    витрину не попадают, интерфейс отсева здесь не воспроизводится.
    """

    phrase: str
    series: pd.Series  # месячный ряд из to_monthly_series
    growth: GrowthScore
    filtered_out: bool = False


@dataclass(frozen=True)
class ScoreComponents:
    """Компоненты скора роста, все в [0, 1] — для объяснимости на витрине."""

    anomaly: float  # 1 - 1/score: сила роста, монотонна по score
    volume: float  # log10(1+частотность) / log10(1+VOLUME_TOP)
    freshness: float  # 1 / возраст роста в месяцах
    growth_age_months: int  # сколько месяцев назад начался рост (>= 1)


@dataclass(frozen=True)
class RankedPhrase:
    """Строка витрины: фраза, класс, ранг, скор и его компоненты."""

    phrase: str
    klass: TrendClass
    rank: int  # 1..N в общем порядке витрины
    score: float  # взвешенная сумма компонент (только для «растёт»)
    components: ScoreComponents | None  # None вне класса «растёт»


def phrase_record(
    y: pd.Series, phrase: str = "", filtered_out: bool = False
) -> PhraseRecord:
    """Месячный ряд → запись для ранжирования (детекция внутри)."""

    return PhraseRecord(
        phrase=phrase,
        series=y,
        growth=score_growth(y, phrase=phrase),
        filtered_out=filtered_out,
    )


def phrase_record_from_frame(
    frame: pd.DataFrame, phrase: str | None = None, filtered_out: bool = False
) -> PhraseRecord:
    """DataFrame загрузчика (``period``/``queries``) → запись ранжирования."""

    resolved = frame.attrs.get("phrase") if phrase is None else phrase
    resolved = resolved if resolved is not None else ""
    y = to_monthly_series(frame)
    return PhraseRecord(
        phrase=resolved, series=y, growth=score_growth(y, phrase=resolved), filtered_out=filtered_out
    )


def seasonal_max_min_ratio(y: pd.Series) -> float:
    """Амплитуда месячного профиля: max / min средних по месяцам года.

    Месячный профиль усредняет каждый календарный месяц по всей истории
    (после MIN_HISTORY=24 их минимум два) — разовые всплески сглаживаются,
    остаётся устойчивая сезонная компонента.
    """

    month_means = y.groupby(pd.PeriodIndex(y.index).month).mean().astype(float)
    lo = float(month_means.min())
    if lo <= 0:
        # Нулевой месяц в профиле делит амплитуду на ноль; такой ряд
        # детекция роста уже отвергла бы (нулевой прогноз), здесь
        # трактуем как «сезонности нет».
        return 0.0
    return float(month_means.max()) / lo


def growth_age_months(y: pd.Series) -> int | None:
    """Сколько месяцев назад начался рост: длина хвоста месяцев,
    у которых значение >= 1.5 × значения год назад.

    Идём от конца ряда назад, пока месяц держит порог роста; первый
    месяц ниже порога — граница начала. None — порог не держит даже
    последний месяц (рост не подтверждён).
    """

    values = y.astype(float).to_numpy()
    age = 0
    for i in range(len(values) - 1, SP - 1, -1):
        year_ago = values[i - SP]
        if year_ago > 0 and values[i] >= GROWTH_RATIO_THRESHOLD * year_ago:
            age += 1
        else:
            break
    return age if age > 0 else None


def classify(record: PhraseRecord) -> TrendClass:
    """Класс фразы: рост/падение по скору, сезонность — по амплитуде."""

    score = record.growth.score
    if score >= GROWTH_RATIO_THRESHOLD:
        return TrendClass.GROWING
    if score <= DECLINE_RATIO_THRESHOLD:
        return TrendClass.FALLING
    if seasonal_max_min_ratio(record.series) >= SEASONAL_MAX_MIN_RATIO:
        return TrendClass.SEASONAL
    return TrendClass.STABLE


def growth_score_components(record: PhraseRecord) -> ScoreComponents:
    """Компоненты скора роста. Нормировка относительно самой фразы."""

    anomaly = 1.0 - 1.0 / record.growth.score
    # Объём — базовый уровень ряда (обучающая история до окна скоринга),
    # а не среднее окна: окно растущей фразы уже включает сам рост, и объём
    # по окну конфликат с аномальностью — две фразы с одинаковой базовой
    # частотностью, но разной силой роста получали бы разные volume
    # (корреляция компонент).
    base = record.series.iloc[: -len(record.growth.actual)]
    mean_volume = float(base.astype(float).mean())
    volume = min(math.log10(1.0 + mean_volume) / math.log10(1.0 + VOLUME_TOP), 1.0)
    age = growth_age_months(record.series)
    if age is None:
        # Классифицировано как рост, а хвост месяцев порога не держит —
        # окно скоринга сгладнило картину; считаем рост начавшимся в
        # пределах окна скоринга, свежесть максимальна по определению.
        age = 1
    freshness = 1.0 / age
    return ScoreComponents(
        anomaly=anomaly, volume=volume, freshness=freshness, growth_age_months=age
    )


def rank_showcase(records: list[PhraseRecord]) -> list[RankedPhrase]:
    """Записи фраз → отсортированная витрина (ранги 1..N).

    Порядок: классы в порядке CLASS_ORDER, внутри «растёт» — по скору
    убыванию, при равенстве — по фразе (детерминированность); вне
    «растёт» семантического ранжирования нет — по фразе. Скор фразы
    зависит только от её собственных данных, поэтому добавление одной
    фразы не меняет скоры и относительный порядок остальных.
    """

    kept = [r for r in records if not r.filtered_out]
    rows: list[tuple[TrendClass, float, str, ScoreComponents | None]] = []
    for record in kept:
        klass = classify(record)
        components = growth_score_components(record) if klass is TrendClass.GROWING else None
        score = (
            ANOMALY_WEIGHT * components.anomaly
            + VOLUME_WEIGHT * components.volume
            + FRESHNESS_WEIGHT * components.freshness
            if components is not None
            else 0.0
        )
        rows.append((klass, score, record.phrase, components))

    rows.sort(
        key=lambda row: (
            CLASS_ORDER.index(row[0]),
            -row[1],
            row[2],
        )
    )
    return [
        RankedPhrase(phrase=phrase, klass=klass, rank=i + 1, score=score, components=components)
        for i, (klass, score, phrase, components) in enumerate(rows)
    ]


__all__ = [
    "ANOMALY_WEIGHT",
    "CLASS_ORDER",
    "DECLINE_RATIO_THRESHOLD",
    "FRESHNESS_WEIGHT",
    "PhraseRecord",
    "RankedPhrase",
    "SEASONAL_MAX_MIN_RATIO",
    "ScoreComponents",
    "TrendClass",
    "VOLUME_TOP",
    "VOLUME_WEIGHT",
    "classify",
    "growth_age_months",
    "growth_score_components",
    "phrase_record",
    "phrase_record_from_frame",
    "rank_showcase",
    "seasonal_max_min_ratio",
]

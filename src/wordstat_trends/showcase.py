"""Сериализация витрины для сборки сайта (issue #104, фаза 5; ряд — #105, ниши — #106).

Граница по образцу ``short_term.py`` / ``ranking.py``: ядро выдаёт структуру
данных, рендер — фаза витрины. ``rank_showcase`` (#85) отдаёт
``list[RankedPhrase]`` с pandas-объектами — сборка сайта (Actions, без
зависимостей проекта) не может их прочитать, поэтому ядро сериализует
витрину в JSON-артефакт, который ``scripts/build_site.py`` кладёт в шаблоны.

Артефакт (``showcase/v2``):

.. code-block:: json

    {
      "schema": "showcase/v2",
      "generated_at": "2026-09-09",
      "niches": [
        {"topic": "...", "class": "GROWING", "score": 0.55,
         "phrases": [{"phrase": "...", "rank": 1}]}
      ],
      "phrases": [
        {"phrase": "...", "class": "GROWING", "rank": 1,
         "score": 0.62,
         "components": {"anomaly": ..., "volume": ..., "freshness": ...,
                        "growth_age_months": ...},
         "ratio": 2.0,
         "window_months": 3,
         "series": {"periods": ["2024-01", ...], "values": [101, ...]}}
      ]
    }

``class`` — имя члена :class:`TrendClass` (стабильный машинный ключ, не
русская отображаемая строка: витрина переводит класс ключом локали).
``components`` — ``None`` вне класса «растёт» (как в :class:`RankedPhrase`).

Ряд истории и окно скоринга (issue #105): ``ratio`` — отношение
факт/прогноз из :class:`GrowthScore` (в ``RankedPhrase`` попадает только
взвешенный скор витрины), ``series`` — месячный ряд целиком: витрина
показывает и историю, и окно скоринга (``window_months`` последних точек).
Читается и v1 (без ряда — карточки без графика и объяснения), пишется
только v2.

Ниши (issue #106, формулы предрегистрированы в docs/TRENDS.md): ниша =
кластер `Cluster` + доминирующий класс состава + агрегированный скор.
Считаются ядром на основном BERTA-конвейере и фиксируются в артефакте —
сборка сайта в Actions не касается extra `nlp` (выбор зафиксирован в
issue-комментарии #106). Продакшн-пути, который прогоняет кластеризацию
и пишет непустые ``niches`` (CLI/пайплайн сбора), в репо пока нет —
встраивание в конвейер артефакта относится к эпику #14; сейчас артефакт
с ``niches`` получается только прямыми вызовами (тесты). ``niches`` — всегда список (может быть пустым:
кластеризация не нашла ни одной фразы с лексическими токенами).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from wordstat_trends.nlp.clustering import ClusteringResult
from wordstat_trends.trends.ranking import (
    CLASS_ORDER,
    PhraseRecord,
    RankedPhrase,
    TrendClass,
    rank_showcase,
)

SCHEMA = "showcase/v2"

#: Схемы, которые читает `load_showcase`: v1 не содержит ряда и ratio —
#: витрина собирается, но без графиков и объяснений.
READABLE_SCHEMAS = frozenset({"showcase/v1", SCHEMA})

#: Сколько фраз состава ниши показывать на витрине (ссылки на карточки).
NICHE_TOP_PHRASES = 5


class ShowcaseError(ValueError):
    """Артефакт витрины не читается: не та схема или битая структура."""


@dataclass(frozen=True)
class Niche:
    """Ниша витрины: кластер + доминирующий класс + агрегированный скор.

    Формулы предрегистрированы в docs/TRENDS.md: доминирующий класс — мода
    классов состава (тай-брейк — CLASS_ORDER), скор — медиана скоров фраз
    доминирующего класса. ``members`` — фразы состава в порядке витрины.
    """

    top_lemmas: list[str]
    klass: TrendClass
    score: float
    members: list[RankedPhrase]
    # Метрики устойчивости ниши (issue #115): доля состава, отсеянного
    # ступенью #84, и медиана возраста роста состава. Считаются только
    # когда build_niches переданы записи до отсева (иначе дефолт:
    # «отсева нет», возраст — по составу витрины).
    filtered_share: float = 0.0
    median_growth_age_months: int | None = None

    @property
    def topic(self) -> str:
        """Метка-тема ниши: топ-леммы кластера через запятую."""
        return ", ".join(self.top_lemmas)


def build_niches(
    ranked: list[RankedPhrase],
    result: ClusteringResult,
    records: list[PhraseRecord] | None = None,
) -> list[Niche]:
    """Связка кластеров с классами трендов: кластеры + ранги → ниши.

    Кластеризируются все фразы сразу (см. docs/TRENDS.md); фразы кластера
    отображаются обратно на строки витрины — состав ниши упорядочен по
    рангу (интересные фразы сверху). Итог отсортирован: CLASS_ORDER, затем
    скор убыванием, затем метка-тема (детерминированность).

    ``records`` (записи **до** отсева #84, keyword-передача опциональна) —
    шов для метрик устойчивости ниши (#115): без него ``ranked`` знает
    только выжившие фразы, а кластер — весь свой состав; вместе они дают
    долю отфильтрованного состава. Конвейер #106 не меняется: без
    ``records`` метрики принимают дефолт («отсева нет»).
    """

    by_phrase = {row.phrase: row for row in ranked}
    by_record = {record.phrase: record for record in records} if records is not None else {}
    niches: list[Niche] = []
    for cluster in result.clusters:
        members = sorted(
            (by_phrase[phrase] for phrase in cluster.phrases if phrase in by_phrase),
            key=lambda row: row.rank,
        )
        if not members:
            continue
        # доминирующий класс: максимум частоты; тай-брейк — CLASS_ORDER
        counts = {klass: 0 for klass in CLASS_ORDER}
        for row in members:
            counts[row.klass] += 1
        dominant = min(CLASS_ORDER, key=lambda klass: (-counts[klass], CLASS_ORDER.index(klass)))
        dominant_scores = [row.score for row in members if row.klass is dominant]
        # устойчивость (#115): доля отфильтрованного состава — по кластеру
        # против записей до отсева; медиана возраста роста — по всем
        # растущим фразам состава (components не None только у «растёт»);
        # читается скорингом только для ниш с классом «растёт».
        total = sum(1 for phrase in cluster.phrases if phrase in by_record)
        filtered = sum(
            1
            for phrase in cluster.phrases
            if phrase in by_record and by_record[phrase].filtered_out
        )
        ages = [
            row.components.growth_age_months
            for row in members
            if row.components is not None
        ]
        niches.append(
            Niche(
                top_lemmas=cluster.top_lemmas,
                klass=dominant,
                score=float(statistics.median(dominant_scores)),
                members=members,
                filtered_share=(filtered / total) if total else 0.0,
                median_growth_age_months=(
                    int(statistics.median(ages)) if ages else None
                ),
            )
        )
    niches.sort(key=lambda niche: (CLASS_ORDER.index(niche.klass), -niche.score, niche.topic))
    return niches


def serialize_showcase(
    records: list[PhraseRecord],
    generated_at: date | None = None,
    *,
    result: ClusteringResult | None = None,
) -> dict:
    """Записи фраз (+ результат кластеризации) → словарь артефакта.

    Ранжирование — внутри: серии живут в :class:`PhraseRecord`, а не в
    :class:`RankedPhrase`, поэтому на вход принимаются записи (как в
    ``rank_showcase``); отфильтрованные ступенью отсева (#84) не попадают
    в витрину. ``result`` keyword-only: третий позиционный аргумент
    исторически был ``generated_at``, и позиционный вызов со старой
    сигнатурой молча скормил бы ``date`` в ``build_niches`` с невнятным
    AttributeError.
    """

    ranked = rank_showcase(records)
    by_phrase = {record.phrase: record for record in records}
    niches = build_niches(ranked, result, records) if result is not None else []
    return {
        "schema": SCHEMA,
        "generated_at": (generated_at or date.today()).isoformat(),
        "niches": [
            {
                "topic": niche.topic,
                "class": niche.klass.name,
                "score": round(niche.score, 6),
                "filtered_share": round(niche.filtered_share, 6),
                "median_growth_age_months": niche.median_growth_age_months,
                "phrases": [
                    {"phrase": row.phrase, "rank": row.rank}
                    for row in niche.members[:NICHE_TOP_PHRASES]
                ],
            }
            for niche in niches
        ],
        "phrases": [
            {
                "phrase": row.phrase,
                "class": row.klass.name,
                "rank": row.rank,
                "score": round(row.score, 6),
                "components": (
                    {
                        "anomaly": round(row.components.anomaly, 6),
                        "volume": round(row.components.volume, 6),
                        "freshness": round(row.components.freshness, 6),
                        "growth_age_months": row.components.growth_age_months,
                    }
                    if row.components is not None
                    else None
                ),
                "ratio": round(float(by_phrase[row.phrase].growth.score), 6),
                "window_months": len(by_phrase[row.phrase].growth.actual),
                "series": {
                    "periods": [str(p) for p in by_phrase[row.phrase].series.index],
                    "values": [
                        int(v) if float(v).is_integer() else round(float(v), 3)
                        for v in by_phrase[row.phrase].series
                    ],
                },
            }
            for row in ranked
        ],
    }


def save_showcase(
    records: list[PhraseRecord],
    path: Path | str,
    generated_at: date | None = None,
    *,
    result: ClusteringResult | None = None,
) -> Path:
    """Витрина (+ результат кластеризации) → файл артефакта."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            serialize_showcase(records, generated_at, result=result), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return target


def load_showcase(path: Path | str) -> dict:
    """Файл артефакта → словарь; схема и структура проверяются, не молча.

    v1 читается для обратной совместимости (карточки без ряда); всё, что
    не v1/v2, — ошибка: молча показать пустое состояние поверх
    существующих данных хуже громкого отказа.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") not in READABLE_SCHEMAS:
        raise ShowcaseError(
            f"не артефакт витрины (ожидается одна из схем {sorted(READABLE_SCHEMAS)})"
        )
    phrases = data.get("phrases")
    if not isinstance(phrases, list):
        raise ShowcaseError("артефакт без списка phrases")
    niches = data.get("niches", [])
    if not isinstance(niches, list):
        raise ShowcaseError("артефакт с некорректным списком niches")
    return data


__all__ = [
    "NICHE_TOP_PHRASES",
    "Niche",
    "READABLE_SCHEMAS",
    "SCHEMA",
    "ShowcaseError",
    "build_niches",
    "load_showcase",
    "save_showcase",
    "serialize_showcase",
]

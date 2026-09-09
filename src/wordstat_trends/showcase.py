"""Сериализация витрины для сборки сайта (issue #104, фаза 5).

Граница по образцу ``short_term.py`` / ``ranking.py``: ядро выдаёт структуру
данных, рендер — фаза витрины. ``rank_showcase`` (#85) отдаёт
``list[RankedPhrase]`` с pandas-объектами — сборка сайта (Actions, без
зависимостей проекта) не может их прочитать, поэтому ядро сериализует
витрину в JSON-артефакт, который ``scripts/build_site.py`` кладёт в шаблоны.

Артефакт (``showcase/v1``):

.. code-block:: json

    {
      "schema": "showcase/v1",
      "generated_at": "2026-09-09",
      "phrases": [
        {"phrase": "...", "class": "GROWING", "rank": 1,
         "score": 0.62,
         "components": {"anomaly": ..., "volume": ..., "freshness": ...,
                        "growth_age_months": ...}}
      ]
    }

``class`` — имя члена :class:`TrendClass` (стабильный машинный ключ, не
русская отображаемая строка: витрина переводит класс ключом локали).
``components`` — ``None`` вне класса «растёт» (как в :class:`RankedPhrase`).
Ряд истории и окно скоринга в артефакт не входят — это #105.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from wordstat_trends.trends.ranking import RankedPhrase

SCHEMA = "showcase/v1"


class ShowcaseError(ValueError):
    """Артефакт витрины не читается: не та схема или битая структура."""


def serialize_showcase(
    ranked: list[RankedPhrase], generated_at: date | None = None
) -> dict:
    """Отранжированная витрина → JSON-совместимый словарь артефакта."""

    return {
        "schema": SCHEMA,
        "generated_at": (generated_at or date.today()).isoformat(),
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
            }
            for row in ranked
        ],
    }


def save_showcase(
    ranked: list[RankedPhrase], path: Path | str, generated_at: date | None = None
) -> Path:
    """Витрина → файл артефакта (каталог результатов сбора)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(serialize_showcase(ranked, generated_at), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_showcase(path: Path | str) -> dict:
    """Файл артефакта → словарь; схема и структура проверяются, не молча."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ShowcaseError(f"не артефакт витрины (ожидалась схема {SCHEMA})")
    phrases = data.get("phrases")
    if not isinstance(phrases, list):
        raise ShowcaseError("артефакт без списка phrases")
    return data


__all__ = ["SCHEMA", "ShowcaseError", "load_showcase", "save_showcase", "serialize_showcase"]

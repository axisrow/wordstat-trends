"""Загрузка run-каталога wordstat-cli в DataFrame (issue #61, Фаза 1).

Run-каталог — то, что оставляет после себя ``wordstat collect --keep-raw``:
``manifest.json`` (метаданные прогона: ``phrase``, ``region``, ``created_at``,
выведенные ``dtypes``) и сырые CSV представлений. Этот модуль — слой чтения
уже собранных данных: свой формат не изобретается, разбор CSV и периодов
переиспользует :mod:`wordstat_trends.collect.stitch` (формат — фактический,
из замера #2, ``docs/DATA.md``: UTF-8 с BOM, CR-only переводы строк — отсюда
``splitlines()``, а не чтение по ``\\n``; разделитель ``;``, десятичная
запятая, тысячи через ASCII-пробел).

Пустой датасет (0 строк данных — как ``top_popular``/``top_related`` во всех
трёх прогонах замера #2, баг сборщика) не принимается молча за валидный:
``EmptyDatasetError``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from wordstat.models import CollectionManifest

from wordstat_trends.collect.stitch import parse_period, parse_queries

DYNAMICS_VIEW = "dynamics"


class LoaderError(Exception):
    """Базовая ошибка загрузки run-каталога."""


class ManifestError(LoaderError):
    """manifest.json отсутствует или не валиден."""


class EmptyDatasetError(LoaderError):
    """Датасет представления не содержит ни одной строки данных."""


@dataclass(frozen=True)
class RunMetadata:
    """Метаданные прогона из manifest.json — кладутся в ``df.attrs``."""

    phrase: str
    region: str
    created_at: datetime
    dtypes: dict[str, str]


def load_manifest(run_directory: Path) -> tuple[CollectionManifest, RunMetadata]:
    """Прочитать manifest.json run-каталога в метаданные загрузчика."""

    path = run_directory / "manifest.json"
    if not path.is_file():
        raise ManifestError(f"нет manifest.json в {run_directory}")
    try:
        manifest = CollectionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ManifestError(f"{path} не является валидным манифестом wordstat: {exc}") from exc

    dtypes: dict[str, str] = {}
    for export in manifest.exports:
        if export.view == DYNAMICS_VIEW:
            dtypes = dict(export.dtypes)
    metadata = RunMetadata(
        phrase=manifest.phrase,
        region=manifest.region,
        created_at=manifest.created_at,
        dtypes=dtypes,
    )
    return manifest, metadata


def parse_dynamics_rows(text: str) -> list[tuple[str, int, float]]:
    """Строки CSV динамики → (метка периода, число запросов, доля %).

    Отличается от :func:`wordstat_trends.collect.stitch.parse_dynamics_csv`
    только сохранением третьей колонки «Доля от всех запросов, %» (склейке
    окон она не нужна, EDA — да). Пустой датасет здесь — ``EmptyDatasetError``,
    а не «молча пустой список».
    """

    # BOM здесь режется повторно (utf-8-sig в load_run его уже снял) — защита
    # для прямых вызовов с сырым текстом; литерал ниже — невидимый U+FEFF.
    reader = csv.reader(io.StringIO(text.lstrip("﻿"), newline=""), delimiter=";")
    rows = list(reader)
    if not rows:
        raise EmptyDatasetError("пустой CSV динамики")

    header = [cell.strip().lower() for cell in rows[0]]
    try:
        period_i = header.index("период")
        queries_i = header.index("число запросов")
        share_i = header.index("доля от всех запросов, %")
    except ValueError as exc:
        raise LoaderError(f"неожиданный заголовок динамики: {rows[0]!r}") from exc

    parsed: list[tuple[str, int, float]] = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        try:
            year, month = parse_period(row[period_i])
            share = float(row[share_i].strip().replace(",", "."))
        except (ValueError, IndexError) as exc:
            raise LoaderError(f"не удалось разобрать строку динамики: {row!r}") from exc
        parsed.append((f"{year}-{month:02d}", parse_queries(row[queries_i]), share))

    if not parsed:
        raise EmptyDatasetError("датасет dynamics не содержит строк данных (см. docs/DATA.md: top_popular/top_related)")
    return parsed


def load_run(run_directory: Path, view: str = DYNAMICS_VIEW) -> pd.DataFrame:
    """Run-каталог → DataFrame с метаданными прогона в ``df.attrs``.

    Колонки: ``period`` (``YYYY-MM``, месячная метка), ``queries`` (int64),
    ``share_pct`` (float, доля от всех запросов в процентах). Данные
    сортируются по периоду.
    """

    if view != DYNAMICS_VIEW:
        raise LoaderError(f"представление {view!r} не поддерживается загрузчиком (только dynamics)")

    _manifest, metadata = load_manifest(run_directory)
    csv_path = run_directory / f"{view}.csv"
    if not csv_path.is_file():
        raise LoaderError(f"нет {view}.csv в {run_directory} (нужен прогон с --keep-raw)")

    rows = parse_dynamics_rows(csv_path.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"]).sort_values("period").reset_index(drop=True)
    frame.attrs["phrase"] = metadata.phrase
    frame.attrs["region"] = metadata.region
    frame.attrs["created_at"] = metadata.created_at
    frame.attrs["dtypes"] = metadata.dtypes
    return frame


__all__ = [
    "DYNAMICS_VIEW",
    "EmptyDatasetError",
    "LoaderError",
    "ManifestError",
    "RunMetadata",
    "load_manifest",
    "load_run",
    "parse_dynamics_rows",
]

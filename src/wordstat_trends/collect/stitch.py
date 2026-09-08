"""Склейка перекрывающихся окон динамики в один непрерывный ряд (issue #6).

Полный ряд с января 2018 одной выгрузкой не взять: окно одного запроса —
максимум 60 месяцев (см. ``wordstat_trends.collect.config``). Склейка:

1. окна запрашиваются с нахлёстом в несколько месяцев;
2. значения в зоне нахлёста сверяются по колонке «Число запросов» —
   расхождение означает смену методики Вордстата, поэтому ``падаем``
   (``WindowMismatchError``), а не усредняем;
3. непрерывность итогового ряда обязательна: дыра молча ломает ``sp=12``
   (``SeriesGapError``).

Формат CSV — фактический, из замера #2 (``docs/DATA.md``): UTF-8 c BOM,
разделитель полей ``;``, период «август 2024», тысячи через ASCII-пробел,
CR-only переводы строк, четвёртое поле пустое.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from wordstat_trends.collect.errors import SeriesGapError, WindowMismatchError

#: Насколько месяцев окна обязаны перекрываться, чтобы нахлёст можно было
#: сверить. Минимальное окно интерфейса — 3 месяца, поэтому двойной нахлёст
#: ничего не стоит.
DEFAULT_MIN_OVERLAP = 2

_MONTHS_RU = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}

Period = tuple[int, int]  # (год, месяц)


def month_key(year: int, month: int) -> int:
    """Месяц как монотонное число — для сравнений и шага непрерывности."""
    return year * 12 + (month - 1)


@dataclass(frozen=True)
class MonthlyPoint:
    """Одна месячная точка ряда."""

    year: int
    month: int
    queries: int

    @property
    def period(self) -> Period:
        return (self.year, self.month)

    @property
    def key(self) -> int:
        return month_key(self.year, self.month)

    @property
    def label(self) -> str:
        months = {v: k for k, v in _MONTHS_RU.items()}
        return f"{months[self.month]} {self.year}"


def parse_period(raw: str) -> Period:
    """«август 2024» → (2024, 8). Русское название месяца текстом, не дата."""
    name, _, year = raw.strip().rpartition(" ")
    month = _MONTHS_RU.get(name.strip().lower())
    if month is None or not year.isdigit():
        raise ValueError(f"неожиданный формат периода: {raw!r} (ожидалось «август 2024»)")
    return (int(year), month)


def parse_queries(raw: str) -> int:
    """«1 234 632» → 1234632. Разделитель тысяч — ASCII-пробел (замер #2)."""
    digits = raw.strip().replace(" ", "")
    if not digits.isdigit():
        raise ValueError(f"неожиданный формат числа запросов: {raw!r}")
    return int(digits)


def parse_dynamics_csv(text: str) -> list[MonthlyPoint]:
    """Разобрать CSV динамики Вордстата в список месячных точек.

    Принимает сырой текст файла как есть: BOM снимается, CR-only переводы
    строк обрабатывает ``splitlines()``, четвёртое (пустое) поле игнорируется.
    Сверка нахлёста идёт по колонке «Число запросов» — не «Показов» (замер #2).
    """
    reader = csv.reader(io.StringIO(text.lstrip("﻿"), newline=""), delimiter=";")
    rows = list(reader)
    if not rows:
        raise ValueError("пустой CSV динамики")

    header = [cell.strip().lower() for cell in rows[0]]
    try:
        period_i = header.index("период")
        queries_i = header.index("число запросов")
    except ValueError as exc:
        raise ValueError(f"неожиданный заголовок динамики: {rows[0]!r}") from exc

    points = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        try:
            year, month = parse_period(row[period_i])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"не удалось разобрать строку динамики: {row!r}") from exc
        points.append(MonthlyPoint(year, month, parse_queries(row[queries_i])))
    return points


def _dedupe(points: Sequence[MonthlyPoint]) -> dict[Period, MonthlyPoint]:
    """Убрать дубликаты периода внутри одного окна: совпадающие значения
    схлопываются, расходящиеся — WindowMismatchError ещё до склейки окон."""
    by_period: dict[Period, MonthlyPoint] = {}
    for p in points:
        seen = by_period.get(p.period)
        if seen is not None and seen.queries != p.queries:
            raise WindowMismatchError(p.period, seen.queries, p.queries)
        by_period[p.period] = p
    return by_period


def stitch(  # noqa: PLR0912 — линейный конвейер проверок, ветвление — часть контракта ошибок
    windows: Sequence[Sequence[MonthlyPoint]],
    *,
    expected_from: Period | None = None,
    expected_to: Period | None = None,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> list[MonthlyPoint]:
    """Склеить перекрывающиеся окна в один непрерывный месячный ряд.

    Порядок окон не важен. В зоне нахлёста значения обязаны совпасть
    (расхождение → ``WindowMismatchError`` с периодом и обоими значениями).
    Итог обязан быть непрерывным по месяцам (дыра → ``SeriesGapError``);
    если задан ожидаемый диапазон, ряд обязан покрывать его целиком —
    «ряд короче ожидаемого» тоже ``SeriesGapError``.

    Возвращает точки, отсортированные от старого к новому.
    """
    if not windows:
        raise SeriesGapError(None, None, "не передано ни одного окна")

    ordered = [dict(sorted(_dedupe(w).items())) for w in windows]
    ordered.sort(key=lambda w: next(iter(w)))
    merged: dict[Period, MonthlyPoint] = dict(ordered[0])

    for window in ordered[1:]:
        overlap = sorted(set(window) & set(merged))
        if len(overlap) < min_overlap:
            edge = max(merged)
            first = min(window)
            raise SeriesGapError(
                edge,
                first,
                f"окна не перекрываются: нахлёст {len(overlap)} мес. < {min_overlap} (окно {first}…{max(window)})",
            )
        for period in overlap:
            if merged[period].queries != window[period].queries:
                raise WindowMismatchError(period, merged[period].queries, window[period].queries)
        for period, point in window.items():
            if period not in merged:
                merged[period] = point

    points = [merged[p] for p in sorted(merged)]

    # Непрерывность: соседние месяцы обязаны отличаться ровно на один месяц.
    for prev, cur in zip(points, points[1:]):
        if cur.key - prev.key != 1:
            raise SeriesGapError(prev.period, cur.period, f"пропущено {cur.key - prev.key - 1} мес.")

    # Покрытие ожидаемого диапазона: дыра на краю — тоже дыра («ряд короче
    # ожидаемого» из тест-списка issue #6).
    if expected_from is not None and month_key(*expected_from) < points[0].key:
        raise SeriesGapError(None, points[0].period, f"ожидается ряд с {expected_from[1]:02d}.{expected_from[0]}")
    if expected_to is not None and month_key(*expected_to) > points[-1].key:
        raise SeriesGapError(points[-1].period, None, f"ожидается ряд по {expected_to[1]:02d}.{expected_to[0]}")

    return points


def stitch_csvs(texts: Iterable[str], **kwargs) -> list[MonthlyPoint]:
    """Удобство: склеить сразу сырые CSV-тексты выгрузок."""
    return stitch([parse_dynamics_csv(t) for t in texts], **kwargs)


__all__ = [
    "DEFAULT_MIN_OVERLAP",
    "MonthlyPoint",
    "Period",
    "month_key",
    "parse_dynamics_csv",
    "parse_period",
    "parse_queries",
    "stitch",
    "stitch_csvs",
]

"""Склейка окон динамики (issue #6).

Пять обязательных кейсов из формулировки: чистая склейка, расхождение в
нахлёсте, дыра в середине, дыра на стыке окон, ряд короче ожидаемого.
Плюс разбор фактического формата CSV из замера #2.
"""

import pytest

from wordstat_trends.collect.errors import SeriesGapError, WindowMismatchError
from wordstat_trends.collect.stitch import (
    MonthlyPoint,
    parse_dynamics_csv,
    parse_period,
    parse_queries,
    stitch,
)


def points(start_year: int, start_month: int, count: int, *, base: int = 100) -> list[MonthlyPoint]:
    """Последовательные месячные точки со значениями base + key (детерминизм)."""
    total = start_year * 12 + (start_month - 1)
    return [MonthlyPoint((total + i) // 12, (total + i) % 12 + 1, base + total + i) for i in range(count)]


def window_csv(points_: list[MonthlyPoint]) -> str:
    """CSV в фактическом формате замера #2: BOM, CR-only, завершающая `;`."""
    header = "Период;Число запросов;Доля от всех запросов, %;Заголовок графика"
    rows = []
    for p in points_:
        share = f"{p.queries / 10**8:.5f}".replace(".", ",")
        rows.append(f"{p.label};{p.queries:,}".replace(",", " ") + f";{share};")
    return "﻿" + "\r".join([header, *rows]) + "\r"


# --- разбор фактического формата -------------------------------------------------


def test_parse_period_russian_month():
    assert parse_period("август 2024") == (2024, 8)
    assert parse_period("январь 2018") == (2018, 1)
    assert parse_period(" декабрь 2026 ") == (2026, 12)


def test_parse_period_rejects_non_month_text():
    with pytest.raises(ValueError, match="формат периода"):
        parse_period("01.2024")


def test_parse_queries_ascii_space_thousands():
    assert parse_queries("1 234 632") == 1234632
    assert parse_queries("997 977") == 997977
    assert parse_queries("42") == 42
    with pytest.raises(ValueError, match="числа запросов"):
        parse_queries("1,5")


def test_parse_dynamics_csv_real_format():
    text = window_csv(points(2024, 8, 3))
    rows = parse_dynamics_csv(text)
    assert [(p.year, p.month, p.queries) for p in rows] == [(2024, 8, 24395), (2024, 9, 24396), (2024, 10, 24397)]


# --- пять обязательных кейсов ------------------------------------------------------


def test_clean_stitch_two_overlapping_windows():
    """Чистая склейка: 36 месяцев из двух окон с нахлёстом в 3 месяца."""
    old = points(2024, 8, 15)  # 2024-08 … 2025-10
    new = points(2024, 8, 36)  # 2024-08 … 2027-07
    merged = stitch([old, new])
    assert len(merged) == 36
    assert (merged[0].year, merged[0].month) == (2024, 8)
    assert (merged[-1].year, merged[-1].month) == (2027, 7)
    assert all(p.queries == 100 + p.key for p in merged)


def test_mismatch_in_overlap_raises_with_both_values():
    """Расхождение в нахлёсте → WindowMismatchError с периодом и обоими значениями."""
    old = points(2024, 8, 15)
    new = points(2024, 8, 36)
    conflicting = MonthlyPoint(2025, 3, 999)  # тот же период, другое число
    new = [conflicting if (p.year, p.month) == (2025, 3) else p for p in new]

    with pytest.raises(WindowMismatchError) as excinfo:
        stitch([old, new])
    assert excinfo.value.period == (2025, 3)
    assert excinfo.value.left == 24402  # 100 + month_key(2025, 3) == 100 + 2025*12 + 2
    assert excinfo.value.right == 999
    assert "03.2025" in str(excinfo.value)
    assert "24402" in str(excinfo.value) and "999" in str(excinfo.value)


def test_gap_in_middle_raises():
    """Дыра в середине одного окна: пропущен месяц → SeriesGapError с краями."""
    pts = points(2024, 8, 5)  # 2024-08 … 2024-12
    holed = pts[:2] + pts[3:]  # выброшен 2024-10

    with pytest.raises(SeriesGapError) as excinfo:
        stitch([holed])
    assert excinfo.value.before == (2024, 9)
    assert excinfo.value.after == (2024, 11)


def test_gap_at_window_joint_raises():
    """Дыра на стыке: второе окно начинается позже конца первого без перекрытия."""
    old = points(2024, 8, 12)  # … 2025-07
    new = points(2025, 10, 6)  # с 2025-10, окно 2025-07…2025-09 потеряно

    with pytest.raises(SeriesGapError) as excinfo:
        stitch([old, new])
    assert excinfo.value.before == (2025, 7)
    assert excinfo.value.after == (2025, 10)
    assert "нахлёст" in str(excinfo.value)


def test_series_shorter_than_expected_raises():
    """Ряд короче ожидаемого: старт позже ожидаемого края → SeriesGapError."""
    old = points(2019, 1, 15)
    new = points(2019, 1, 60)

    with pytest.raises(SeriesGapError) as excinfo:
        stitch([old, new], expected_from=(2018, 1))
    assert excinfo.value.after == (2019, 1)

    with pytest.raises(SeriesGapError) as excinfo:
        stitch([old, new], expected_from=(2019, 1), expected_to=(2025, 1))
    assert excinfo.value.before == (2023, 12)  # points(2019, 1, 60) заканчивается 2023-12


def test_stitch_bridging_window_between_two_disjoint_windows():
    """Окна A и B не пересекаются, но C мостит их: порядок не важен и
    цепочность не требуется — A/B не должны уронить SeriesGapError до C."""
    a = points(2024, 1, 10)  # 2024-01 … 2024-10
    b = points(2025, 8, 10)  # 2025-08 … 2026-05
    c = points(2024, 8, 15)  # 2024-08 … 2025-10 — мост
    merged = stitch([a, b, c])
    assert (merged[0].year, merged[0].month) == (2024, 1)
    assert (merged[-1].year, merged[-1].month) == (2026, 5)
    assert len(merged) == 29
    # И в произвольном порядке подачи.
    assert stitch([b, c, a]) == merged


def test_unreachable_window_still_raises():
    """Окно, которое ничем не мостится, остаётся дырой в любом порядке."""
    a = points(2024, 1, 10)
    far = points(2030, 1, 5)
    with pytest.raises(SeriesGapError, match="нахлёст"):
        stitch([a, far])


def test_stitch_requires_windows():
    with pytest.raises(SeriesGapError):
        stitch([])


def test_stitch_min_overlap_is_more_than_one_month():
    """Нахлёст в один месяц не принимается: DEFAULT_MIN_OVERLAP = 2 (сверить
    смену методики по одной точке нельзя)."""
    old = points(2024, 8, 12)  # … 2025-07
    new = points(2025, 7, 6)  # ровно один общий месяц — 2025-07

    with pytest.raises(SeriesGapError, match="нахлёст"):
        stitch([old, new])


def test_stitch_csvs_end_to_end_with_real_fixture():
    """Реальная фикстура из замера #2 + синтетическое старое окно: 24 + нахлёст."""
    from pathlib import Path

    fixture = (Path(__file__).resolve().parents[1] / "tests/fixtures/dynamics_seasonal.csv").read_text(encoding="utf-8")
    recent = parse_dynamics_csv(fixture)
    assert len(recent) == 24  # дефолтное окно платформы

    # Старое окно: 22 месяца, завершающиеся на 2 месяца позже старта нового
    # окна → нахлёст ≥ 2 месяцев. Значения в нахлёсте повторяем точно.
    first = recent[0]
    overlap_total = first.year * 12 + (first.month - 1) + 2
    synthetic = points(2022, 7, overlap_total - (2022 * 12 + 6))
    by_period = {(p.year, p.month): p for p in recent}
    synthetic = [by_period.get((p.year, p.month), p) for p in synthetic]

    merged = stitch([synthetic, recent])
    assert len(merged) == len(synthetic) + 24 - 2  # нахлёст — ровно 2 месяца
    assert (merged[0].year, merged[0].month) == (2022, 7)
    assert (merged[-1].year, merged[-1].month) == (2026, 7)

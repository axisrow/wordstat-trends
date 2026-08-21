"""Дневной парсер и нарезка окна — на синтетических мини-примерах.

Синтетика здесь допустима намеренно: проверяется разбор формата и арифметика
календаря, а не поведение спроса. В самом эксперименте синтетических данных
быть не должно.
"""

from pathlib import Path

import pandas as pd
import pytest

from wordstat_trends.daily_window import (
    SP,
    TRAIN_DAYS,
    check_continuous,
    make_window,
    weekday_counts,
)
from wordstat_trends.dynamics_io import load_dynamics

# Заголовок дневной выгрузки: первая колонка «Дата», не «Период».
HEADER = "Дата;Число запросов;Доля от всех запросов, %;Динамика ... 21.08.2026;"


def write_daily(path: Path, start: str, days: int, *, skip: set[int] = frozenset()) -> Path:
    """Пишет дневной CSV в точности как Вордстат: BOM, CR-only, ';'."""
    rows = [HEADER]
    for offset in range(days):
        if offset in skip:
            continue
        day = pd.Timestamp(start) + pd.Timedelta(days=offset)
        rows.append(f"{day.strftime('%d.%m.%Y')};{1000 + offset};0,0011;")
    path.write_bytes(("\r".join(rows) + "\r").encode("utf-8-sig"))
    return path


def test_parses_dotted_dates_not_russian_months(tmp_path):
    series = load_dynamics(write_daily(tmp_path / "d.csv", "2026-06-23", 5), granularity="daily")
    assert len(series) == 5
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index[0] == pd.Timestamp("2026-06-23")
    assert series.iloc[0] == 1000


def test_day_and_month_are_not_swapped(tmp_path):
    """`06.07.2026` — 6 июля, а не 7 июня. Формат задан явно, не по локали."""
    path = tmp_path / "d.csv"
    path.write_bytes((f"{HEADER}\r06.07.2026;500;0,001;\r").encode("utf-8-sig"))
    series = load_dynamics(path, granularity="daily")
    assert series.index[0] == pd.Timestamp("2026-07-06")


def test_daily_parser_rejects_russian_month(tmp_path):
    """Месячную выгрузку нельзя молча прочитать как дневную."""
    path = tmp_path / "d.csv"
    path.write_bytes((f"{HEADER}\rавгуст 2024;500;0,001;\r").encode("utf-8-sig"))
    with pytest.raises(ValueError, match="DD.MM.YYYY"):
        load_dynamics(path, granularity="daily")


def test_daily_rejects_monthly_header(tmp_path):
    """«Период» — месячный заголовок; дневная выгрузка называет колонку «Дата»."""
    path = tmp_path / "d.csv"
    path.write_bytes(
        ("Период;Число запросов;Доля;\r23.06.2026;1;0,1;\r").encode("utf-8-sig")
    )
    with pytest.raises(ValueError, match="заголовок"):
        load_dynamics(path, granularity="daily")


def test_monthly_parser_still_works_on_real_fixture():
    """Дневная ветка не сломала месячную."""
    series = load_dynamics(Path(__file__).parent / "fixtures" / "dynamics_seasonal.csv")
    assert len(series) == 24
    assert series.index[0] == pd.Period("2024-08", freq="M")


def test_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "d.csv"
    path.write_bytes((f"{HEADER}\r23.06.2026;1;0,1;\r23.06.2026;2;0,1;\r").encode("utf-8-sig"))
    with pytest.raises(ValueError, match="повторяются"):
        load_dynamics(path, granularity="daily")


def test_window_of_56_days_is_balanced(tmp_path):
    """23.06.2026 (вт) + 56 дней → каждый день недели ровно 8 раз."""
    series = load_dynamics(write_daily(tmp_path / "d.csv", "2026-06-23", 58), granularity="daily")
    window = make_window(series)
    assert len(window.train) == TRAIN_DAYS
    assert window.weekday_counts == [8] * SP
    assert window.train.index[0] == pd.Timestamp("2026-06-23")
    assert window.train.index[-1] == pd.Timestamp("2026-08-17")


def test_holdout_is_whatever_remains(tmp_path):
    """58 строк → holdout 2 дня, а не 59 − 56 = 3."""
    series = load_dynamics(write_daily(tmp_path / "d.csv", "2026-06-23", 58), granularity="daily")
    window = make_window(series)
    assert len(window.holdout) == 2
    assert list(window.holdout.index.strftime("%d.%m")) == ["18.08", "19.08"]


def test_unbalanced_window_is_rejected(tmp_path):
    """Окно не кратно семи — профиль смещён, дальше идти нельзя."""
    series = load_dynamics(write_daily(tmp_path / "d.csv", "2026-06-23", 58), granularity="daily")
    with pytest.raises(ValueError, match="не кратна|смещён"):
        make_window(series, train_days=50)


def test_gap_in_dates_is_rejected(tmp_path):
    """Дыра сдвигает весь хвост — детренд считает соседние строки соседними днями."""
    series = load_dynamics(
        write_daily(tmp_path / "d.csv", "2026-06-23", 58, skip={10}), granularity="daily"
    )
    with pytest.raises(ValueError, match="пропущены"):
        make_window(series)


def test_short_series_is_rejected_not_padded(tmp_path):
    series = load_dynamics(write_daily(tmp_path / "d.csv", "2026-06-23", 40), granularity="daily")
    with pytest.raises(ValueError, match="Достраивать"):
        make_window(series)


def test_weekday_counts_orders_monday_first(tmp_path):
    series = load_dynamics(write_daily(tmp_path / "d.csv", "2026-06-22", 7), granularity="daily")
    assert weekday_counts(series) == [1] * SP
    assert series.index[0].dayofweek == 0


def test_check_continuous_accepts_clean_series(tmp_path):
    series = load_dynamics(write_daily(tmp_path / "d.csv", "2026-06-23", 14), granularity="daily")
    check_continuous(series)

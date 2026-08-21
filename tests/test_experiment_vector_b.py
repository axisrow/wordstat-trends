from datetime import date, timedelta
from pathlib import Path

import pytest

from wordstat_trends.experiment_vector_b import (
    load_wordstat_daily,
    parse_daily_period,
    parse_request_count,
    split_experiment_window,
)


def _write_daily_fixture(path: Path, *, rows: int = 59) -> None:
    start = date(2026, 6, 23)
    lines = ["Период;Число запросов;Доля, %;График"]
    for offset in range(rows):
        current = start + timedelta(days=offset)
        lines.append(f"{current:%d.%m.%Y};1 234,5;0,5;x")
    path.write_bytes(("\ufeff" + "\r".join(lines) + "\r").encode())


def test_parses_measured_daily_fields() -> None:
    assert parse_daily_period("22.06.2026").date() == date(2026, 6, 22)
    assert parse_request_count("1 234,5") == 1234.5


@pytest.mark.parametrize("value", ["2026-06-22", "22 июня 2026", "июнь 2026"])
def test_rejects_non_daily_period(value: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Wordstat daily period"):
        parse_daily_period(value)


def test_loads_bom_cr_only_and_validates_balanced_training_window(tmp_path: Path) -> None:
    path = tmp_path / "daily.csv"
    _write_daily_fixture(path)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" in raw and b"\n" not in raw

    series = load_wordstat_daily(path)
    train, holdout = split_experiment_window(series)

    assert len(series) == 59
    assert len(train) == 56
    assert len(holdout) == 3
    assert train.index[0].date() == date(2026, 6, 23)
    assert train.index[-1].date() == date(2026, 8, 17)
    assert train.groupby(train.index.dayofweek).size().tolist() == [8] * 7
    assert series.iloc[0] == 1234.5


def test_rejects_short_daily_export_without_padding(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    _write_daily_fixture(path, rows=58)

    series = load_wordstat_daily(path)
    with pytest.raises(ValueError, match="Expected 59 real daily rows, got 58"):
        split_experiment_window(series)

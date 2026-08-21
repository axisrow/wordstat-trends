"""Разбор сырого CSV Вордстата — против реальных фикстур, а не синтетики.

Формат враждебен наивному чтению (BOM, CR-only переводы строк, десятичная
запятая, русские месяцы текстом), поэтому проверяется дословно; см.
`docs/DATA.md`.
"""

from pathlib import Path

import pandas as pd
import pytest

from wordstat_trends.dynamics_io import load_dynamics

FIXTURES = Path(__file__).parent / "fixtures"
ALL_FIXTURES = ("dynamics_seasonal.csv", "dynamics_high_freq.csv", "dynamics_mid_freq.csv")


def test_fixture_is_actually_bom_and_cr_only():
    """Страховка на сами фикстуры: если формат «починят», тесты ниже врут."""
    raw = (FIXTURES / "dynamics_seasonal.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "BOM пропал — фикстура уже не сырая"
    assert b"\r" in raw
    assert b"\n" not in raw, "появился LF — фикстура больше не CR-only"


@pytest.mark.parametrize("filename", ALL_FIXTURES)
def test_loads_twenty_four_monthly_points(filename):
    series = load_dynamics(FIXTURES / filename)
    assert len(series) == 24
    assert isinstance(series.index, pd.PeriodIndex)
    assert series.index.freqstr == "M"
    assert series.index[0] == pd.Period("2024-08", freq="M")
    assert series.index[-1] == pd.Period("2026-07", freq="M")


@pytest.mark.parametrize("filename", ALL_FIXTURES)
def test_index_is_sorted_and_gapless(filename):
    series = load_dynamics(FIXTURES / filename)
    assert series.index.is_monotonic_increasing
    expected = pd.period_range(series.index[0], series.index[-1], freq="M")
    assert list(series.index) == list(expected)


def test_parses_thousands_separator_and_known_values():
    """Числа вида `36 806` — с обычным пробелом внутри."""
    series = load_dynamics(FIXTURES / "dynamics_seasonal.csv")
    assert series.iloc[0] == 36806
    assert series.loc[pd.Period("2024-12", freq="M")] == 1234632
    assert series.iloc[-1] == 29017
    assert series.dtype.kind == "i"


def test_december_peak_survives_parsing():
    """Сезонный сигнал виден в сырых числах — если нет, разбор что-то потерял."""
    series = load_dynamics(FIXTURES / "dynamics_seasonal.csv")
    assert series.max() / series.min() > 60
    assert series.idxmax().month == 12


def test_rejects_file_with_unexpected_header(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text("Период;Показов;\rавгуст 2024;1;\r", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="заголовок"):
        load_dynamics(broken)


def test_rejects_unknown_month(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text(
        "Период;Число запросов;Доля;\rавгуста 2024;1;\r",
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError, match="месяц"):
        load_dynamics(broken)


def test_sorts_rows_that_arrive_out_of_order(tmp_path):
    shuffled = tmp_path / "shuffled.csv"
    shuffled.write_text(
        "Период;Число запросов;Доля;\r"
        "декабрь 2024;3;\r"
        "август 2024;1;\r"
        "октябрь 2024;2;\r",
        encoding="utf-8-sig",
    )
    series = load_dynamics(shuffled)
    assert list(series) == [1, 2, 3]
    assert series.index[0] == pd.Period("2024-08", freq="M")

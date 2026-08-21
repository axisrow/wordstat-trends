"""Тесты для scripts/experiment_vector_d.py (MVP #23, вариант "Г")."""

from pathlib import Path

import numpy as np
import pytest

from scripts.experiment_vector_d import (
    FIXTURES,
    check_statsforecast_model_choice,
    compare_seasonal_cycles_within_full_series,
    fit_autoets_vector,
    load_dynamics_csv,
    truncate_series,
)


@pytest.mark.parametrize("path", FIXTURES.values(), ids=FIXTURES.keys())
def test_load_dynamics_csv_parses_all_fixtures(path: Path):
    series = load_dynamics_csv(path)

    assert len(series) == 24
    # Ряд начинается с августа 2024 и идёт помесячно без пропусков (docs/DATA.md).
    assert str(series.index[0]) == "2024-08"
    assert str(series.index[-1]) == "2026-07"
    assert series.index.is_monotonic_increasing
    assert (series > 0).all()


def test_load_dynamics_csv_seasonal_peak_in_december():
    # «Новогодние подарки»: пик спроса — декабрь 2024, известен из docs/DATA.md.
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    assert series.loc["2024-12"] == 1234632
    assert series.loc["2024-12"] == series.max()


def test_truncate_series_drops_from_the_end():
    series = load_dynamics_csv(FIXTURES["high_freq (купить телефон)"])
    truncated = truncate_series(series, 3)
    assert len(truncated) == 21
    assert truncated.index[-1] == series.index[-4]


def test_fit_autoets_vector_on_full_series_returns_calendar_aligned_seasonal():
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    vector = fit_autoets_vector(series, sp=12)

    assert len(vector.seasonal) == 12
    # Декабрьский сезонный коэффициент (индекс 11) — максимум профиля.
    assert int(np.argmax(vector.seasonal)) == 11


def test_fit_autoets_vector_fails_on_series_shorter_than_two_seasonal_cycles():
    """Главный результат MVP: AutoETS(sp=12) требует >= 2 полных цикла (24
    точки). Ряд из 21 точки (24 - 3) физически не фитится — это не дрейф
    коэффициентов, а отказ на этапе инициализации сезонности.
    """
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    truncated = truncate_series(series, 3)

    with pytest.raises(ValueError, match="two full seasonal cycles"):
        fit_autoets_vector(truncated, sp=12)


def test_within_series_cycle_comparison_returns_high_correlation_on_full_series():
    """Побочное измерение (раздел 2b отчёта): цикл 1 и цикл 2 внутри одного
    фита на полном ряде должны быть сильно скоррелированы — это не проверка
    внешней устойчивости (см. caveat в результате), а факт про один фит.
    """
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    result = compare_seasonal_cycles_within_full_series(series, sp=12)

    assert "error" not in result
    assert len(result["cycle_1"]) == 12
    assert len(result["cycle_2"]) == 12
    assert result["corr"] > 0.9
    assert "caveat" in result


def test_within_series_cycle_comparison_reports_error_on_short_series():
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    truncated = truncate_series(series, 3)  # 21 точка, < 2*sp

    result = compare_seasonal_cycles_within_full_series(truncated, sp=12)
    assert "error" in result


def test_statsforecast_cross_check_reports_model_structure():
    """Раздел 3 отчёта: statsforecast должен фититься без ошибок (в отличие
    от основного бэкенда) и вернуть структуру выбранной модели.
    """
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    result = check_statsforecast_model_choice(series, sp=12)

    assert result["method"].startswith("ETS(")
    assert len(result["components"]) == 4
    assert isinstance(result["has_seasonal"], bool)

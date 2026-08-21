"""Тесты для scripts/experiment_vector_d.py (MVP #23, вариант "Г")."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_vector_d import (
    DAILY_FIXTURES,
    FIXTURES,
    _parse_daily_period,
    check_statsforecast_model_choice,
    check_weekday_balance,
    compare_seasonal_cycles_within_full_series,
    fit_autoets_vector,
    load_dynamics_csv,
    main_daily,
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


# --- Вторая итерация (дневные данные, sp=7): парсер и проверка баланса дней недели.
# Дневных фикстур ещё нет в репозитории (собираются отдельно) — эти тесты покрывают
# только код на контролируемых мини-примерах, не подменяют реальный эксперимент.


def test_parse_daily_period_reads_dotted_date():
    # Формат дневной выгрузки — "22.06.2026" (день.месяц.год с точками),
    # НЕ русское название месяца, как в месячных фикстурах.
    period = _parse_daily_period("22.06.2026")
    assert period == pd.Period("2026-06-22", freq="D")


def test_parse_daily_period_rejects_ru_month_format():
    # Месячный парсер и дневной несовместимы — проверяем, что дневной именно
    # падает на "август 2024", а не молча даёт неверный результат.
    with pytest.raises(ValueError):
        _parse_daily_period("август 2024")


def test_check_weekday_balance_detects_balanced_window():
    # 56-дневное окно, кратное семи, начинающееся с понедельника — каждый
    # день недели встречается ровно 8 раз.
    index = pd.period_range(start="2026-06-22", periods=56, freq="D")  # понедельник
    series = pd.Series(range(56), index=index)

    result = check_weekday_balance(series)

    assert result["balanced"] is True
    assert set(result["counts_by_weekday"].values()) == {8}
    assert sum(result["counts_by_weekday"].values()) == 56


def test_check_weekday_balance_detects_unbalanced_window():
    # Окно длиной не кратной семи (49 + 3 дня) — баланс должен быть нарушен.
    index = pd.period_range(start="2026-06-22", periods=52, freq="D")
    series = pd.Series(range(52), index=index)

    result = check_weekday_balance(series)

    assert result["balanced"] is False


def test_main_daily_fails_clearly_when_fixtures_are_missing():
    """Дневные фикстуры ещё не собраны (см. docs/EXPERIMENT_VECTOR_D.md) —
    main_daily() должна падать с понятной ошибкой, а не молча пропускать
    отсутствующие данные или подставлять что-то вместо них.
    """
    assert not any(path.exists() for path in DAILY_FIXTURES.values()), (
        "Ожидалось, что дневные фикстуры ещё не собраны — если они уже появились, "
        "этот тест и main_daily() пора запускать на реальных данных, а не проверять отказ"
    )

    with pytest.raises(FileNotFoundError, match="Дневные фикстуры ещё не собраны"):
        main_daily()

"""Вектор параметров: свойства, на которые опирается отчёт варианта В.

Проверяется не «числа те же самые», а утверждения, которые отчёт делает:
профиль нормирован, все месяцы покрыты на всех трёх длинах ряда, AutoETS
падает на укороченном ряде (а не подменяет модель молча).
"""

from pathlib import Path

import numpy as np
import pytest

from wordstat_trends.demand_vector import SP, autoets_vector, month_counts, seasonal_index
from wordstat_trends.dynamics_io import load_dynamics

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def seasonal_series():
    return load_dynamics(FIXTURES / "dynamics_seasonal.csv")


@pytest.fixture(scope="module")
def flat_series():
    return load_dynamics(FIXTURES / "dynamics_high_freq.csv")


def test_seasonal_index_is_normalised(seasonal_series):
    profile = seasonal_index(seasonal_series)
    assert profile.shape == (SP,)
    assert profile.mean() == pytest.approx(1.0)


def test_seasonal_index_finds_december_peak(seasonal_series):
    profile = seasonal_index(seasonal_series)
    # Индекс 11 — декабрь: профиль в календарном порядке, январь нулевой.
    assert int(np.argmax(profile)) == 11
    assert profile[11] > 5


def test_flat_phrase_has_small_amplitude(flat_series):
    """«Купить телефон» не сезонная — профиль обязан это показывать."""
    assert seasonal_index(flat_series).std() < 0.2


@pytest.mark.parametrize("cut", [0, 3, 6])
def test_all_months_covered_at_every_truncation(seasonal_series, cut):
    """Ключевая предпосылка проверки устойчивости: дыр в покрытии нет.

    Именно этим оцениватель по средним месяца отличается от индекса на
    скользящем среднем, у которого на 18 точках выпадают ноябрь и декабрь.
    """
    window = seasonal_series.iloc[:-cut] if cut else seasonal_series
    counts = month_counts(window)
    assert counts.shape == (SP,)
    assert counts.min() >= 1
    assert counts.sum() == len(window)


def test_seasonal_shape_survives_truncation(seasonal_series):
    """Форма сезонной фразы воспроизводится на укороченном ряде."""
    reference = seasonal_index(seasonal_series)
    for cut in (3, 6):
        profile = seasonal_index(seasonal_series.iloc[:-cut])
        assert np.corrcoef(reference, profile)[0, 1] > 0.99
        assert int(np.argmax(profile)) == 11


def test_seasonal_index_needs_a_full_year(seasonal_series):
    with pytest.raises(ValueError, match="минимум"):
        seasonal_index(seasonal_series.iloc[:8])


def test_autoets_vector_on_full_series(seasonal_series):
    vector = autoets_vector(seasonal_series)
    assert vector.seasonal.shape == (SP,)
    # Профиль нормируется явно — ETS свои состояния не нормирует.
    assert vector.seasonal.mean() == pytest.approx(1.0)
    assert int(np.argmax(vector.seasonal)) == 11
    assert vector.spec.startswith("ETS(")
    assert vector.as_array().shape == (SP + 2,)


@pytest.mark.parametrize("cut", [3, 6])
def test_autoets_refuses_short_series_instead_of_dropping_seasonality(seasonal_series, cut):
    """Центральный измеренный факт отчёта.

    На 21 и 18 точках AutoETS(sp=12) обязан падать. Если однажды он начнёт
    возвращать результат, вывод отчёта устареет — и этот тест должен об этом
    сообщить, а не промолчать.
    """
    with pytest.raises(ValueError, match="two full seasonal cycles"):
        autoets_vector(seasonal_series.iloc[:-cut])

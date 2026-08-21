"""Вектор `sp=7`: свойства, на которые опирается вторая итерация отчёта."""

from pathlib import Path

import numpy as np
import pytest

from wordstat_trends.daily_window import SP, make_window
from wordstat_trends.dynamics_io import load_dynamics
from wordstat_trends.weekly_vector import BACKENDS, sktime_vector, statsforecast_vector

FIXTURES = Path(__file__).parent / "fixtures"


def window(name):
    return make_window(load_dynamics(FIXTURES / f"dynamics_daily_{name}_mvp.csv", granularity="daily"))


def empirical_profile(series):
    """Средние по дням недели — опора, не зависящая ни от какой модели."""
    values = series.astype(float)
    means = values.groupby(values.index.dayofweek).mean()
    return (means / means.mean()).to_numpy()


@pytest.fixture(scope="module")
def seasonal_train():
    return window("seasonal").train


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_profile_matches_empirical_weekday_means(backend, seasonal_train):
    """Профиль обязан совпасть с эмпирикой.

    Это тест на выравнивание: бэкенды хранят сезонные состояния в разном
    порядке, и повёрнутый профиль — самая правдоподобная и самая опасная
    ошибка здесь. Она даёт «расхождение бэкендов», выглядящее как результат.
    """
    vector = BACKENDS[backend](seasonal_train)
    assert np.corrcoef(vector.seasonal, empirical_profile(seasonal_train))[0, 1] > 0.95


def test_backends_agree_on_seasonal_phrase(seasonal_train):
    sk = sktime_vector(seasonal_train)
    sf = statsforecast_vector(seasonal_train)
    assert np.corrcoef(sk.seasonal, sf.seasonal)[0, 1] > 0.99


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_profile_is_normalised(backend, seasonal_train):
    vector = BACKENDS[backend](seasonal_train)
    assert vector.seasonal.shape == (SP,)
    assert vector.seasonal.mean() == pytest.approx(1.0, rel=1e-6)


def test_weekend_dip_is_found(seasonal_train):
    """У поисковой фразы будни выше выходных — суббота должна быть провалом."""
    seasonal = sktime_vector(seasonal_train).seasonal
    assert seasonal[:5].mean() > seasonal[5:].mean()
    assert int(np.argmin(seasonal)) in (5, 6)


def test_additive_seasonality_is_not_normalised_by_mean():
    """Аддитивный профиль нельзя делить на среднее — оно около нуля.

    Наивная нормировка давала амплитуду ~5588 у заведомо плоской фразы.
    """
    vector = statsforecast_vector(window("high_freq").train)
    assert vector.spec.endswith(",A)")
    assert vector.amplitude < 0.5


def test_refit_is_executable_on_shortened_windows(seasonal_train):
    """Ради этого делалась вторая итерация: на sp=12 рефит падал вовсе."""
    for cut in (7, 14):
        vector = sktime_vector(seasonal_train.iloc[:-cut])
        assert vector.seasonal.shape == (SP,)
        assert len(seasonal_train.iloc[:-cut]) // SP >= 6


def test_seasonal_shape_survives_truncation(seasonal_train):
    reference = sktime_vector(seasonal_train).seasonal
    for cut in (7, 14):
        profile = sktime_vector(seasonal_train.iloc[:-cut]).seasonal
        assert np.corrcoef(reference, profile)[0, 1] > 0.9


def test_aic_is_not_comparable_across_backends(seasonal_train):
    """AIC двух бэкендов несопоставимы: разные определения правдоподобия.

    statsforecast считает концентрированное правдоподобие `−n/2·log(SSE)`,
    опуская аддитивные константы, которые statsmodels включает. Разрыв
    остаётся даже при одинаковой спецификации, поэтому «разница AIC» между
    бэкендами не означает, что одна модель лучше другой.
    """
    import warnings

    from sktime.forecasting.ets import AutoETS
    from statsforecast.models import AutoETS as StatsForecastAutoETS

    values = seasonal_train.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = AutoETS(auto=False, error="add", trend=None, seasonal=None, sp=SP, n_jobs=1)
        sk.fit(values)
        sf = StatsForecastAutoETS(season_length=SP, model="ANN")
        sf.fit(values.to_numpy())

    sf_loglik = float(np.ravel(sf.model_["loglik"])[0])
    residuals = np.asarray(sf.model_["residuals"], dtype=float)
    concentrated = -len(values) / 2 * np.log(np.sum(residuals**2))

    # Правдоподобие statsforecast — ровно концентрированная форма.
    assert sf_loglik == pytest.approx(concentrated, abs=0.01)
    # И оно систематически ниже, чем у statsmodels, при той же спецификации.
    assert sf_loglik < float(sk._fitted_forecaster.llf) - 20

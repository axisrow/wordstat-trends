"""Guardrail issue #44: согласованность декомпозиции AIC с вердиктами AutoETS.

Ревью PR #36 нашло класс ошибки «ручная формула критерия рядом с библиотечной
молча расходится»: ручной подсчёт AIC/AICc брал k = len(res.params), тогда как
statsmodels профилирует sigma2 аналитически — df_model = len(params) + 1. На
ячейке «курсы/w=21/aicc» из-за недосчёта знак ΔAICc расходился с вердиктом
AutoETS. Два регрессионных теста:

1. знаки: на каждой ячейке сетки (окна × фразы × критерии, дневные фикстуры)
   знак ΔAIC/ΔAICc из декомпозиции совпадает с вердиктом AutoETS(auto=True);
2. тождество: -2*loglik + 2*df_model == res.aic (и формула AICc) в пределах
   допуска — напрямую ловит недосчёт степеней свободы (тот самый +1).
"""

import math

import pytest

from scripts.experiment_vector_d import load_daily_dynamics_csv, truncate_series
from scripts.issue32_seasonal_structure import (
    FIXTURES,
    TRAIN_WINDOW,
    WINDOWS,
    run_real_grid,
    statsmodels_grid,
)


@pytest.fixture(scope="module")
def real_grid():
    # Полная сетка A/B (вердикты AutoETS + декомпозиция) считается один раз
    # на модуль — фит auto на 42 ячейках × 2 критерия недёшев.
    return run_real_grid()


@pytest.mark.parametrize("name", sorted(FIXTURES.keys()))
@pytest.mark.parametrize("w", WINDOWS)
def test_sign_of_delta_criteria_matches_autoets_verdict(real_grid, name, w):
    """Знак ΔAIC/ΔAICc (лучше сезонная ⇒ отрицательный) обязан совпадать с
    has_seasonal вердиктом AutoETS(auto=True) на той же ячейке.
    """
    entry = real_grid[name][w]
    for ic in ("aic", "aicc"):
        verdict = entry[ic]["has_seasonal"]
        delta = entry["decomp"][f"d_{ic}_seasonal_minus_non"]
        assert (delta < 0) == verdict, (
            f"{name}/w={w}/{ic}: вердикт AutoETS has_seasonal={verdict}, "
            f"но знак Δ={delta:.3f} говорит об обратном"
        )


@pytest.mark.parametrize("name", sorted(FIXTURES.keys()))
@pytest.mark.parametrize("w", WINDOWS)
def test_criterion_identity_holds_with_df_model(name, w):
    """Тождество критерия на каждом фите сетки: df_model = len(params) + 1
    (sigma2 профилируется и не входит в params), и -2*loglik + 2*df_model
    совпадает с res.aic, а формула AICc — с res.aicc. Подсчёт k = len(params)
    (недосчёт +1) делает любое из равенств ложным.
    """
    path = FIXTURES[name]
    full = load_daily_dynamics_csv(path).iloc[:TRAIN_WINDOW]
    y = truncate_series(full, TRAIN_WINDOW - w).astype(float).to_numpy()

    grid = statsmodels_grid(y)
    assert grid, f"{name}/w={w}: пустая сетка"
    checked = 0
    for fit in grid:
        if "error" in fit:  # недопустимый фит фиксируется, но не проверяется
            continue
        label = f"{name}/w={w}/spec={fit['spec']}"
        # Недосчёт степеней свободы: k обязан включать профилированную sigma2.
        assert fit["k"] == fit["n_params"] + 1, (
            f"{label}: k={fit['k']} != len(params)={fit['n_params']} + 1 "
            "(sigma2 не учтена как степень свободы)"
        )
        manual_aic = -2.0 * fit["loglik"] + 2.0 * fit["k"]
        assert math.isclose(manual_aic, fit["aic"], rel_tol=1e-9, abs_tol=1e-6), (
            f"{label}: -2*loglik + 2*k = {manual_aic} != res.aic = {fit['aic']}"
        )
        denom = fit["n"] - fit["k"] - 1
        if denom <= 0:
            # На окне 14 у сезонных спек n - k - 1 = 0: AICc вырожден (statsmodels
            # возвращает inf), проверяем только AIC-тождество.
            assert math.isinf(fit["aicc"]), (
                f"{label}: n - k - 1 = {denom}, ожидался inf в res.aicc, "
                f"получено {fit['aicc']}"
            )
            continue
        manual_aicc = manual_aic + 2.0 * fit["k"] * (fit["k"] + 1) / denom
        assert math.isclose(manual_aicc, fit["aicc"], rel_tol=1e-9, abs_tol=1e-6), (
            f"{label}: формула AICc = {manual_aicc} != res.aicc = {fit['aicc']}"
        )
        checked += 1
    assert checked >= 2, f"{name}/w={w}: сошлись только {checked} фитов сетки"

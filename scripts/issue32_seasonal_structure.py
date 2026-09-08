"""Issue #32: механизм смены структуры модели (has_seasonal) между окнами.

Исследование по issue #32 (разбор наблюдения MVP #23 / варианта Г): почему
AutoETS(sp=7, auto=True) меняет саму структуру модели (есть/нет сезонная
компонента) между окнами 56 и 42 дня.

Три части:
  A. Реальные дневные фикстуры MVP #23: AutoETS(sp=7, auto) на окнах
     56/49/42/35/28/21/14 (общее начало, укорочение с конца), критерии
     aic и aicc.
  B. Механизм: прямой перебор допустимых спецификаций statsmodels.ETSModel
     на каждом окне; для лучшей сезонной и лучшей несезонной спеки —
     loglik, число параметров и AIC/AICc/BIC. Разделяет «данные не
     поддерживают сезонность» от «критерий не окупает параметры».
  C. Симуляционная калибровка: 100 реплик синтетики с уровнем/профилем/шумом
     каждой реальной фразы (профиль сохранён / константа / чистый шум);
     частота has_seasonal на 56 днях и частота смены структуры 56->42.

Результаты и выводы: docs/ISSUE_32_SEASONAL_STRUCTURE.md.
Запуск: python scripts/issue32_seasonal_structure.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from experiment_vector_d import load_daily_dynamics_csv, truncate_series  # noqa: E402
from sktime.forecasting.ets import AutoETS  # noqa: E402
from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # noqa: E402

FIXTURES = {
    "seasonal": REPO / "tests/fixtures/dynamics_daily_seasonal_mvp.csv",
    "mid_freq": REPO / "tests/fixtures/dynamics_daily_mid_freq_mvp.csv",
    "high_freq": REPO / "tests/fixtures/dynamics_daily_high_freq_mvp.csv",
}
WINDOWS = [56, 49, 42, 35, 28, 21, 14]
TRAIN_WINDOW = 56
SP = 7
N_REPS = 100
SEED = 32


def sktime_has_seasonal(y: pd.Series, ic: str) -> tuple[bool, str]:
    """Фит AutoETS(sp=7, auto) и выбранная спецификация (error/trend/seasonal)."""
    f = AutoETS(auto=True, sp=SP, n_jobs=1, information_criterion=ic)
    f.fit(y)
    # Приватный атрибут sktime — statsmodels ETSResults с выбранной спекой.
    model = f._fitted_forecaster.model  # type: ignore[attr-defined]
    spec = (
        f"{model.error[0].upper()}"
        f"{'A' if model.trend else 'N'}"
        f"{'A' if model.seasonal == 'add' else ('M' if model.seasonal == 'mul' else 'N')}"
    )
    return model.seasonal is not None, spec


def statsmodels_grid(y: np.ndarray) -> list[dict]:
    """Перебор допустимых спецификаций ETS на ряде y (restrict-правила как в auto).

    Сетка повторяет sktime AutoETS(auto=True) с дефолтами (allow_multiplicative_
    trend=False, restrict=True): error add/mul, trend add/None, seasonal
    add/mul/None, damped True/False (не бывает при trend=None); при
    аддитивной ошибке мультипликативные компоненты запрещены. AIC/AICc/BIC —
    собственные значения statsmodels (k = df_model = len(params) + 1,
    включая sigma2).
    """
    out = []
    positive = bool((y > 0).all())
    errors = ["add", "mul"] if positive else ["add"]
    seasonals = [None, "add", "mul"] if positive else [None, "add"]
    for error in errors:
        for trend in [None, "add"]:
            for seasonal in seasonals:
                if error == "add" and seasonal == "mul":
                    continue
                for damped in [False, True]:
                    if trend is None and damped:
                        continue
                    spec = (
                        f"{error[0].upper()}"
                        f"{'A' if trend else 'N'}"
                        f"{'d' if damped else ''}"
                        f"{'A' if seasonal == 'add' else ('M' if seasonal == 'mul' else 'N')}"
                    )
                    try:
                        m = ETSModel(
                            y,
                            error=error,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=seasonal,
                            seasonal_periods=SP if seasonal else None,
                            initialization_method="estimated",
                        )
                        res = m.fit(disp=False)
                        out.append(
                            {
                                "spec": spec,
                                "seasonal": seasonal is not None,
                                "loglik": res.llf,
                                "k": res.df_model,
                                "n_params": len(res.params),
                                "n": len(y),
                                "aic": res.aic,
                                "aicc": res.aicc,
                                "bic": res.bic,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 — фиксируем и недопустимые фиты
                        out.append({"spec": spec, "error": str(exc)[:60]})
    return out


def best_of(grid: list[dict], seasonal: bool, key: str) -> dict:
    ok = [g for g in grid if key in g and g["seasonal"] is seasonal]
    if not ok:
        label = "сезонной" if seasonal else "несезонной"
        raise ValueError(f"нет сошедшихся фитов в группе ({label}, {key})")
    return min(ok, key=lambda g: g[key])


# ---------- A + B: реальные фикстуры (запуск — в main) ----------
def run_real_grid() -> dict:
    """Сетка окон × фразы × критерии: вердикты AutoETS + декомпозиция критерия.

    Отдельная функция (а не module-level код), чтобы guardrail-тест согласованности
    (issue #44) проверял именно этот код, а не собственную копию.
    """
    real: dict = {}
    for name, path in FIXTURES.items():
        full = load_daily_dynamics_csv(path).iloc[:TRAIN_WINDOW]
        real[name] = {}
        for w in WINDOWS:
            series = truncate_series(full, TRAIN_WINDOW - w)
            y = series.astype(float).to_numpy()
            entry: dict = {"window": w, "n": len(y)}
            for ic in ["aicc", "aic"]:
                has_seas, spec = sktime_has_seasonal(series.astype(float), ic)
                entry[ic] = {"has_seasonal": has_seas, "spec": spec}
            grid = statsmodels_grid(y)
            best_s = best_of(grid, True, "aicc")
            best_n = best_of(grid, False, "aicc")
            best_s_aic = best_of(grid, True, "aic")
            best_n_aic = best_of(grid, False, "aic")
            entry["decomp"] = {
                "best_seasonal": best_s,
                "best_nonseasonal": best_n,
                "d_aicc_seasonal_minus_non": best_s["aicc"] - best_n["aicc"],
                "d_aic_seasonal_minus_non": best_s_aic["aic"] - best_n_aic["aic"],
                "d_loglik_non_minus_seasonal": best_n["loglik"] - best_s["loglik"],
                "d_k": best_s["k"] - best_n["k"],
                "penalty_s_aicc_minus_aic": best_s["aicc"] - best_s["aic"],
                "penalty_n_aicc_minus_aic": best_n["aicc"] - best_n["aic"],
            }
            real[name][w] = entry
            d = entry["decomp"]
            print(
                name,
                w,
                entry["aicc"],
                entry["aic"],
                "dAICc(S-N)={:.1f}".format(d["d_aicc_seasonal_minus_non"]),
                "dLL={:.2f}".format(d["d_loglik_non_minus_seasonal"]),
                "dk={:d}".format(d["d_k"]),
            )
    return real


def synth_from(real_y: np.ndarray, keep_profile: bool, n: int, gen) -> np.ndarray:
    """Синтетика с уровнем/шумом реальной фразы; профиль — реальный или константа."""
    y = real_y.astype(float)
    dow = np.arange(len(y)) % SP
    prof = pd.Series(y).groupby(dow).transform("mean").to_numpy()
    noise_sd = (y - prof).std(ddof=SP)
    level = y.mean()
    target = prof / prof.mean() if keep_profile else np.ones(SP)
    idx = np.arange(n) % SP
    return level * target[idx] + gen.normal(0, noise_sd, n)


def synth_null(real_y: np.ndarray, n: int, gen) -> np.ndarray:
    y = real_y.astype(float)
    return gen.normal(y.mean(), y.std(), n)


def main() -> None:
    print("== A/B. Реальные фикстуры: сетка окон ==")
    real = run_real_grid()

    # ---------- C: симуляция ----------
    print("== C. Симуляция ==")
    rng = np.random.default_rng(SEED)
    sim: dict = {}
    for name, path in FIXTURES.items():
        y56 = load_daily_dynamics_csv(path).iloc[:TRAIN_WINDOW].astype(float).to_numpy()
        sim[name] = {}
        conditions = {
            "seasonal": lambda n, g: synth_from(y56, True, n, g),
            "no_seasonal_profile": lambda n, g: synth_from(y56, False, n, g),
            "null_noise": lambda n, g: synth_null(y56, n, g),
        }
        for cond, gen_fn in conditions.items():
            flips = {"aicc": 0, "aic": 0}
            seas56 = {"aicc": 0, "aic": 0}
            seas42 = {"aicc": 0, "aic": 0}
            for _ in range(N_REPS):
                # 42-дневное окно — усечение того же 56-дневного ряда
                # (конвенция MVP #23), а не независимый розыгрыш.
                s56 = pd.Series(gen_fn(56, rng))
                s42 = s56.iloc[:42]
                for ic in ["aicc", "aic"]:
                    hs = {}
                    for w, s in [(56, s56), (42, s42)]:
                        hs[w], _ = sktime_has_seasonal(s.astype(float), ic)
                    if hs[56]:
                        seas56[ic] += 1
                    if hs[42]:
                        seas42[ic] += 1
                    if hs[56] != hs[42]:
                        flips[ic] += 1
            sim[name][cond] = {
                "n_reps": N_REPS,
                "has_seasonal_at_56": seas56,
                "has_seasonal_at_42": seas42,
                "structure_flip_56_to_42": flips,
            }
            print(name, cond, sim[name][cond])

    results = {"real": real, "simulation": sim}
    out = Path("/tmp/issue32_seasonal_structure_results.json")
    out.write_text(json.dumps(results, indent=1, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()

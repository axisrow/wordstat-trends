"""Issue #34: порог длины окна для детекции недельной сезонности (sp=7).

Сетка окон 14/21/28/35/42/49/56 дней (2..8 полных недель), все окна
заканчиваются ОДНОЙ и той же датой и вложены как суффиксы (окно 14 —
последние 14 дней, окно 56 — весь базовый отрезок). Это отличается от MVP #23
(experiment_vector_d.py), где окна имели общее НАЧАЛО: там вопрос был
«как меняется вектор при укорочении с конца», здесь — «сколько истории
нужно, чтобы увидеть сезонность».

Объём: сбор 7 новых фраз заблокирован на axisrow/wordstat-cli#29 (дневной
сбор падает DownloadTimeoutError), поэтому прогон идёт по 3 существующим
дневным фикстурам MVP #23 — это НЕ заявленный в issue объём (10 фраз),
мощность ограничена, что отражено в отчёте. Данные не синтезируются и не
достраиваются: синтетика — только отдельная калибровка.

Четыре метода детекции (вердикт в каждой ячейке — «увидел ли sp=7», а не
«какой sp вернулся»: детектор, вернувший sp=4, сезонность с периодом 7 НЕ
обнаружил):
1. AutoETS(sp=7, auto=True, information_criterion="aic") — сезонность
   косвенно, как побочный результат выбора спецификации по AIC. "aic", не
   дефолтный "aicc" — чтобы совпадать с fit_autoets_vector MVP #23; aicc
   записывается рядом в JSON как кросс-проверка (в issue #32 показано, что
   выбор критерия сам меняет вердикт).
2. SeasonalityACF(candidate_sp=[7], p_threshold=0.05) — ACF-детектор.
3. SeasonalityACFqstat(candidate_sp=[7], p_threshold=0.05,
   p_adjust="fdr_by") — Q-статистика Льюнга—Бокса.
4. SeasonalityPeriodogram() — периодограмма (дефолты min_period=4,
   max_period=None, thresh=0.10). Вердикт sp==7, любое другое значение
   (включая 1 и соседние 6/8) — «нет».

Почему candidate_sp=[7], а не широкий список: вопрос issue — вердикт именно
по sp=7, и калибровочная таблица issue построена так же (детекторы в ней
возвращают только 1 или 7). Широкий список кандидатов ([2..12, 14]) на
коротких рядах обрезается nlags (кандидаты длиннее nlags вызывают IndexError
внутри sktime), а на длинных — размывает вердикт: ACF начинает возвращать
прочие значимые лаги. Список кандидатов [2..12, 14] прогнан отдельно как
чувствительность (raw_sp_wide в JSON), в основную таблицу не входит.

Детекторы применяются к сырому ряду уровней (без дифференцирования) — так
построена калибровочная таблица issue (проверено: на diff-ряде
периодограммная колонка таблицы не воспроизводится).

Fail-closed (docs/MODELING.md): исключение любого метода на любом окне —
вердикт «нет» с пометкой error и кодом исключения, никогда не молчаливый
пропуск; NaN среди значений ряда — тоже «нет».

Запуск: python scripts/issue34_window_threshold.py
Результат: JSON в stdout; таблица и разбор — docs/ISSUE_34_WINDOW_THRESHOLD.md.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from wordstat_trends.run_meta import run_metadata

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from experiment_vector_d import check_weekday_balance, load_daily_dynamics_csv  # noqa: E402

SP = 7
# Базовое train-окно 56 дней = первые 56 строк фикстур (23.06–17.08.2026),
# как TRAIN_WINDOW_DAYS в experiment_vector_d.py; 2 последних строки (18–19.08,
# holdout MVP) НЕ входят ни в одно окно — общая конечная дата окон 17.08.
BASE_WINDOW_DAYS = 56
WINDOWS = [14, 21, 28, 35, 42, 49, 56]

FIXTURES = {
    "seasonal (новогодние подарки)": REPO / "tests/fixtures/dynamics_daily_seasonal_mvp.csv",
    "high_freq (купить телефон)": REPO / "tests/fixtures/dynamics_daily_high_freq_mvp.csv",
    "mid_freq (курсы английского)": REPO / "tests/fixtures/dynamics_daily_mid_freq_mvp.csv",
}

CANDIDATE_SP = [7]  # обоснование — модульный docstring
CANDIDATE_SP_WIDE = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
P_THRESHOLD = 0.05


def suffix_window(base: pd.Series, days: int) -> pd.Series:
    """Окно длиной `days`, заканчивающееся той же датой, что base (суффикс)."""
    if days > len(base) or days % SP != 0:
        raise ValueError(f"Окно {days} некорректно: длина базы {len(base)}, кратность {SP} нарушена")
    return base.iloc[len(base) - days :]


def verdict_from_sp(raw_sp: float | int | None) -> tuple[str, object]:
    """Вердикт по sp=7 из сырого значения детектора. Вердикт — «увидел ли
    период 7», не «какой sp вернулся»: sp=4 при периоде 7 — «нет».

    Fail-closed (docs/MODELING.md): None/NaN трактуется как «нет», не как
    пропуск — молчаливый NaN в таблице вердиктов читался бы как «сезонности
    нет» без проверки.
    """
    if raw_sp is None:
        return "нет", raw_sp
    sp = float(raw_sp)
    if math.isnan(sp):
        return "нет", raw_sp
    return ("да" if sp == SP else "нет"), raw_sp


def detect_autoets(series: pd.Series, ic: str = "aic") -> dict:
    """AutoETS(sp=7, auto): вердикт — содержит ли выбранная по IC спецификация
    сезонную компоненту (косвенная детекция, как в MVP #23)."""
    from sktime.forecasting.ets import AutoETS

    y = series.reset_index(drop=True).astype(float)
    y.index = pd.RangeIndex(len(y))
    forecaster = AutoETS(auto=True, sp=SP, n_jobs=1, information_criterion=ic)
    forecaster.fit(y)
    fitted = forecaster._fitted_forecaster  # type: ignore[attr-defined]  # statsmodels ETSResults
    model = fitted.model  # type: ignore[attr-defined]  # statsmodels ETSResults
    # та же проверка структуры, что в fit_autoets_vector MVP #23
    has_seasonal = "seasonal" in fitted.states.columns and len(fitted.states) >= SP  # type: ignore[attr-defined]
    spec = f"{model.error}/{model.trend}/{model.seasonal}"
    return {"verdict": "да" if has_seasonal else "нет", "spec": spec}


def detect_sktime_detector(series: pd.Series, detector_name: str) -> dict:
    """Специализированный детектор sktime.param_est.seasonality. Возвращает
    вердикт по sp=7 и сырой возвращённый sp (для разбора расхождений)."""
    from sktime.param_est.seasonality import SeasonalityACF, SeasonalityACFqstat, SeasonalityPeriodogram

    y = series.reset_index(drop=True).astype(float)
    detectors = {
        "acf": lambda: SeasonalityACF(candidate_sp=CANDIDATE_SP, p_threshold=P_THRESHOLD),
        "qstat": lambda: SeasonalityACFqstat(
            candidate_sp=CANDIDATE_SP, p_threshold=P_THRESHOLD, p_adjust="fdr_by"
        ),
        "periodogram": lambda: SeasonalityPeriodogram(),
    }
    det = detectors[detector_name]()
    det.fit(y)
    raw_sp = det.get_fitted_params()["sp"]
    verdict, _ = verdict_from_sp(raw_sp)
    # numpy-целое -> python int, иначе JSON-сериализация отчёта падает
    raw = None if raw_sp is None else int(raw_sp)
    significant = det.get_fitted_params().get("sp_significant", [])
    return {
        "verdict": verdict,
        "raw_sp": raw,
        "sp_significant": [int(s) for s in np.atleast_1d(significant)],
    }


def run_cell(series: pd.Series) -> dict:
    """Все 4 метода на одном окне. Каждое исключение — fail-closed вердикт
    «нет» с записью ошибки (см. detect_* и модульный docstring)."""
    cell: dict = {}
    for name, fn in [
        ("autoets_aic", lambda: detect_autoets(series, "aic")),
        ("autoets_aicc", lambda: detect_autoets(series, "aicc")),
        ("acf", lambda: detect_sktime_detector(series, "acf")),
        ("qstat", lambda: detect_sktime_detector(series, "qstat")),
        ("periodogram", lambda: detect_sktime_detector(series, "periodogram")),
    ]:
        try:
            cell[name] = fn()
        except Exception as exc:  # noqa: BLE001 — отказ метода репортим как fail-closed «нет»
            cell[name] = {"verdict": "нет", "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    return cell


def sensitivity_wide_candidates(series: pd.Series) -> dict:
    """Чувствительность к списку кандидатов: тот же прогон ACF/Qstat с
    CANDIDATE_SP_WIDE (кандидаты длиннее nlags окна выбрасываются — sktime
    индексирует confint по nlags и падает на лагах вне диапазона)."""
    from sktime.param_est.seasonality import SeasonalityACF, SeasonalityACFqstat

    n = len(series)
    nlags = min(int(10 * np.log10(n)), n - 1)
    cand = [c for c in CANDIDATE_SP_WIDE if c <= nlags]
    y = series.reset_index(drop=True).astype(float)
    out: dict = {"candidates_used": cand}
    for name, det in [
        ("acf", SeasonalityACF(candidate_sp=cand, p_threshold=P_THRESHOLD)),
        ("qstat", SeasonalityACFqstat(candidate_sp=cand, p_threshold=P_THRESHOLD)),
    ]:
        try:
            det.fit(y)
            out[name] = int(det.get_fitted_params()["sp"])
        except Exception as exc:  # noqa: BLE001
            out[name] = f"error: {type(exc).__name__}"
    return out


def calibration_synth() -> dict:
    """Калибровка на заведомо сезонном ряде: синусоида периода 7 + шум.

    Рецепт воспроизведения калибровочной таблицы issue #34 (подобран
    перебором в /tmp- зондах, см. отчёт): legacy API np.random.seed(0),
    y = 10 + 4*sin(2*pi*t/7) + N(0, 2), t = 0..55, окна — префиксы.
    Точное совпадение ВСЕЙ таблицы issue не достигается ни одним
    опробованным рецептом; при этом рецепте колонка Periodogram совпадает
    побитово (4,5,7,8,7,8,7), порог ACF — на шаг позже issue (42 vs 35),
    порог Qstat — на шаг позже (21 vs 14). Расхождение зафиксировано в
    отчёте, калибровка используется как ориентировочная нижняя граница.
    """
    np.random.seed(0)
    t = np.arange(BASE_WINDOW_DAYS)
    y56 = 10.0 + 4.0 * np.sin(2 * np.pi * t / SP) + np.random.normal(0, 2, BASE_WINDOW_DAYS)
    rows = {}
    for n in WINDOWS:
        y = pd.Series(y56[:n])
        rows[n] = run_cell(y)
    return rows


def main() -> int:
    report: dict = {
        # seed=0: исторический seed калибровочной синусоиды (calibration_synth),
        # см. аудит в wordstat_trends/seeds.py
        "run": run_metadata(inputs=tuple(FIXTURES.values()), seed=0),
        "sp": SP,
        "windows": WINDOWS,
        "anchor": "все окна заканчиваются конечной датой базового 56-дневного отрезка (суффиксы)",
        "methods": {
            "autoets_aic": "AutoETS(sp=7, auto=True, ic=aic) — как fit_autoets_vector MVP #23",
            "acf": f"SeasonalityACF(candidate_sp={CANDIDATE_SP}, p_threshold={P_THRESHOLD})",
            "qstat": f"SeasonalityACFqstat(candidate_sp={CANDIDATE_SP}, p_threshold={P_THRESHOLD}, p_adjust=fdr_by)",
            "periodogram": "SeasonalityPeriodogram(min_period=4, max_period=None, thresh=0.10)",
        },
        "scope_note": "3 фикстуры вместо 10 фраз: сбор заблокирован на axisrow/wordstat-cli#29",
        "calibration_synthetic": calibration_synth(),
        "phrases": {},
    }

    for label, path in FIXTURES.items():
        full = load_daily_dynamics_csv(path)
        if len(full) < BASE_WINDOW_DAYS:
            raise ValueError(f"Фикстура '{label}' короче {BASE_WINDOW_DAYS} дней: {len(full)}")
        base = full.iloc[:BASE_WINDOW_DAYS]
        phrase: dict = {"base_end_date": str(base.index[-1]), "windows": {}}
        for w in WINDOWS:
            series = suffix_window(base, w)
            balance = check_weekday_balance(series)
            if not balance["balanced"]:
                raise ValueError(f"Окно {w} дней '{label}' не сбалансировано по дням недели: {balance}")
            phrase["windows"][w] = {
                "n": len(series),
                "start": str(series.index[0]),
                "end": str(series.index[-1]),
                "weekday_balance": balance["counts_by_weekday"],
                "cells": run_cell(series),
                "raw_sp_wide": sensitivity_wide_candidates(series),
            }
        report["phrases"][label] = phrase

    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

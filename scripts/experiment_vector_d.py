"""MVP-проверка гипотезы #23 (независимый вариант "Г"): устойчивость вектора
параметров AutoETS(sp=12), снятого с месячных фикстур Wordstat.

Формат фикстур — см. docs/DATA.md: UTF-8 с BOM, переводы строк только CR
(`\\r`, без `\\n`), разделитель полей `;`, десятичная запятая, период вида
"август 2024" (русское название месяца текстом + год).

Порядок проверки (обязателен, см. issue #23):
1. Разобрать три фикстуры в месячные ряды.
2. Снять вектор AutoETS(sp=12): уровень, тренд, 12 сезонных коэффициентов
   (sktime/statsmodels — основной бэкенд).
3. Устойчивость: пересчитать вектор на ряде, укороченном на 3 и на 6 точек
   (месяцев), сравнить с вектором на полном ряде.
3b. Кросс-проверка вторым независимым бэкендом (statsforecast — R-совместимый
    порт `forecast::ets`): какую спецификацию (есть/нет сезонность) выбирает
    он же на тех же рядах — не для сравнения коэффициентов, а чтобы увидеть,
    выбирают ли два независимых AIC-подбора вообще одну и ту же структуру
    модели.
4. Только если устойчивость подтвердилась — проверить, различает ли вектор
   сезонный профиль («новогодние подарки») от плоского («купить телефон»).

Запуск: python scripts/experiment_vector_d.py
(время выполнения не измеряется и не репортится — не характеристика метода)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ограничить BLAS/OMP-потоки ДО импорта numpy/statsmodels/statsforecast — на
# машине, где параллельно гоняются несколько независимых прогонов MVP #23,
# каждый процесс иначе пытается занять все ядра.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

FIXTURES = {
    "seasonal (новогодние подарки)": FIXTURES_DIR / "dynamics_seasonal.csv",
    "high_freq (купить телефон)": FIXTURES_DIR / "dynamics_high_freq.csv",
    "mid_freq (курсы английского)": FIXTURES_DIR / "dynamics_mid_freq.csv",
}

RU_MONTHS = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}


def _parse_ru_period(period: str) -> pd.Period:
    """"август 2024" -> pandas Period с частотой M."""
    month_name, year = period.strip().split()
    month = RU_MONTHS[month_name.lower()]
    return pd.Period(year=int(year), month=month, day=1, freq="M")


def load_dynamics_csv(path: Path) -> pd.Series:
    """Разобрать сырой CSV Вордстата (dynamics) в месячный pd.Series.

    Особенности формата (docs/DATA.md), обрабатываются явно:
    - UTF-8 с BOM -> encoding="utf-8-sig".
    - Переводы строк только CR -> splitlines() (не ручной split("\\n")).
    - Разделитель полей ";", 4-е поле (заголовок графика) пустое, игнорируется.
    - Число запросов: разделитель тысяч — обычный пробел, без десятичной части.
    - Период: "август 2024" — русский месяц текстом + год.
    """
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"Пустой файл: {path}")

    header = lines[0].split(";")
    assert header[0] == "Период", f"Неожиданный заголовок в {path}: {header}"

    records: list[tuple[pd.Period, float]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(";")
        period_raw, count_raw = fields[0], fields[1]
        period = _parse_ru_period(period_raw)
        count = float(count_raw.replace(" ", "").replace(" ", ""))
        records.append((period, count))

    periods, counts = zip(*records, strict=True)
    series = pd.Series(counts, index=pd.PeriodIndex(periods, freq="M"), name="count")
    series = series.sort_index()
    return series


@dataclass
class ParamVector:
    """Вектор параметров модели: уровень, тренд, 12 сезонных коэффициентов.

    has_trend=False означает, что AutoETS выбрал спецификацию БЕЗ трендовой
    компоненты (не что тренд измерен и равен нулю) — trend в этом случае 0.0
    по построению, а не по данным.
    """

    level: float
    trend: float
    has_trend: bool
    seasonal: np.ndarray  # длина 12, индекс 0 = январь
    model_spec: str  # "error/trend/seasonal[/damped]", как выбрал AutoETS

    def to_dict(self) -> dict:
        return {
            "level": round(float(self.level), 6),
            "trend": round(float(self.trend), 6) if self.has_trend else None,
            "model_spec": self.model_spec,
            "seasonal": [round(float(x), 6) for x in self.seasonal],
        }


def fit_autoets_vector(series: pd.Series, sp: int = 12) -> ParamVector:
    """Снять вектор параметров AutoETS(sp=sp) с месячного ряда.

    Использует sktime AutoETS. Возвращает уровень, тренд (slope на конец
    выборки) и полный сезонный профиль длиной sp, выровненный по календарным
    месяцам (индекс 0 = январь, ..., 11 = декабрь) — чтобы ряды с разным
    стартовым месяцем были сравнимы напрямую.
    """
    from sktime.forecasting.ets import AutoETS

    y = series.reset_index(drop=True).astype(float)
    y.index = pd.RangeIndex(len(y))

    forecaster = AutoETS(auto=True, sp=sp, n_jobs=1, information_criterion="aic")
    forecaster.fit(y)

    # Приватный атрибут sktime — единственный способ достать вектор параметров
    # ETS (level/trend/seasonal), а не только точечный прогноз.
    fitted = forecaster._fitted_forecaster  # type: ignore[attr-defined]  # statsmodels ETSResults
    states = fitted.states  # DataFrame с одной строкой на точку ряда: level, [trend], [seasonal]

    model = fitted.model  # ETSModel: хранит выбранную спецификацию error/trend/seasonal
    has_trend = "trend" in states.columns
    model_spec = f"{model.error}/{model.trend}/{model.seasonal}"
    if getattr(model, "damped_trend", False):
        model_spec += "/damped"

    last_state = states.iloc[-1]
    level = float(last_state["level"])
    trend = float(last_state["trend"]) if has_trend else 0.0

    if "seasonal" in states.columns and len(states) >= sp:
        # statsmodels хранит ОДНО сезонное значение на строку (не sp столбцов) —
        # последние sp строк дают ровно один полный сезонный цикл, где строка
        # states.iloc[-1] соответствует последней точке ряда, states.iloc[-2] —
        # предпоследней, и т.д. Раскладываем эти sp значений по календарным
        # месяцам, используя месяц последней точки как якорь.
        raw_seasonal = states["seasonal"].tail(sp).to_numpy(dtype=float)
    else:
        raw_seasonal = np.zeros(sp)

    last_period = series.index[-1]
    last_month = last_period.month  # 1..12
    calendar_seasonal = np.zeros(sp)
    # raw_seasonal[-1] — последняя точка (месяц last_month), raw_seasonal[-2] —
    # месяц перед ней, и т.д. в обратном порядке.
    for offset, value in enumerate(reversed(raw_seasonal.tolist())):
        month = ((last_month - 1 - offset) % sp) + 1  # 1..12
        calendar_seasonal[month - 1] = value

    return ParamVector(
        level=level, trend=trend, has_trend=has_trend, seasonal=calendar_seasonal, model_spec=model_spec
    )


def truncate_series(series: pd.Series, drop_last: int) -> pd.Series:
    """Укоротить ряд, отбросив drop_last последних точек."""
    if drop_last <= 0:
        return series
    return series.iloc[:-drop_last]


def compare_seasonal_cycles_within_full_series(series: pd.Series, sp: int = 12) -> dict:
    """Дополнительное (не требующее рефита) измерение устойчивости: сравнить
    сезонный коэффициент цикла 1 (первые sp строк states — 2024-08..2025-07)
    и цикла 2 (последние sp строк — 2025-08..2026-07) внутри ОДНОГО фита на
    полном 24-точечном ряде.

    Не заменяет проверку "пересчитать вектор на укороченном ряде" (пункт 2/3
    issue #23) — это она физически неисполнима на 24 точках (см. отчёт).
    Здесь измеряется другое: дрейфует ли сезонный профиль между двумя
    имеющимися циклами ОДНОГО фита. Оговорка: цикл 1 частично загрязнён
    эвристикой инициализации (statsmodels использует его же для старта), так
    что даже совпадение циклов 1 и 2 не отменяет вывод о непроверяемости
    внешней устойчивости (рефит на других данных).
    """
    from sktime.forecasting.ets import AutoETS

    if len(series) < 2 * sp:
        return {"error": f"нужно >= {2 * sp} точек, есть {len(series)}"}

    y = series.reset_index(drop=True).astype(float)
    y.index = pd.RangeIndex(len(y))
    forecaster = AutoETS(auto=True, sp=sp, n_jobs=1, information_criterion="aic")
    forecaster.fit(y)
    fitted = forecaster._fitted_forecaster  # type: ignore[attr-defined]

    if "seasonal" not in fitted.states.columns or len(fitted.states) < 2 * sp:
        return {"error": "seasonal state недоступен или короче 2*sp"}

    seasonal_values = fitted.states["seasonal"].to_numpy(dtype=float)
    cycle_1 = seasonal_values[-2 * sp : -sp]  # 2024-08..2025-07
    cycle_2 = seasonal_values[-sp:]  # 2025-08..2026-07

    corr = float(np.corrcoef(cycle_1, cycle_2)[0, 1])
    mae = float(np.mean(np.abs(cycle_2 - cycle_1)))
    full_range = float(seasonal_values.max() - seasonal_values.min())
    return {
        "cycle_1": [round(float(x), 6) for x in cycle_1],
        "cycle_2": [round(float(x), 6) for x in cycle_2],
        "corr": round(corr, 6),
        "mae": round(mae, 6),
        "mae_rel_to_full_range": round(mae / full_range, 6) if full_range != 0 else float("nan"),
        "caveat": "цикл 1 частично определяет эвристику инициализации — не независимая проверка внешней устойчивости",
    }


def compare_vectors(full: ParamVector, other: ParamVector) -> dict:
    """Численно сравнить два вектора: абсолютная и относительная разница
    уровня/тренда, и по сезонному профилю — корреляция Пирсона + средняя
    абсолютная разница (обе части сравниваются только там, где сезонность
    заведомо непустая, т.е. по всем 12 календарным месяцам).
    """
    level_diff = other.level - full.level
    level_rel = level_diff / full.level if full.level != 0 else float("nan")

    trend_diff = other.trend - full.trend

    seasonal_corr = float(np.corrcoef(full.seasonal, other.seasonal)[0, 1])
    seasonal_mae = float(np.mean(np.abs(full.seasonal - other.seasonal)))
    seasonal_full_range = float(full.seasonal.max() - full.seasonal.min())
    seasonal_mae_rel = seasonal_mae / seasonal_full_range if seasonal_full_range != 0 else float("nan")

    return {
        "level_diff_abs": round(float(level_diff), 6),
        "level_diff_rel": round(float(level_rel), 6),
        "trend_diff_abs": round(float(trend_diff), 6),
        "seasonal_corr": round(seasonal_corr, 6),
        "seasonal_mae": round(seasonal_mae, 6),
        "seasonal_mae_rel_to_full_range": round(seasonal_mae_rel, 6),
    }


def run_stability_check(series: pd.Series, label: str, full_vector: ParamVector) -> dict:
    """Пункт 3: снять вектор на ряде, укороченном на 3 и на 6 точек,
    сравнить с уже снятым вектором на полном ряде.

    Ключи результата различают два разных исхода намеренно:
    - not_measurable: AutoETS отказался фититься (ValueError на инициализации) —
      устойчивость НЕ измерена, это не то же самое, что "коэффициенты уплыли".
    - unstable: модель зафиттилась, но разница с полным рядом велика —
      это и есть "заметный дрейф" из формулировки issue #23.
    """
    result: dict = {"label": label, "n_points_full": len(series), "vector_full": full_vector.to_dict()}

    for drop in (3, 6):
        truncated = truncate_series(series, drop)
        n = len(truncated)
        entry: dict = {"n_points": n}
        try:
            truncated_vector = fit_autoets_vector(truncated)
            entry["vector"] = truncated_vector.to_dict()
            entry["comparison_vs_full"] = compare_vectors(full_vector, truncated_vector)
            entry["outcome"] = "measured"
        except Exception as exc:  # noqa: BLE001 — репортим отказ модели как данные MVP
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["outcome"] = "not_measurable"
        result[f"truncated_minus_{drop}"] = entry

    return result


def check_statsforecast_model_choice(series: pd.Series, sp: int = 12) -> dict:
    """Пункт 3b: какую спецификацию ETS выбирает statsforecast (независимая
    от sktime/statsmodels реализация — R-совместимый порт forecast::ets) на
    том же ряде. Не пересчитывает коэффициенты sktime-вектора — сравнивает
    только СТРУКТУРУ выбранной модели (есть сезонность или нет).
    """
    from statsforecast.models import AutoETS as SFAutoETS

    y = series.astype(float).to_numpy()
    model = SFAutoETS(season_length=sp)
    model.fit(y)
    components = model.model_["components"]  # например "MNNN" = mult. error, no trend, no seasonal
    has_seasonal = components[2] != "N"
    return {
        "method": model.model_["method"],
        "components": components,
        "has_seasonal": has_seasonal,
        "aic": round(float(model.model_["aic"]), 4),
    }


def run_discrimination_check(vectors: dict[str, ParamVector]) -> dict:
    """Пункт 4: различает ли вектор сезонный профиль «новогодних подарков»
    от плоского профиля «купить телефон» — по размаху сезонной компоненты
    и по позиции пика.
    """
    result = {}
    for label, vector in vectors.items():
        seasonal = vector.seasonal
        peak_month = int(np.argmax(seasonal)) + 1  # 1..12
        result[label] = {
            "seasonal_range": round(float(seasonal.max() - seasonal.min()), 6),
            "seasonal_std": round(float(seasonal.std()), 6),
            "peak_month": peak_month,
            "peak_value": round(float(seasonal.max()), 6),
        }
    return result


def main() -> int:
    report: dict = {"fixtures": {}}
    full_vectors: dict[str, ParamVector] = {}
    any_not_measurable = False
    any_unstable = False

    for label, path in FIXTURES.items():
        series = load_dynamics_csv(path)
        full_vector = fit_autoets_vector(series)
        full_vectors[label] = full_vector

        stability = run_stability_check(series, label, full_vector)
        stability["within_series_cycle_comparison"] = compare_seasonal_cycles_within_full_series(series)
        stability["statsforecast_cross_check"] = {
            f"drop_{drop}": check_statsforecast_model_choice(truncate_series(series, drop)) for drop in (0, 3, 6)
        }
        report["fixtures"][label] = stability

        for drop in (3, 6):
            entry = stability[f"truncated_minus_{drop}"]
            if entry["outcome"] == "not_measurable":
                any_not_measurable = True
                continue
            comparison = entry["comparison_vs_full"]
            # Порог: относительная разница уровня > 20% ИЛИ корреляция
            # сезонного профиля < 0.5 считается "заметным дрейфом".
            if abs(comparison["level_diff_rel"]) > 0.2 or comparison["seasonal_corr"] < 0.5:
                any_unstable = True

    report["stability_not_measurable"] = any_not_measurable
    report["stability_unstable"] = any_unstable
    stop_condition_triggered = any_not_measurable or any_unstable
    report["stability_stop_condition_triggered"] = stop_condition_triggered

    if not stop_condition_triggered:
        report["discrimination"] = run_discrimination_check(full_vectors)
    else:
        report["discrimination"] = "SKIPPED — стоп-условие устойчивости сработало (пункт 2 issue #23)"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

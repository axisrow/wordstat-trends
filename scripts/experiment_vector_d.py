"""MVP-проверка гипотезы #23 (независимый вариант "Г"): устойчивость вектора
параметров AutoETS, снятого с фикстур Wordstat. Две итерации, две грануляции:

ПЕРВАЯ ИТЕРАЦИЯ (main(), sp=12, месячные фикстуры, уже собраны) —
отрицательный результат, см. docs/EXPERIMENT_VECTOR_D.md.
Формат фикстур — см. docs/DATA.md: UTF-8 с BOM, переводы строк только CR
(`\\r`, без `\\n`), разделитель полей `;`, десятичная запятая, период вида
"август 2024" (русское название месяца текстом + год).

Порядок проверки первой итерации (обязателен, см. issue #23):
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

ВТОРАЯ ИТЕРАЦИЯ (main_daily(), sp=7, дневные фикстуры, окно 56 дней = 8
полных недель пн→вс) — предрегистрация в docs/EXPERIMENT_VECTOR_D.md,
прогон ждёт дневных фикстур (DAILY_FIXTURES, собираются отдельно). Формат
дневного «Периода» — DD.MM.YYYY (например "22.06.2026"), отдельный парсер
(_parse_daily_period, load_daily_dynamics_csv) — НЕ переиспользует русско-
месячный. Перед фитом обязательна проверка check_weekday_balance() —
каждый день недели должен встречаться ровно 8 раз на 56-дневном окне.

Запуск:
- python scripts/experiment_vector_d.py          -> первая итерация
- python scripts/experiment_vector_d.py --daily  -> вторая итерация
(время выполнения не измеряется и не репортится — не характеристика метода)
"""

from __future__ import annotations

import json
import math
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

# Вторая итерация (sp=7, дневная грануляция) — фикстуры собраны сбором,
# общим для всех параллельных воркеров одного эксперимента (не собирались
# этим скриптом), см. docs/EXPERIMENT_VECTOR_D.md, вторая итерация.
# main_daily() явно и понятно падает, если файла нет — не подставляет и не
# синтезирует данные вместо него.
DAILY_FIXTURES = {
    "seasonal (новогодние подарки)": FIXTURES_DIR / "dynamics_daily_seasonal_mvp.csv",
    "high_freq (купить телефон)": FIXTURES_DIR / "dynamics_daily_high_freq_mvp.csv",
    "mid_freq (курсы английского)": FIXTURES_DIR / "dynamics_daily_mid_freq_mvp.csv",
}

# Пороги устойчивости первой итерации (sp=12, месячные, main()) — объявлены
# в первом прогоне этого варианта MVP, независимо от второй итерации.
# НАМЕРЕННО отличаются от DAILY_STABILITY_THRESHOLDS ниже (0.5 vs 0.7 для
# корреляции, нет отдельного порога MAE/размах) — это два самостоятельных
# набора критериев для двух самостоятельных предрегистраций, не забытое
# переиспользование одной константы.
MONTHLY_STABILITY_THRESHOLDS = {
    "min_seasonal_corr": 0.5,
    "max_level_diff_rel_abs": 0.2,
}

# Предрегистрированные пороги устойчивости второй итерации (docs/EXPERIMENT_VECTOR_D.md):
# объявлены ДО прогона на реальных данных, не корректируются по результату.
DAILY_STABILITY_THRESHOLDS = {
    "min_seasonal_corr": 0.7,
    "max_level_diff_rel_abs": 0.2,
    "max_seasonal_mae_rel": 0.3,
}

# Train-окно второй итерации: 56 дней = 8 полных недель (пн→вс), кратно sp=7.
# Собранные фикстуры дают 58 строк (23.06–19.08.2026) — конечная дата в
# заголовке графика (20.08.2026) на день БОЛЬШЕ последней строки данных
# (эксклюзивная граница), поэтому реальных строк 58, не 59. Train = первые
# 56 строк (23.06–17.08.2026); holdout = ОСТАВШИЕСЯ строки после train,
# считается как len(series) - TRAIN_WINDOW_DAYS, а не как фиксированное
# число — на этих фикстурах это 2 дня (18–19.08), не 3.
TRAIN_WINDOW_DAYS = 56

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
    """"август 2024" -> pandas Period с частотой M. Формат месячных фикстур
    первой итерации (sp=12): русское название месяца текстом + год.
    """
    month_name, year = period.strip().split()
    month = RU_MONTHS[month_name.lower()]
    return pd.Period(year=int(year), month=month, day=1, freq="M")


def _parse_daily_period(period: str) -> pd.Period:
    """"22.06.2026" -> pandas Period с частотой D. Формат дневных фикстур
    второй итерации (sp=7): дата с точками DD.MM.YYYY — НЕ русский месяц,
    как в месячном парсере выше. Смешивать эти два парсера нельзя: дневная
    выгрузка Вордстата не использует текстовое название месяца.
    """
    day, month, year = period.strip().split(".")
    return pd.Period(year=int(year), month=int(month), day=int(day), freq="D")


def _read_raw_csv_lines(path: Path) -> list[str]:
    """Прочитать сырой CSV Вордстата (общая механика для месячных и дневных
    выгрузок, см. docs/DATA.md):
    - UTF-8 с BOM -> encoding="utf-8-sig".
    - Переводы строк только CR -> splitlines() (не ручной split("\\n")).
    Возвращает строки файла без дополнительной фильтрации — вызывающий код
    сам решает, что делать с заголовком и пустыми строками.
    """
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"Пустой файл: {path}")
    return lines


def load_dynamics_csv(path: Path) -> pd.Series:
    """Разобрать сырой CSV Вордстата (dynamics, МЕСЯЧНАЯ грануляция, sp=12,
    первая итерация) в месячный pd.Series.

    Особенности формата (docs/DATA.md), обрабатываются явно:
    - UTF-8 с BOM, переводы строк только CR — см. _read_raw_csv_lines.
    - Разделитель полей ";", 4-е поле (заголовок графика) пустое, игнорируется.
    - Число запросов: разделитель тысяч — обычный пробел, без десятичной части.
    - Период: "август 2024" — русский месяц текстом + год.
    """
    lines = _read_raw_csv_lines(path)

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

    if not series.index.is_unique:
        raise ValueError(f"Дубликаты месяцев в {path}: {series.index[series.index.duplicated()].tolist()}")
    expected = pd.period_range(start=series.index.min(), end=series.index.max(), freq="M")
    if not series.index.equals(expected):
        missing = expected.difference(series.index)
        raise ValueError(f"Пропуски месяцев (непрерывность нарушена) в {path}: {missing.tolist()}")

    return series


def load_daily_dynamics_csv(path: Path) -> pd.Series:
    """Разобрать сырой CSV Вордстата (dynamics, ДНЕВНАЯ грануляция, sp=7,
    вторая итерация) в дневной pd.Series.

    Отличия от load_dynamics_csv (месячный парсер НЕ переиспользуется):
    - Заголовок первой колонки — "Дата", НЕ "Период", как в месячной
      выгрузке (проверено на живом прогоне wordstat-cli).
    - Период: "22.06.2026" — дата с точками DD.MM.YYYY, не русский месяц.
    - Заголовок графика (4-е поле CSV) содержит конечную дату диапазона,
      которая на день БОЛЬШЕ последней строки данных (эксклюзивная граница) —
      даты берутся из строк данных, заголовок графика не используется как
      источник дат.
    Остальная механика формата совпадает с месячным парсером: BOM, CR-only
    переводы строк, разделитель ";", пробел как разделитель тысяч; как и
    месячный, проверяет уникальность и непрерывность дат целиком по файлу.
    """
    lines = _read_raw_csv_lines(path)

    header = lines[0].split(";")
    assert header[0] == "Дата", f"Неожиданный заголовок в {path}: {header}"

    records: list[tuple[pd.Period, float]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(";")
        period_raw, count_raw = fields[0], fields[1]
        period = _parse_daily_period(period_raw)
        count = float(count_raw.replace(" ", "").replace(" ", ""))
        records.append((period, count))

    periods, counts = zip(*records, strict=True)
    series = pd.Series(counts, index=pd.PeriodIndex(periods, freq="D"), name="count")
    series = series.sort_index()

    # Та же проверка целостности временной оси, что в месячном парсере
    # (load_dynamics_csv) — внутри самой функции загрузки, на всём файле
    # (включая holdout-хвост), а не только в вызывающем коде на train-окне
    # (issue #33, находка 5). check_weekday_balance() остаётся отдельной
    # дополнительной проверкой распределения по дням недели.
    if not series.index.is_unique:
        raise ValueError(f"Дубликаты дат в {path}: {series.index[series.index.duplicated()].tolist()}")
    expected = pd.period_range(start=series.index.min(), end=series.index.max(), freq="D")
    if not series.index.equals(expected):
        missing = expected.difference(series.index)
        raise ValueError(f"Пропуски дат (непрерывность нарушена) в {path}: {missing.tolist()}")

    return series


def check_weekday_balance(series: pd.Series) -> dict:
    """Предрегистрированная проверка для окна второй итерации (sp=7): каждый
    день недели должен встречаться РОВНО одинаковое число раз (8 раз на
    56-дневном окне) — иначе недельный профиль смещён в пользу дней,
    представленных чаще. Не запускать AutoETS(sp=7), пока эта проверка не
    прошла (см. docs/EXPERIMENT_VECTOR_D.md, вторая итерация).

    Баланс частот weekday сам по себе не гарантирует корректную временную
    ось: окно с пропущенной датой и дубликатом другой даты того же дня
    недели может сохранить равные частоты по всем 7 weekday, оставаясь
    некорректным (не непрерывная последовательность дней). Поэтому
    balanced дополнительно требует уникальности и непрерывности дат.
    """
    weekday_names = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    weekdays = series.index.to_timestamp().weekday
    counts_by_index = pd.Series(weekdays).value_counts().sort_index()
    counts = {weekday_names[i]: int(counts_by_index.get(i, 0)) for i in range(7)}
    is_contiguous_and_unique = series.index.is_unique and (
        len(series) == 0
        or series.index.equals(pd.period_range(start=series.index.min(), end=series.index.max(), freq="D"))
    )
    balanced = len(set(counts.values())) == 1 and is_contiguous_and_unique
    return {"counts_by_weekday": counts, "balanced": balanced}


@dataclass
class ParamVector:
    """Вектор параметров модели: уровень, тренд, sp сезонных коэффициентов.

    has_trend=False означает, что AutoETS выбрал спецификацию БЕЗ трендовой
    компоненты (не что тренд измерен и равен нулю) — trend в этом случае 0.0
    по построению, а не по данным.

    has_seasonal=False означает то же самое для сезонности: AutoETS выбрал
    спецификацию БЕЗ сезонной компоненты — seasonal в этом случае [0.0]*sp
    по построению (модель её не оценивала), а не "измеренный плоский
    профиль". Различие существенно для метрик сравнения: корреляция и MAE
    сезонного профиля не определены/бессмысленны, если хотя бы один из
    сравниваемых векторов has_seasonal=False (см. to_dict).
    """

    level: float
    trend: float
    has_trend: bool
    has_seasonal: bool
    seasonal: np.ndarray  # длина sp, индекс 0 = январь (sp=12) или понедельник (sp=7)
    model_spec: str  # "error/trend/seasonal[/damped]", как выбрал AutoETS

    def to_dict(self) -> dict:
        return {
            "level": round(float(self.level), 6),
            "trend": round(float(self.trend), 6) if self.has_trend else None,
            "model_spec": self.model_spec,
            "has_seasonal": self.has_seasonal,
            "seasonal": [round(float(x), 6) for x in self.seasonal] if self.has_seasonal else None,
        }


def _seasonal_state_is_valid(fitted, seasonal: str | None) -> bool:
    """Содержательная валидность сезонного состояния зафиченной модели.

    statsmodels не гарантирует, что «сошедшийся» по критерию SLSQP фит
    осмыслен: на коротких рядах оптимизация может остановиться в вырожденной
    точке, где мультипликативные сезонные множители уходят ≤ 0 (в CI на
    linux/OpenBLAS 11 из 12 множителей были ≈ -1e7 при строго положительном
    ряде — issue #37). Для seasonal="mul" множитель ≤ 0 или не-finite — это
    не альтернативная модель, а численный мусор: такая сезонность делает
    прогноз отрицательным и не имеет содержательной интерпретации.
    """
    if seasonal is None:
        return True
    values = fitted.states["seasonal"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return False
    if seasonal == "mul" and np.any(values <= 0):
        return False
    return True


def _fit_autoets(series: pd.Series, sp: int = 12, information_criterion: str = "aic"):
    """Зафитить AutoETS на ряде и вернуть fitted-модель (statsmodels
    ETSResults). Выделено из fit_autoets_vector, чтобы диагностические
    функции могли переиспользовать уже посчитанный фит вместо дублирующего
    рефита (issue #33, находка 3).

    Выбор спецификации воспроизводит sktime AutoETS(auto=True): та же сетка
    error × trend × seasonal × damped (для строго положительного ряда —
    2×2×3×2 минус вырожденные damped-без-тренда), тот же information
    criterion, несошедшиеся фиты (ConvergenceWarning / mle_retvals) получают
    NaN и выбывают из выбора. Отличие от sktime (issue #37): добавлен фильтр
    _seasonal_state_is_valid — фиты с невалидным сезонным состоянием (mul
    множитель ≤ 0) тоже выбывают. Без этого фильтра выбор AIC на 24 точках
    платформо-зависим: на linux/OpenBLAS выигрывал mul/add/mul/damped с
    вырожденным профилем (argmax=0 вместо декабрьского 11), на macOS/
    Accelerate — корректный mul/add/mul. Вектор — ядро гипотезы #23 и не
    должен молча меняться от реализации BLAS. С фильтром на macOS выбор не
    меняется вовсе (damped-вариант и так проигрывал по AIC), а вырожденные
    linux-фиты отсекаются содержательным критерием, а не тюнингом порогов.
    """
    import warnings
    from itertools import product

    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = series.reset_index(drop=True).astype(float)
    y.index = pd.RangeIndex(len(y))

    ic_values: list[float] = []
    fitted_results = []
    for error, trend, seasonal, damped in product(
        ("add", "mul"), (None, "add"), (None, "add", "mul"), (False, True)
    ):
        if trend is None and damped:
            continue  # демпфировать нечего — как в сетке sktime
        if error == "add" and (trend == "mul" or seasonal == "mul"):
            continue  # restrict=True sktime: аддитивная ошибка с мультипликативными
            # компонентами исключается из перебора (ограничение Hyndman)
        model = ETSModel(
            y,
            error=error,
            trend=trend,
            damped_trend=damped,
            seasonal=seasonal,
            seasonal_periods=sp,
        )
        converged = True
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                fitted = model.fit(disp=False)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                converged = False
            mle_retvals = getattr(fitted, "mle_retvals", None)
            if isinstance(mle_retvals, dict) and not bool(mle_retvals.get("converged", True)):
                converged = False
        except Exception:  # noqa: BLE001 — сегмент сетки не фитится вовсе (напр., < 2*sp): NaN, как у sktime
            ic_values.append(float("nan"))
            fitted_results.append(None)
            continue
        ic_values.append(
            float(getattr(fitted, information_criterion))
            if converged and _seasonal_state_is_valid(fitted, seasonal)
            else float("nan")
        )
        fitted_results.append(fitted)

    if not fitted_results or np.all(np.isnan(ic_values)):
        raise ValueError(
            f"ни одна спецификация ETS(sp={sp}) не сошлась или не прошла "
            f"проверку сезонного состояния на ряде длиной {len(y)}"
        )
    return fitted_results[int(np.nanargmin(ic_values))]


def fit_autoets_vector(
    series: pd.Series, sp: int = 12, information_criterion: str = "aic", calendar_anchor: str = "month"
) -> ParamVector:
    """Снять вектор параметров AutoETS(sp=sp) с ряда.

    Использует statsmodels ETSModel с перебором спецификаций, воспроизводящим
    выбор sktime AutoETS(auto=True) (см. _fit_autoets). Возвращает уровень,
    тренд (slope на конец выборки) и полный сезонный профиль длиной sp.

    calendar_anchor определяет, как выровнять сезонный профиль по
    содержательным координатам (а не по позиции в массиве), чтобы ряды с
    разным стартовым периодом были сравнимы напрямую:
    - "month" (sp=12, первая итерация): индекс 0 = январь, ..., 11 = декабрь.
    - "weekday" (sp=7, вторая итерация): индекс 0 = понедельник, ..., 6 = воскресенье.
    Другие sp без явного anchor-соответствия не поддерживаются — вектор в
    этом случае остаётся выровненным просто "с конца ряда назад", без
    привязки к календарю (используйте только для sp из {7, 12}).
    """
    return vector_from_fitted(
        _fit_autoets(series, sp=sp, information_criterion=information_criterion),
        series,
        sp=sp,
        calendar_anchor=calendar_anchor,
    )


def vector_from_fitted(fitted, series: pd.Series, sp: int = 12, calendar_anchor: str = "month") -> ParamVector:
    """Извлечь ParamVector из УЖЕ зафиченной модели (см. _fit_autoets) —
    без повторного фита. Ряд нужен только для календарного якоря последней
    точки. Логика извлечения прежняя (fit_autoets_vector), вынесена, чтобы
    полный фит в main() переиспользовался диагностикой (issue #33, находка 3).
    """
    states = fitted.states  # DataFrame с одной строкой на точку ряда: level, [trend], [seasonal]

    model = fitted.model  # ETSModel: хранит выбранную спецификацию error/trend/seasonal
    has_trend = "trend" in states.columns
    model_spec = f"{model.error}/{model.trend}/{model.seasonal}"
    if getattr(model, "damped_trend", False):
        model_spec += "/damped"

    last_state = states.iloc[-1]
    level = float(last_state["level"])
    trend = float(last_state["trend"]) if has_trend else 0.0

    has_seasonal = "seasonal" in states.columns and len(states) >= sp
    if has_seasonal:
        # statsmodels хранит ОДНО сезонное значение на строку (не sp столбцов) —
        # последние sp строк дают ровно один полный сезонный цикл, где строка
        # states.iloc[-1] соответствует последней точке ряда, states.iloc[-2] —
        # предпоследней, и т.д. Раскладываем эти sp значений по календарным
        # координатам, используя координату последней точки как якорь.
        raw_seasonal = states["seasonal"].tail(sp).to_numpy(dtype=float)
    else:
        raw_seasonal = np.zeros(sp)

    last_period = series.index[-1]
    if calendar_anchor == "month":
        last_coord = last_period.month  # 1..12
    elif calendar_anchor == "weekday":
        last_coord = last_period.to_timestamp().weekday() + 1  # 1..7 (1=понедельник)
    else:
        raise ValueError(f"Неизвестный calendar_anchor: {calendar_anchor!r}")

    calendar_seasonal = np.zeros(sp)
    # raw_seasonal[-1] — последняя точка (координата last_coord), raw_seasonal[-2] —
    # координата перед ней, и т.д. в обратном порядке.
    for offset, value in enumerate(reversed(raw_seasonal.tolist())):
        coord = ((last_coord - 1 - offset) % sp) + 1  # 1..sp
        calendar_seasonal[coord - 1] = value

    return ParamVector(
        level=level,
        trend=trend,
        has_trend=has_trend,
        has_seasonal=has_seasonal,
        seasonal=calendar_seasonal,
        model_spec=model_spec,
    )


def truncate_series(series: pd.Series, drop_last: int) -> pd.Series:
    """Укоротить ряд, отбросив drop_last последних точек."""
    if drop_last <= 0:
        return series
    return series.iloc[:-drop_last]


def compare_seasonal_cycles_within_full_series(series: pd.Series, sp: int = 12, fitted_forecaster=None) -> dict:
    """Дополнительное (не требующее рефита) измерение устойчивости: сравнить
    сезонный коэффициент цикла 1 (первые sp строк states — 2024-08..2025-07)
    и цикла 2 (последние sp строк — 2025-08..2026-07) внутри ОДНОГО фита на
    полном 24-точечном ряде.

    fitted_forecaster — результат _fit_autoets на том же ряде, если он уже
    посчитан вызывающим кодом (main() фитит тот же ряд для full_vector):
    переиспользуется вместо дублирующего рефита (issue #33, находка 3). Без
    него (None) функция фитит сама — самостоятельный запуск диагностики.

    Не заменяет проверку "пересчитать вектор на укороченном ряде" (пункт 2/3
    issue #23) — это она физически неисполнима на 24 точках (см. отчёт).
    Здесь измеряется другое: дрейфует ли сезонный профиль между двумя
    имеющимися циклами ОДНОГО фита. Оговорка: цикл 1 частично загрязнён
    эвристикой инициализации (statsmodels использует его же для старта), так
    что даже совпадение циклов 1 и 2 не отменяет вывод о непроверяемости
    внешней устойчивости (рефит на других данных).
    """
    if len(series) < 2 * sp:
        return {"error": f"нужно >= {2 * sp} точек, есть {len(series)}"}

    fitted = fitted_forecaster if fitted_forecaster is not None else _fit_autoets(series, sp=sp)

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
    абсолютная разница.

    both_seasonal=False означает, что хотя бы один из двух векторов не имеет
    сезонной компоненты (has_seasonal=False у full или other) — в этом
    случае seasonal_corr/seasonal_mae/seasonal_mae_rel_to_full_range
    вычислены на нулевых заглушках и НЕ являются измерением сходства
    профилей; поле both_seasonal должно проверяться явно перед тем, как
    полагаться на эти значения (иначе NaN < порог тихо проходит как
    "стабильно", хотя фактически сезонность не сравнивалась вовсе).

    Внимание: MAE и относительный MAE сравнимы только если оба вектора
    получены с одинаковой параметризацией сезонной компоненты (mul vs add) —
    сравнение мультипликативных коэффициентов (в районе 1.0) с аддитивными
    сдвигами (в единицах ряда) даёт формально исчисляемое, но содержательно
    бессмысленное число. Поэтому seasonal_spec_compatible проверяет
    совпадение сезонного компонента model_spec ("error/trend/SEASONAL[/damped]",
    третий слот) обоих векторов; при несовпадении сезонные метрики
    неприменимы, и is_comparison_unstable трактует это как нестабильность
    (issue #33, находка 4).
    """
    level_diff = other.level - full.level
    level_rel = level_diff / full.level if full.level != 0 else float("nan")

    trend_diff = other.trend - full.trend

    both_seasonal = full.has_seasonal and other.has_seasonal
    # Совпадение сезонного компонента спецификации (mul vs add); полные
    # model_spec могут расходиться в error/trend/damped — это на сезонные
    # метрики не влияет.
    seasonal_spec_compatible = full.model_spec.split("/")[2] == other.model_spec.split("/")[2]
    seasonal_corr = float(np.corrcoef(full.seasonal, other.seasonal)[0, 1]) if both_seasonal else float("nan")
    seasonal_mae = float(np.mean(np.abs(full.seasonal - other.seasonal))) if both_seasonal else float("nan")
    seasonal_full_range = float(full.seasonal.max() - full.seasonal.min()) if both_seasonal else 0.0
    seasonal_mae_rel = (
        seasonal_mae / seasonal_full_range if both_seasonal and seasonal_full_range != 0 else float("nan")
    )

    return {
        "level_diff_abs": round(float(level_diff), 6),
        "level_diff_rel": round(float(level_rel), 6),
        "trend_diff_abs": round(float(trend_diff), 6),
        "both_seasonal": both_seasonal,
        "seasonal_spec_compatible": seasonal_spec_compatible,
        "seasonal_corr": round(seasonal_corr, 6),
        "seasonal_mae": round(seasonal_mae, 6),
        "seasonal_mae_rel_to_full_range": round(seasonal_mae_rel, 6),
    }


def run_stability_check(
    series: pd.Series,
    label: str,
    full_vector: ParamVector,
    drops: tuple[int, ...] = (3, 6),
    sp: int = 12,
    calendar_anchor: str = "month",
) -> dict:
    """Пункт 3: снять вектор на ряде, укороченном на каждое значение из
    `drops` точек, сравнить с уже снятым вектором на полном ряде.

    Первая итерация (sp=12, месячные): drops=(3, 6) — укорочение в месяцах.
    Вторая итерация (sp=7, дневные): drops=(7, 14) — укорочение в днях
    (−1 и −2 недели), calendar_anchor="weekday".

    Ключи результата различают два разных исхода намеренно:
    - not_measurable: AutoETS отказался фититься (ValueError на инициализации) —
      устойчивость НЕ измерена, это не то же самое, что "коэффициенты уплыли".
    - unstable: модель зафиттилась, но разница с полным рядом велика —
      это и есть "заметный дрейф" из формулировки issue #23.

    Ловится только ValueError — ожидаемый класс отказа AutoETS.fit() на
    слишком коротком ряде (issue #33, находка 2). Баги в последующих
    to_dict()/compare_vectors (KeyError, несовместимость shape) — это дефект
    кода, а не экспериментальная данность: они пробрасываются наружу, а не
    репортятся как not_measurable.
    """
    result: dict = {"label": label, "n_points_full": len(series), "vector_full": full_vector.to_dict()}

    for drop in drops:
        truncated = truncate_series(series, drop)
        n = len(truncated)
        entry: dict = {"n_points": n}
        try:
            truncated_vector = fit_autoets_vector(truncated, sp=sp, calendar_anchor=calendar_anchor)
        except ValueError as exc:  # ожидаемый отказ модели, не баг кода
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["outcome"] = "not_measurable"
            result[f"truncated_minus_{drop}"] = entry
            continue
        entry["vector"] = truncated_vector.to_dict()
        entry["comparison_vs_full"] = compare_vectors(full_vector, truncated_vector)
        entry["outcome"] = "measured"
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


def check_sktime_model_choice(series: pd.Series, sp: int = 12, information_criterion: str = "aic") -> dict:
    """Какую структуру модели выбирает основной бэкенд (sktime/statsmodels)
    при заданном information_criterion. Используется, чтобы отделить эффект
    выбора критерия (AIC vs AICc) от эффекта различия реализаций при
    сравнении с statsforecast (см. docs/EXPERIMENT_VECTOR_D.md, раздел 3):
    fit_autoets_vector() в этом скрипте всегда использует "aic" — осознанный
    выбор автора, а не дефолт sktime (дефолт — "aicc", как и у statsforecast).
    """
    from sktime.forecasting.ets import AutoETS

    y = series.reset_index(drop=True).astype(float)
    y.index = pd.RangeIndex(len(y))

    forecaster = AutoETS(auto=True, sp=sp, n_jobs=1, information_criterion=information_criterion)
    forecaster.fit(y)
    fitted = forecaster._fitted_forecaster  # type: ignore[attr-defined]
    model = fitted.model
    # Та же проверка, что в fit_autoets_vector (states.columns + длина) —
    # не model.seasonal is not None: это два формально разных вопроса
    # («что запросили при фите» против «можно ли извлечь sp значений
    # сезонных states»), которые могут в принципе разойтись; здесь важно
    # именно извлекаемое состояние, а не сама спецификация.
    has_seasonal = "seasonal" in fitted.states.columns and len(fitted.states) >= sp
    return {
        "information_criterion": information_criterion,
        "error": model.error,
        "trend": model.trend,
        "seasonal": model.seasonal,
        "has_seasonal": has_seasonal,
        "damped": bool(model.damped_trend),
    }


def empirical_weekday_profile(series: pd.Series) -> np.ndarray:
    """Простое среднее по дням недели, без всякой модели — независимая опора
    для оценки сезонного профиля (см. docs/EXPERIMENT_VECTOR_D.md, вторая
    итерация, «купить телефон»): профиль, разошедшийся с эмпирикой, означал
    бы ошибку снятия, а не свойство спроса. Возвращает вектор длины 7,
    нормированный к среднему 1, индекс 0 = понедельник.
    """
    weekdays = series.index.to_timestamp().weekday
    grouped = pd.Series(series.to_numpy(), index=weekdays).groupby(level=0).mean()
    profile = np.array([grouped.get(i, np.nan) for i in range(7)])
    return profile / np.nanmean(profile)


def fit_statsforecast_seasonal_profile(series: pd.Series, sp: int = 7) -> tuple[np.ndarray | None, str]:
    """Извлечь нормированный сезонный профиль из statsforecast (не только
    структуру модели, как check_statsforecast_model_choice — сами
    коэффициенты), выровненный по дням недели так же, как в
    fit_autoets_vector. Используется только для диагностики граничного
    случая («купить телефон», см. docs/EXPERIMENT_VECTOR_D.md) — НЕ входит
    в основной вектор и не участвует в стоп-условии устойчивости (см.
    test_statsforecast_is_a_diagnostic_backend_not_wired_into_the_stability_gate).

    Возвращает (None, components), если модель не включает сезонность.
    """
    from statsforecast.models import AutoETS as SFAutoETS

    y = series.astype(float).to_numpy()
    model = SFAutoETS(season_length=sp)
    model.fit(y)
    components = model.model_["components"]
    has_seasonal = components[2] != "N"
    if not has_seasonal:
        return None, components

    has_trend = components[1] != "N"
    seasonal_col = 1 + (1 if has_trend else 0)
    states = model.model_["states"]
    level = float(states[-1, 0])
    seasonal_states = states[-sp:, seasonal_col]

    last_period = series.index[-1]
    last_weekday = last_period.to_timestamp().weekday()  # 0=понедельник
    calendar_seasonal = np.zeros(sp)
    for offset, value in enumerate(reversed(seasonal_states.tolist())):
        coord = (last_weekday - offset) % sp
        calendar_seasonal[coord] = value

    normalized = 1 + calendar_seasonal / level
    return normalized, components


def run_discrimination_check(vectors: dict[str, ParamVector]) -> dict:
    """Пункт 4: различает ли вектор сезонный профиль «новогодних подарков»
    от плоского профиля «купить телефон» — по размаху сезонной компоненты
    и по позиции пика.

    "peak_position" — 1-based индекс пика внутри вектора: для sp=12 (первая
    итерация, calendar_anchor="month") это номер месяца (1=январь..12=декабрь);
    для sp=7 (вторая итерация, calendar_anchor="weekday") это номер дня недели
    (1=понедельник..7=воскресенье). Само значение sp/anchor здесь не хранится
    — вызывающий код (main/main_daily) знает, какую итерацию репортит.
    """
    result = {}
    for label, vector in vectors.items():
        seasonal = vector.seasonal
        peak_position = int(np.argmax(seasonal)) + 1  # 1-based
        result[label] = {
            "seasonal_range": round(float(seasonal.max() - seasonal.min()), 6),
            "seasonal_std": round(float(seasonal.std()), 6),
            "peak_position": peak_position,
            "peak_value": round(float(seasonal.max()), 6),
        }
    return result


def is_comparison_unstable(comparison: dict, thresholds: dict) -> bool:
    """Единая логика стоп-условия устойчивости, общая для main() и
    main_daily().

    Все метрики сравнения (level_diff_rel, при both_seasonal=True —
    seasonal_corr и, при наличии в thresholds, seasonal_mae_rel_to_full_range)
    могут прийти NaN в вырожденных случаях: full.level == 0 (level_diff_rel),
    константный сезонный вектор с нулевой дисперсией (seasonal_corr через
    np.corrcoef) или плоский полный сезонный профиль (seasonal_mae_rel).
    Прямое `nan < порог`/`nan > порог`/`abs(nan) > порог` в Python всегда
    возвращает False — что тихо трактовало бы неизмеренный случай как
    "стабильно". Поэтому каждая метрика, участвующая в сравнении,
    явно проверяется на NaN и трактуется как нестабильность.

    both_seasonal=False само по себе тоже считается нестабильностью:
    сезонность заявленно есть на полном ряде, но не воспроизводится на
    укороченном (или наоборот), значит вектор параметров модели не
    устойчив в заявленном виде.

    seasonal_spec_compatible=False (mul vs add сезонная компонента между
    full и other, см. compare_vectors) — тоже нестабильность: сезонные
    метрики при несовпадении параметризации формально исчислимы, но
    несопоставимы (артефакт разных шкал, а не дрейф профиля), поэтому
    сравнение считается неприменимым, аналогично both_seasonal=False
    (issue #33, находка 4).
    """
    level_diff_rel = comparison["level_diff_rel"]
    if math.isnan(level_diff_rel) or abs(level_diff_rel) > thresholds["max_level_diff_rel_abs"]:
        return True
    if not comparison["both_seasonal"]:
        return True
    if not comparison["seasonal_spec_compatible"]:
        return True
    seasonal_corr = comparison["seasonal_corr"]
    if math.isnan(seasonal_corr) or seasonal_corr < thresholds["min_seasonal_corr"]:
        return True
    max_seasonal_mae_rel = thresholds.get("max_seasonal_mae_rel")
    if max_seasonal_mae_rel is not None:
        seasonal_mae_rel = comparison["seasonal_mae_rel_to_full_range"]
        if math.isnan(seasonal_mae_rel) or seasonal_mae_rel > max_seasonal_mae_rel:
            return True
    return False


def main() -> int:
    report: dict = {"fixtures": {}}
    full_vectors: dict[str, ParamVector] = {}
    any_not_measurable = False
    any_unstable = False

    for label, path in FIXTURES.items():
        series = load_dynamics_csv(path)
        # Один фит на фикстуру: full_vector и within_series-диагностика
        # переиспользуют один и тот же fitted (issue #33, находка 3).
        fitted = _fit_autoets(series)
        full_vector = vector_from_fitted(fitted, series, sp=12, calendar_anchor="month")
        full_vectors[label] = full_vector

        stability = run_stability_check(series, label, full_vector)
        stability["within_series_cycle_comparison"] = compare_seasonal_cycles_within_full_series(
            series, fitted_forecaster=fitted
        )
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
            # MONTHLY_STABILITY_THRESHOLDS: относительная разница уровня > 20%
            # ИЛИ корреляция сезонного профиля < 0.5 считается "заметным дрейфом".
            # both_seasonal=False тоже считается нестабильностью (см.
            # is_comparison_unstable) — сезонность не воспроизвелась на
            # укороченном ряде так же, как на полном.
            if is_comparison_unstable(comparison, MONTHLY_STABILITY_THRESHOLDS):
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
    # Exit code — machine-readable контракт (issue #33, находка 1): сработавшее
    # стоп-условие устойчивости обязано давать ненулевой код, чтобы
    # автоматизация (CI-проверка гипотезы, gate) не считала прогон успешным,
    # парся JSON ради единственного флага.
    return 1 if stop_condition_triggered else 0


def main_daily() -> int:
    """Вторая итерация MVP #23 (sp=7, дневная грануляция, окно 56 дней =
    8 полных недель, пн→вс). См. docs/EXPERIMENT_VECTOR_D.md, раздел
    «Вторая итерация» — предрегистрация критериев, объявленная ДО прогона.

    Требует дневных фикстур в DAILY_FIXTURES (собираются отдельно от этого
    скрипта — см. docs/EXPERIMENT_VECTOR_D.md о сериализации сбора между
    параллельными воркерами). Если файла нет, падает с понятной ошибкой —
    не подставляет и не синтезирует данные вместо него.

    Пороги устойчивости — DAILY_STABILITY_THRESHOLDS, предрегистрированы:
    corr >= 0.7, |level_diff_rel| <= 0.2, seasonal_mae_rel <= 0.3.
    """
    missing = [str(path) for path in DAILY_FIXTURES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Дневные фикстуры ещё не собраны (см. docs/EXPERIMENT_VECTOR_D.md, "
            f"вторая итерация): {missing}"
        )

    thresholds = DAILY_STABILITY_THRESHOLDS
    report: dict = {"fixtures": {}, "thresholds": thresholds, "train_window_days": TRAIN_WINDOW_DAYS}
    full_vectors: dict[str, ParamVector] = {}
    any_not_measurable = False
    any_unstable = False
    any_no_seasonal_component = False

    for label, path in DAILY_FIXTURES.items():
        full_series = load_daily_dynamics_csv(path)

        if len(full_series) < TRAIN_WINDOW_DAYS:
            raise ValueError(
                f"Окно '{label}' короче train-окна: {len(full_series)} строк < "
                f"{TRAIN_WINDOW_DAYS} требуемых. Не достраиваю синтетикой."
            )

        # Train — первые TRAIN_WINDOW_DAYS строк (23.06–17.08, 8 полных недель).
        # Holdout — ВСЁ, что осталось после train (не фиксированное число: на
        # собранных фикстурах это 2 дня, 18–19.08, не 3 — фактическая длина
        # ряда минус train, а не 58 - 56 посчитанное заранее).
        series = truncate_series(full_series, len(full_series) - TRAIN_WINDOW_DAYS)
        holdout = full_series.iloc[TRAIN_WINDOW_DAYS:]

        weekday_balance = check_weekday_balance(series)
        if not weekday_balance["balanced"]:
            raise ValueError(
                f"Train-окно '{label}' не сбалансировано по дням недели "
                f"(нужно ровно 8 на каждый день): {weekday_balance['counts_by_weekday']}"
            )

        full_vector = fit_autoets_vector(series, sp=7, calendar_anchor="weekday")
        full_vectors[label] = full_vector

        stability = run_stability_check(series, label, full_vector, drops=(7, 14), sp=7, calendar_anchor="weekday")
        stability["weekday_balance"] = weekday_balance
        stability["holdout"] = {
            "n_points": len(holdout),
            "dates": [str(p) for p in holdout.index],
            "note": "Не используется для фита — контроль вне обучающего окна, не входит в проверку устойчивости.",
        }
        stability["statsforecast_cross_check"] = {
            f"drop_{drop}": check_statsforecast_model_choice(truncate_series(series, drop), sp=7)
            for drop in (0, 7, 14)
        }
        # Кросс-проверка выбора критерия (докажено в первой итерации: AIC vs
        # AICc сами по себе меняют, видит ли sktime сезонность) — вызвано с
        # тем же information_criterion="aicc", что дефолт statsforecast, на
        # train-окне и обоих рефитах, чтобы отделить "разные критерии" от
        # "разные реализации" в статистике раздела 3 второй итерации.
        stability["sktime_aicc_cross_check"] = {
            f"drop_{drop}": check_sktime_model_choice(truncate_series(series, drop), sp=7, information_criterion="aicc")
            for drop in (0, 7, 14)
        }
        report["fixtures"][label] = stability

        for drop in (7, 14):
            entry = stability[f"truncated_minus_{drop}"]
            if entry["outcome"] == "not_measurable":
                any_not_measurable = True
                continue
            comparison = entry["comparison_vs_full"]
            if not comparison["both_seasonal"]:
                # Хотя бы один из двух векторов не имеет сезонной компоненты —
                # seasonal_corr/seasonal_mae_rel НЕ измерены (NaN), сравнение
                # неприменимо. Отдельно помечаем сам факт отсутствия сезонности
                # для честного отчёта; is_comparison_unstable ниже уже
                # трактует both_seasonal=False как нестабильность сама по себе
                # (не полагаясь на NaN-сравнение с порогом).
                any_no_seasonal_component = True
            if is_comparison_unstable(comparison, thresholds):
                any_unstable = True

    report["stability_not_measurable"] = any_not_measurable
    report["stability_unstable"] = any_unstable
    report["stability_no_seasonal_component_somewhere"] = any_no_seasonal_component
    stop_condition_triggered = any_not_measurable or any_unstable
    report["stability_stop_condition_triggered"] = stop_condition_triggered

    if not stop_condition_triggered:
        report["discrimination"] = run_discrimination_check(full_vectors)
    else:
        report["discrimination"] = "SKIPPED — стоп-условие устойчивости сработало (см. предрегистрацию)"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    # Тот же контракт, что в main() (issue #33, находка 1).
    return 1 if stop_condition_triggered else 0


if __name__ == "__main__":
    # `python scripts/experiment_vector_d.py` -> первая итерация (sp=12, месячные, уже собранные фикстуры).
    # `python scripts/experiment_vector_d.py --daily` -> вторая итерация (sp=7, дневные), как только фикстуры появятся.
    if "--daily" in sys.argv:
        sys.exit(main_daily())
    sys.exit(main())

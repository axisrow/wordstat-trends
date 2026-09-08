"""Структурные сдвиги (issue #7): разрыв в спросе против разрыва в измерении.

Эпик #1 (Фаза 0): в ряду с 2018 минимум три структурных сдвига — весна 2020
(COVID), весна 2022 (уход брендов), 2024–2025 (смена интерфейса и методики
самого Вордстата — артефакт измерения, а не спроса). Сезонная модель, принявшая
разрыв измерения за спрос, будет воспроизводить его каждый год и рекомендовать
закупку не в тот месяц.

Что делает модуль и чего он сознательно НЕ делает:

- Детекция точек разрыва — ``ruptures`` (Pelt/Binseg) на лог-десезонализированном
  ряде. Сезонная компонента Вордстата мультипликативна и охватывает 2–3 порядка
  (docs/DATA.md): без лога и удаления профиля месяца детектор видит не сдвиг
  уровня, а декабрь.
- Классификация каждого найденного разрыва по типу: переходный шок спроса
  (spike и возврат), устойчивый сдвиг спроса, разрыв измерения, неразрешённый.
  Из самого ряда устойчивый сдвиг спроса и разрыв измерения НЕРАЗЛИМИМЫ —
  разводит их внешняя информация о сборе: швы склейки окон и смены методики
  (#5/#6) и известные содержательные события. Даты передаются как гипотезы,
  которые детектор проверяет (``known_dates_coverage``), а не как хардкод точек.
- Решение по обработке — обязательный атрибут каждого разрыва (требование
  эпика): обрезка ряда, дамми-переменная, сегментная модель или «оставить
  как есть». Решение возвращается структурой, а не комментарием.

Границы: сбор данных, пропуски и склейка окон — зона задач #5/#6, модуль их
не выполняет, а только принимает их результат (месяцы-швы) на вход.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from ruptures import Binseg, Pelt

BreakKind = Literal["demand_transient", "demand_persistent", "measurement", "unresolved"]
Handling = Literal["keep", "dummy", "truncate", "segment"]

# Предрегистрированные параметры детекции (docs/BREAKS.md): зафиксированы до
# прогона на реальных фикстурах, по результату не подбирались.
#
# penalty Pelt: чем меньше, тем больше точек. Значение в единицах std
# десезонализированного лог-ряда подбирается на синтетике (шум против сдвига),
# до прогона на фикстурах.
PELT_PENALTY = 1.0
# Минимальная длина сегмента по обе стороны разрыва: на месячном ряде сдвиг,
# подтверждённый менее чем 3 месяцами, неотличим от переходного шока.
MIN_SEGMENT_SIZE = 3
# Сколько месяцев после разрыва ждём возврата к до-разрывному уровню, чтобы
# назвать шок переходным. COVID-подобный spike в месячных данных возвращается
# за 1–2 месяца.
REVERT_HORIZON = 2
# Возвратом считается снижение |отклонения| до этой доли величины самого сдвига.
REVERT_TOLERANCE = 0.5
# Допуск (в периодах) при сопоставлении известных дат с найденными разрывами.
KNOWN_DATE_TOLERANCE = 1
# Порог переходного шока: робастная z-оценка ЛОКАЛЬНОГО отклонения остатка от
# его центрированной скользящей медианы (нормировка 1.4826*MAD) выше порога.
# Локальная база обязательна: против медианы всего ряда устойчивый сдвиг
# уровня помечает «выбросом» весь пост-сдвиговый сегмент (воспроизведено:
# серия длиной 24 месяца на ряде с одним сдвигом ×5), а ступеньки скользящего
# уровня дают ложные шоки на чистом ряду. Шок ищется в обе стороны —
# отрицательный провал спроса равноправен положительному. Переходный шок
# длиной 1–2 месяца Pelt с min_size>=3 не видит по построению, поэтому шоки
# ищутся отдельным детектором до разрывов уровня.
SPIKE_Z_THRESHOLD = 3.5
# Абсолютный пол амплитуды шока (в лог-единицах): |отклонение| >= 0.3, т.е.
# +/-35 % от локального уровня. Двухусловное правило (z И амплитуда) отсекает
# маргинальные срабатывания чистого шума: при эмпирически тесном MAD z-score
# тяжеловатого шума достигает 3.5-4 на амплитудах ~8 %, тогда как содержательный
# месячный шок спроса — десятки процентов (COVID-инъекция ×4 даёт ~1.4).
SPIKE_MIN_ABS_LOG = 0.3
# Нечётное окно локальной медианы для шок-детектора: медиана робастна к 1–2
# месяцам шока и отслеживает сдвиги уровня, поэтому отклонение от неё видит
# шок, но не сдвиг.
SPIKE_LOCAL_WINDOW = 13
# Окно скользящей медианы для оценки сезонного профиля: 12 месяцев покрывают
# все месяцы года по одному разу, медиана не тянется за сдвигом уровня.
PROFILE_WINDOW = 12

_KIND_HANDLING: dict[BreakKind, Handling] = {
    "demand_transient": "keep",
    "demand_persistent": "dummy",
    "measurement": "truncate",
    "unresolved": "segment",
}

_HANDLING_RATIONALE: dict[Handling, str] = {
    "keep": (
        "Переходный шок спроса: уровень возвращается сам, сезонной модели "
        "мешать не нужно — дамми занизил бы аналог в следующем году."
    ),
    "dummy": (
        "Устойчивый сдвиг спроса: модель должна знать о смене уровня без "
        "потери ряда — индикатор «после даты» в регрессии/признаках."
    ),
    "truncate": (
        "Разрыв измерения: до-разрывная часть собрана другой методикой, "
        "моделировать её одним рядом нельзя — ряд усекается до пост-разрывной "
        "части."
    ),
    "segment": (
        "Неразрешённый устойчивый сдвиг: нет внешних оснований выбрать между "
        "спросом и измерением — ряд делится на сегменты по разрыву, решение "
        "остаётся за отчётом/человеком (см. docs/BREAKS.md)."
    ),
}


@dataclass(frozen=True)
class BreakPoint:
    """Одна точка разрыва и её классификация.

    ``at`` — последний период ДО разрыва (разрыв лежит между ``at`` и
    следующим периодом); если разрыв начинается с первого периода ряда,
    до-разрывного периода нет и ``at`` — первый период нового режима
    (помечено ``evidence["at_series_start"]``). ``delta_log`` — натуральный
    лог: для разрыва уровня это сдвиг среднего между сегментами (1.0 = ×e ≈
    +172 %); для шока (``evidence["spike"]``) — амплитуда пика относительно
    робастного (медианного) уровня ряда, величина того же смысла и шкалы.
    """

    at: pd.Period
    delta_log: float
    kind: BreakKind
    evidence: dict[str, object]


@dataclass(frozen=True)
class BreakDecision:
    """Разрыв + явное решение по обработке (требование эпика #1/#7)."""

    br: BreakPoint
    handling: Handling
    rationale: str


@dataclass(frozen=True)
class KnownDatesCoverage:
    """Сопоставление известных дат с найденными разрывами.

    Известная дата считается покрытой, если детектор нашёл разрыв на неё
    или в пределах ``KNOWN_DATE_TOLERANCE``. Непокрытая дата — это результат,
    а не ошибка: гипотеза «здесь был сдвиг» не подтвердилась детектором.
    """

    covered: tuple[pd.Period, ...]
    missed: tuple[pd.Period, ...]


def deseasonalize_log(series: pd.Series) -> pd.Series:
    """Логарифм ряда минус сезонный профиль периода (месяц для freq="M",
    день недели для freq="D").

    Мультипликативная сезонность на лог-шкале аддитивна. Профиль оценивается
    НЕ медианой «сырого» лог-значения своего периода: если в ряду есть сдвиг
    уровня, такая медиана смешивает до- и пост-сдвиговые годы, профиль
    «расщепляется» пополам и создаёт ложные разрывы каждые 6 месяцев (это
    воспроизведено и зафиксировано в docs/BREAKS.md). Вместо этого:

    1. из лог-ряда вычитается центрированная скользящая медиана
       (окно PROFILE_WINDOW) — локальный уровень, робастный и к сдвигу,
       и к одиночному выбросу;
    2. профиль периода — медиана этих отклонений по своему периоду;
       если у периода меньше двух наблюдений, профиль для него не оценивается
       (фактор 0) — единственное наблюдение оценило бы «сезонность» самого
       себя;
    3. остаток = лог-ряд минус профиль: локальный уровень (включая подлинные
       сдвиги) остаётся в данных, сезонный декабрь — нет.
    """
    if not isinstance(series.index, pd.PeriodIndex):
        raise ValueError(f"Ожидается PeriodIndex, получен {type(series.index)}")
    log = pd.Series(np.log1p(series.astype(float).to_numpy()), index=series.index)
    # min_periods=PROFILE_WINDOW (не частичное окно): медиана неполного набора
    # месяцев сезонально смещена и даёт ложные разрывы на краях ряда. Краям
    # не хватает окна — их уровень переносится от ближайшего полного окна;
    # разрыв в пределах PROFILE_WINDOW/2 от края ряда ненадёжен (см. docs/BREAKS.md).
    local_level = log.rolling(PROFILE_WINDOW, center=True, min_periods=PROFILE_WINDOW).median()
    local_level = local_level.bfill().ffill()
    norm = log - local_level
    if series.index.freqstr is not None and series.index.freqstr.upper().startswith("D"):
        keys = series.index.dayofweek
    else:
        keys = series.index.month
    grouped = pd.Series(norm.to_numpy(), index=keys)
    profile = grouped.groupby(level=0).median()
    profile[grouped.groupby(level=0).count() < 2] = 0.0
    # .get, а не [k]: после усечения ряда (truncate_series) месяц/день недели
    # может полностью отсутствовать в индексе — профиль для него 0.
    resid = log.to_numpy() - np.array([profile.get(k, 0.0) for k in keys], dtype=float)
    return pd.Series(resid, index=series.index, name="resid_log")


def _local_deviation(values: np.ndarray) -> np.ndarray:
    """Отклонение от центрированной скользящей медианы (окно
    SPIKE_LOCAL_WINDOW): локальный уровень, робастный к 1–2 месяцам шока и
    отслеживающий сдвиги уровня. Краям, где полного окна нет, уровень
    переносится от ближайшего полного окна."""
    s = pd.Series(values)
    local = s.rolling(SPIKE_LOCAL_WINDOW, center=True, min_periods=SPIKE_LOCAL_WINDOW).median()
    return (s - local.bfill().ffill()).to_numpy(dtype=float)


def _spike_runs(deviation: np.ndarray, *, z_threshold: float, min_abs: float) -> list[tuple[int, int]]:
    """Непрерывные серии месяцев-выбросов на локальном отклонении: пары
    (индекс первого месяца серии, длина серии). Шок ищется в обе стороны:
    |z| > порога И |отклонение| >= абсолютного пола (двухусловное правило,
    см. SPIKE_MIN_ABS_LOG) — отрицательный шок спроса равноправен
    положительному.
    """
    med = float(np.median(deviation))
    mad = float(np.median(np.abs(deviation - med)))
    if mad == 0.0:
        return []
    z = (deviation - med) / (1.4826 * mad)
    flagged = np.flatnonzero((np.abs(z) > z_threshold) & (np.abs(deviation - med) >= min_abs))
    runs: list[tuple[int, int]] = []
    for i in flagged:
        if runs and int(i) == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((int(i), 1))
    return runs


def detect_spike_months(
    series: pd.Series,
    *,
    z_threshold: float = SPIKE_Z_THRESHOLD,
    min_abs: float = SPIKE_MIN_ABS_LOG,
) -> list[tuple[int, int]]:
    """Непрерывные серии месяцев-выбросов на десезонализированном остатке.

    Возвращает пары (индекс первого месяца серии, длина серии). Робастная
    z-оценка локального отклонения (остаток минус его скользящая медиана):
    |z| > порога И амплитуда >= min_abs — шок ищется в обе стороны,
    отрицательный провал спроса равноправен положительному. Переходный шок
    длится 1–2 месяца — короче MIN_SEGMENT_SIZE, поэтому Pelt не может
    выделить его в сегмент; шоки ищутся здесь, до детекции уровней. Серия
    подряд идущих выбросов — один шок, а не несколько.
    """
    return _spike_runs(
        _local_deviation(deseasonalize_log(series).to_numpy(dtype=float)),
        z_threshold=z_threshold,
        min_abs=min_abs,
    )


def _detect_break_points_values(
    values: np.ndarray,
    *,
    penalty: float,
    min_size: int,
    algorithm: Literal["pelt", "binseg"],
) -> list[tuple[int, float]]:
    """Детекция разрывов уровня на готовом сигнале (уже без выбросов)."""
    if len(values) < 2 * min_size:
        return []
    algo_cls = Pelt if algorithm == "pelt" else Binseg
    algo = algo_cls(model="l2", min_size=min_size, jump=1)
    ends = algo.fit(values).predict(pen=penalty)
    # ends — концы сегментов, включая len(values); внутренние границы — разрывы.
    boundaries = [e for e in ends[:-1] if 0 < e < len(values)]
    result: list[tuple[int, float]] = []
    start = 0
    for b in boundaries:
        delta = float(values[b:].mean()) - float(values[start:b].mean())
        result.append((int(b), delta))
        start = b
    return result


def detect_break_points(
    series: pd.Series,
    *,
    penalty: float = PELT_PENALTY,
    min_size: int = MIN_SEGMENT_SIZE,
    algorithm: Literal["pelt", "binseg"] = "pelt",
) -> list[tuple[int, float]]:
    """Найти точки разрыва УРОВНЯ на десезонализированном лог-ряде.

    Месяцы-выбросы (см. detect_spike_months) перед детекцией заменяются
    соседними значениями: иначе одиночный шок расщепляет сегмент и маскирует
    сдвиг уровня. Возвращает пары (индекс первого периода нового сегмента,
    сдвиг уровня нового сегмента относительно предыдущего). Ряд короче
    2*min_size не несёт ни одного подтверждённого разрыва — возвращается
    пустой список, а не детекция на шумовых масштабах.
    """
    resid = deseasonalize_log(series)
    values = resid.to_numpy(dtype=float).copy()
    deviation = _local_deviation(values)
    for start, length in _spike_runs(deviation, z_threshold=SPIKE_Z_THRESHOLD, min_abs=SPIKE_MIN_ABS_LOG):
        _replace_spike(values, start, length)
    return _detect_break_points_values(values, penalty=penalty, min_size=min_size, algorithm=algorithm)


def _replace_spike(values: np.ndarray, start: int, length: int) -> None:
    """Заменить месяцы шока средним ближайших не-шоковых соседей. Если шок
    прилегает к краю ряда, доступен только один сосед; шок на ВСЁМ ряде не
    заменяется (заменять нечем) — детекция уровней на таком ряде не имеет
    смысла и вернёт пустой список по длине."""
    left = values[start - 1] if start - 1 >= 0 else None
    right = values[start + length] if start + length < len(values) else None
    if left is None and right is None:
        return
    values[start : start + length] = (left if right is None else right if left is None else (left + right) / 2)


def classify_breaks(
    series: pd.Series,
    *,
    measurement_seams: frozenset[pd.Period] = frozenset(),
    demand_events: frozenset[pd.Period] = frozenset(),
    penalty: float = PELT_PENALTY,
    min_size: int = MIN_SEGMENT_SIZE,
    algorithm: Literal["pelt", "binseg"] = "pelt",
) -> list[BreakPoint]:
    """Детекция + классификация разрывов.

    ``measurement_seams`` — периоды, в которых ряд склеен из разных окон
    сбора или сменил методику (результат #5/#6). ``demand_events`` — известные
    содержательные события (2020-03 COVID, 2022-03 уход брендов). Даты —
    гипотезы для классификации найденного, не хардкод точек: если детектор
    ничего не нашёл рядом, дата просто не попадает ни в один разрыв.

    Логика классификации (по убыванию приоритета):
    1. Возврат к до-разрывному уровню в течение REVERT_HORIZON месяцев —
       переходный шок спроса (``demand_transient``), независимо от швов.
    2. Устойчивый сдвиг на шве измерения — ``measurement``.
    3. Устойчивый сдвиг на известном событии — ``demand_persistent``.
    4. Устойчивый сдвиг без внешних оснований — ``unresolved``: из ряда
       спрос/измерение неразличимы, честнее сказать это явно.
    """
    resid = deseasonalize_log(series)
    values = resid.to_numpy(dtype=float)
    deviation = _local_deviation(values)
    spike_runs = _spike_runs(deviation, z_threshold=SPIKE_Z_THRESHOLD, min_abs=SPIKE_MIN_ABS_LOG)
    spike_idx = {i for start, length in spike_runs for i in range(start, start + length)}
    breaks: list[BreakPoint] = []

    # Переходные шоки (1–2 месяца): отдельный детектор, Pelt их не видит.
    for start, length in spike_runs:
        at_series_start = start == 0
        # at — последний период ДО разрыва; у шока с началом ряда до-разрывного
        # периода нет, тогда at — первый период нового режима (помечено в
        # evidence), иначе негативный индекс молча дал бы конец ряда.
        at = series.index[0] if at_series_start else series.index[start - 1]
        peak = int(start + np.argmax(np.abs(deviation[start : start + length])))
        breaks.append(
            BreakPoint(
                at=at,
                delta_log=float(deviation[peak]),
                kind="demand_transient",
                evidence={
                    "spike": True,
                    "at_series_start": at_series_start,
                    "spike_months": length,
                    "spike_z_threshold": SPIKE_Z_THRESHOLD,
                    "spike_min_abs_log": SPIKE_MIN_ABS_LOG,
                    "reverted": True,
                    "revert_horizon": REVERT_HORIZON,
                    "seam": False,
                    "known_demand_event": False,
                    "series_len": len(series),
                },
            )
        )

    # Разрывы уровня: Pelt/Binseg на сигнале со заменёнными выбросами
    # (десезонализация считается один раз, сюда сигнал уже готов).
    cleaned = values.copy()
    for start, length in spike_runs:
        _replace_spike(cleaned, start, length)
    for idx, delta in _detect_break_points_values(cleaned, penalty=penalty, min_size=min_size, algorithm=algorithm):
        at = series.index[idx - 1]
        kind: BreakKind
        reverted = False
        for t in range(idx, min(idx + REVERT_HORIZON, len(values))):
            if t in spike_idx:
                continue
            if abs(values[t] - values[:idx].mean()) <= REVERT_TOLERANCE * abs(delta):
                reverted = True
                break
        next_period = series.index[idx] if idx < len(series) else None
        if reverted:
            kind = "demand_transient"
        elif next_period in measurement_seams:
            kind = "measurement"
        elif next_period in demand_events or at in demand_events:
            kind = "demand_persistent"
        else:
            kind = "unresolved"
        breaks.append(
            BreakPoint(
                at=at,
                delta_log=delta,
                kind=kind,
                evidence={
                    "spike": False,
                    "reverted": reverted,
                    "revert_horizon": REVERT_HORIZON,
                    "seam": next_period in measurement_seams,
                    "known_demand_event": bool(next_period in demand_events or at in demand_events),
                    "series_len": len(series),
                },
            )
        )
    return sorted(breaks, key=lambda b: b.at.ordinal)


def decide_handling(breaks: list[BreakPoint]) -> list[BreakDecision]:
    """Решение по обработке для каждого разрыва (маппинг зафиксирован в
    docs/BREAKS.md до прогона; пустого «решения нет» не существует)."""
    return [
        BreakDecision(br=b, handling=_KIND_HANDLING[b.kind], rationale=_HANDLING_RATIONALE[_KIND_HANDLING[b.kind]])
        for b in breaks
    ]


def known_dates_coverage(breaks: list[BreakPoint], known: frozenset[pd.Period] | list[pd.Period]) -> KnownDatesCoverage:
    """Какие известные даты детектор подтвердил, какие нет.

    Сопоставление с допуском KNOWN_DATE_TOLERANCE: месячная гранулярность и
    min_size сдвигают границу сегмента на соседний период, совпадение «в один
    месяц» засчитывается.
    """
    covered: list[pd.Period] = []
    missed: list[pd.Period] = []
    for d in known:
        ordinal = d.ordinal
        hit = any(abs(b.at.ordinal - ordinal) <= KNOWN_DATE_TOLERANCE for b in breaks)
        (covered if hit else missed).append(d)
    return KnownDatesCoverage(covered=tuple(covered), missed=tuple(missed))


def truncate_series(series: pd.Series, decision: BreakDecision) -> pd.Series:
    """Обрезка ряда по разрыву измерения: остаётся только пост-разрывная часть.

    Лог-линейная методика до разрыва и после несопоставима, моделировать её
    одним рядом нельзя — это единственный разрыв, для которого обрезка
    корректна.
    """
    if decision.handling != "truncate":
        raise ValueError(f"Обрезка определена только для handling='truncate', получено {decision.handling!r}")
    return series[series.index > decision.br.at]


def add_dummy(series: pd.Series, decision: BreakDecision) -> pd.DataFrame:
    """Дамми-переменная «после разрыва» для устойчивого сдвига спроса:
    ряд сохраняется целиком, модель получает индикатор смены режима."""
    if decision.handling != "dummy":
        raise ValueError(f"Дамми определён только для handling='dummy', получено {decision.handling!r}")
    after = series.index > decision.br.at
    return pd.DataFrame({"count": series.astype(float), "post_break": after.astype(int)}, index=series.index)


def build_breaks_report(
    series: pd.Series,
    decisions: list[BreakDecision],
    coverage: KnownDatesCoverage | None = None,
) -> dict[str, object]:
    """Машиночитаемый отчёт для docs/BREAKS.md: точки, типы, решения."""
    return {
        "series_range": [str(series.index.min()), str(series.index.max())],
        "series_len": len(series),
        "breaks": [
            {
                "at": str(d.br.at),
                "delta_log": round(d.br.delta_log, 4),
                "kind": d.br.kind,
                "handling": d.handling,
                "evidence": d.br.evidence,
                "rationale": d.rationale,
            }
            for d in decisions
        ],
        "known_dates_covered": [str(p) for p in (coverage.covered if coverage else [])],
        "known_dates_missed": [str(p) for p in (coverage.missed if coverage else [])],
    }

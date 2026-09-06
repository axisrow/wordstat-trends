"""Краткосрочные тренды на дневных данных (issue #22): детекция всплесков
как отклонения от недельного профиля.

Границы задачи, зафиксированные в issue #22:

- Дневной ряд Вордстата — окно 60 дней, отдельный ряд от месячного
  (дневная грануляция wordstat-cli, axisrow/wordstat-cli#6).
- «Всплеск» здесь — не устойчивый рост из #12, а разовое отклонение от
  недельного профиля (будни против выходных). Отсев разовых всплесков в
  долгосрочном ядре #22 не ослабляет и не трогает — это разные задачи.
- Срок жизни сигнала — дни: каждый результат несёт дату замера
  (``DailySnapshot.measured_through``) явно.
- Раздел витрины (пункт 3 issue #22) — вне этого модуля: витрины (#14)
  ещё не существует, смешивать краткосрочные сигналы с долгосрочными
  нельзя в принципе, поэтому ядро выдаёт структуру данных, а не рендер.

Метод — намеренно без ETS и других моделей: недельный профиль оценивается
эмпирически, доля дня недели внутри его собственной недели устраняет тренд
на горизонте недели, а сравнение с другими неделями окна даёт ожидаемый
уровень для каждого дня (leave-one-week-out). Порог всплеска
SPIKE_RATIO_THRESHOLD предрегистрирован в docs/SHORT_TERM.md до прогона
на фикстурах.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Порог всплеска: факт / ожидание по недельному профилю >= 1.5. Предрегистрирован
# (см. docs/SHORT_TERM.md), не подбирается по результату на фикстурах.
SPIKE_RATIO_THRESHOLD = 1.5

# Профиль и ожидания считаются только по ПОЛНЫМ неделям (пн→вс, все 7 дней):
# у обрезанной крайней недели окна доли дней искажены самой обрезкой, а
# вклад однодневного всплеска в короткую неделю завышен. Окно 60 дней всегда
# даёт как минимум 7 полных недель.
MIN_DAYS_PER_WEEK = 7

WEEKDAY_NAMES = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


@dataclass(frozen=True)
class DailySnapshot:
    """Дневной ряд + честная пометка срока жизни.

    measured_through — дата ПОСЛЕДНЕЙ строки данных, а не конечная дата из
    заголовка графика: у Вордстата граница диапазона в заголовке на день
    больше последней строки (эксклюзивная граница, см. docs/DATA.md и
    scripts/experiment_vector_d.py). Заголовок источником дат не является.
    """

    phrase: str
    series: pd.Series  # freq="D", name="count"
    measured_through: pd.Period  # freq="D", последняя строка данных


@dataclass(frozen=True)
class Spike:
    """Один всплеск: день, отношение факт/ожидание, вклад дня в свою неделю."""

    date: pd.Period
    ratio: float  # факт / ожидание (leave-one-week-out по этому же дню недели)
    weekday: str


def _read_raw_csv_lines(path: Path) -> list[str]:
    """Сырой CSV Вордстата (docs/DATA.md): UTF-8 с BOM, переводы строк только
    CR -> splitlines(). Общая механика с scripts/experiment_vector_d.py."""
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"Пустой файл: {path}")
    return lines


def _parse_daily_period(period: str) -> pd.Period:
    """"22.06.2026" -> Period(freq="D"). Дневная выгрузка не использует
    текстовое название месяца — отдельный парсер, не смешивать с месячным."""
    day, month, year = period.strip().split(".")
    return pd.Period(year=int(year), month=int(month), day=int(day), freq="D")


def load_daily_snapshot(path: Path) -> DailySnapshot:
    """Разобрать сырой дневной CSV Вордстата (dynamics) в DailySnapshot.

    Особенности формата (замерены на живом прогоне wordstat-cli, docs/DATA.md
    и MVP #23): заголовок первой колонки «Дата», период DD.MM.YYYY, разделитель
    «;», пробел как разделитель тысяч. Фраза берётся из 4-го поля заголовка
    графика («Динамика частотности запросов «...», по дням, ...»).
    """
    lines = _read_raw_csv_lines(path)

    header = lines[0].split(";")
    if header[0] != "Дата":
        raise ValueError(f"Неожиданный заголовок в {path}: {header}. Дневная выгрузка начинается с «Дата».")

    chart_title = header[3] if len(header) > 3 else ""
    phrase = _extract_phrase(chart_title)

    records: list[tuple[pd.Period, float]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(";")
        period = _parse_daily_period(fields[0])
        count = float(fields[1].replace(" ", "").replace(" ", ""))
        records.append((period, count))

    periods, counts = zip(*records, strict=True)
    series = pd.Series(counts, index=pd.PeriodIndex(periods, freq="D"), name="count")
    series = series.sort_index()

    if not series.index.is_unique:
        raise ValueError(f"Дубликаты дат в {path}: {series.index[series.index.duplicated()].tolist()}")
    expected = pd.period_range(start=series.index.min(), end=series.index.max(), freq="D")
    if not series.index.equals(expected):
        missing = expected.difference(series.index)
        raise ValueError(f"Пропуски дней (непрерывность нарушена) в {path}: {missing.tolist()}")

    return DailySnapshot(phrase=phrase, series=series, measured_through=series.index[-1])


def _extract_phrase(chart_title: str) -> str:
    """«Динамика частотности запросов «новогодние подарки», по дням, ...» ->
    «новогодние подарки». Пустая строка, если кавычек нет — фраза не извлечена,
    а не выдумана."""
    if "«" in chart_title and "»" in chart_title:
        return chart_title.split("«", 1)[1].split("»", 1)[0]
    return ""


def weekday_profile(series: pd.Series) -> np.ndarray:
    """Эмпирический недельный профиль: доля каждого дня недели в СВОЕЙ
    неделе, усреднённая по полным неделям окна, нормированная к среднему 1.
    Индекс 0 = понедельник.

    Нормировка внутри недели (а не средним по всему окну) устраняет тренд
    на горизонте недели: растущий ряд даёт одинаковые доли в ранней и поздней
    неделе. Неполные крайние недели (< MIN_DAYS_PER_WEEK дней) исключаются.
    """
    shares = _weekday_shares(series)
    profile = shares / shares.mean()
    return profile


def _weekday_shares(series: pd.Series) -> np.ndarray:
    shares_sum = np.zeros(7)
    shares_count = np.zeros(7)
    for _, week in _weeks(series):
        if len(week) < MIN_DAYS_PER_WEEK:
            continue
        week_mean = float(week.mean())
        if week_mean == 0:
            continue
        for period, value in week.items():
            weekday = period.to_timestamp().weekday()
            shares_sum[weekday] += float(value) / week_mean
            shares_count[weekday] += 1
    if (shares_count == 0).any():
        raise ValueError(
            "Недельный профиль не оценивается: у какого-то дня недели нет ни одной полной недели в окне"
        )
    return shares_sum / shares_count


def _week_share_vectors(series: pd.Series) -> list[np.ndarray]:
    """Вектор долей (7 значений, индекс = weekday) для каждой ПОЛНОЙ недели.
    Профиль детектора строится усреднением этих векторов; их раздельное
    хранение нужно для leave-one-week-out — исключения оцениваемой недели."""
    vectors = []
    for _, week in _weeks(series):
        if len(week) < MIN_DAYS_PER_WEEK:
            continue
        week_mean = float(week.mean())
        if week_mean == 0:
            continue
        vector = np.zeros(7)
        for period, value in week.items():
            vector[period.to_timestamp().weekday()] = float(value) / week_mean
        vectors.append(vector)
    return vectors


def _weeks(series: pd.Series):
    """Группировка по календарным неделям (пн→вс) без требования что-либо
    достраивать: крайние неполные недели отдаются как есть, их отсеивает
    вызывающий код по длине."""
    df = pd.DataFrame({"period": series.index, "value": series.to_numpy()})
    df["week"] = df["period"].apply(lambda p: p.to_timestamp().to_period("W-SUN"))
    for _, week in df.groupby("week"):
        yield week["period"], pd.Series(week["value"].to_numpy(), index=week["period"])


def detect_spikes(series: pd.Series, threshold: float = SPIKE_RATIO_THRESHOLD) -> list[Spike]:
    """Всплески как отклонения от недельного профиля (issue #22, пункт 2).

    Ожидание для дня — leave-one-week-out в полном смысле: доля его дня
    недели в профиле, посчитанном по ДРУГИМ полным неделям (собственная
    неделя исключена и из профиля, и из уровня), × среднее его собственной
    недели без самого этого дня. Двойное исключение собственной недели не
    даёт однодневному всплеску ни поднять своё «ожидание» через уровень
    недели, ни сдвинуть недельный профиль через завышенную долю дня; тренд
    при этом учтён — уровень берётся соседними днями, а не всем окном.

    ratio = факт / ожидание; всплеск — ratio >= threshold. Дни из неполных
    крайних недель (< MIN_DAYS_PER_WEEK) не оцениваются: без полной недели
    доля дня в ней искажена обрезкой окна. Нужно минимум 2 полные недели —
    иначе оцениваемой неделе не с чем сравниваться.
    """
    full_weeks = [week for _, week in _weeks(series) if len(week) == MIN_DAYS_PER_WEEK and float(week.mean()) > 0]
    if len(full_weeks) < 2:
        raise ValueError(
            f"Детекция всплесков требует минимум 2 полные недели (пн→вс), в окне {len(full_weeks)}"
        )
    week_shares = _week_share_vectors(series)

    spikes: list[Spike] = []
    for week_index, week in enumerate(full_weeks):
        # Профиль без собственной недели: всплеск не должен сдвигать профиль,
        # против которого сам же и оценивается.
        other_shares = [s for i, s in enumerate(week_shares) if i != week_index]
        profile = np.mean(other_shares, axis=0)
        # Уровень недели без оцениваемого дня: (сумма - день) / (длина - 1).
        week_sum = float(week.sum())
        n = len(week)
        for period, value in week.items():
            weekday = period.to_timestamp().weekday()
            week_level_loo = (week_sum - float(value)) / (n - 1)
            expected = week_level_loo * profile[weekday]
            if expected <= 0:
                continue
            ratio = float(value) / expected
            if ratio >= threshold:
                spikes.append(Spike(date=period, ratio=ratio, weekday=WEEKDAY_NAMES[weekday]))
    return spikes


def short_term_report(snapshot: DailySnapshot, last_days: int = 7, threshold: float = SPIKE_RATIO_THRESHOLD) -> dict:
    """Отчёт по краткосрочным всплескам для витрины (issue #22, пункты 2 и 4).

    Смотрит только последние last_days дней: более старые дни окна — контекст
    профиля, не сигнал. Дата замера — всегда в ответе, это не опция
    (пункт 4: «показывать дату замера явно»).
    """
    series = snapshot.series
    spikes = detect_spikes(series, threshold=threshold)
    recent = [s for s in spikes if s.date > snapshot.measured_through - last_days]
    recent.sort(key=lambda s: s.ratio, reverse=True)
    return {
        "phrase": snapshot.phrase,
        "measured_through": str(snapshot.measured_through),  # срок жизни сигнала — дни
        "threshold": threshold,
        "spikes_recent": [
            {"date": str(s.date), "weekday": s.weekday, "ratio": round(s.ratio, 3)} for s in recent
        ],
        "n_days_window": len(series),
    }

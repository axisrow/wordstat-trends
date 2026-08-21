"""Нарезка и проверка дневного обучающего окна для `sp=7`.

Смысл всех проверок здесь один: **несмещённый недельный профиль**. Если
какой-то день недели встречается чаще других, сезонный коэффициент этого дня
считается по большему числу наблюдений, и профиль перекошен ещё до того, как
модель что-либо оценит. Условие несмещённости — длина окна, кратная семи;
тогда каждый день недели встречается ровно `длина / 7` раз.

Отдельно: сырую длину выгрузки проверять бессмысленно и вредно. Вордстат на
окне `--date-from 2026-06-23 --date-to 2026-08-20` отдаёт 58 строк, а не 59:
метка конца в заголовке графика **эксклюзивна** и на день опережает
последнюю строку данных. Поэтому здесь проверяется обучающее окно (первые
`TRAIN_DAYS` строк), а holdout — это то, что осталось, сколько бы его ни
оказалось.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SP = 7
"""Недельный период на дневном ряде."""

TRAIN_DAYS = 56
"""Обучающее окно: 56 дней = 8 полных недельных циклов."""

TRUNCATIONS = (7, 14)
"""Укорочения для проверки устойчивости: 49 и 42 дня = 7 и 6 циклов."""

_WEEKDAY_NAMES = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


@dataclass(frozen=True)
class DailyWindow:
    """Обучающее окно и то, что осталось за ним."""

    train: pd.Series
    holdout: pd.Series

    @property
    def weekday_counts(self) -> list[int]:
        return weekday_counts(self.train)


def weekday_counts(series: pd.Series) -> list[int]:
    """Сколько раз встречается каждый день недели, понедельник первым."""
    counts = pd.Series(series.index.dayofweek).value_counts()
    return [int(counts.get(day, 0)) for day in range(SP)]


def check_continuous(series: pd.Series) -> None:
    """Даты идут подряд по одному дню, без дыр и повторов.

    Важно не только само по себе: детренд и позиционная арифметика считают,
    что соседние строки — соседние дни. Дыра сдвигает весь хвост молча.
    """
    index = series.index
    if index.has_duplicates:
        raise ValueError("В ряде повторяются даты")
    if not index.is_monotonic_increasing:
        raise ValueError("Даты не упорядочены по возрастанию")
    expected = pd.date_range(index[0], index[-1], freq="D")
    if len(index) != len(expected) or not index.equals(expected):
        missing = expected.difference(index)
        raise ValueError(f"В ряде пропущены дни: {[str(d.date()) for d in missing[:5]]}")


def make_window(series: pd.Series, train_days: int = TRAIN_DAYS) -> DailyWindow:
    """Режет ряд на обучающее окно и holdout, проверяя всё, что может молча сломаться.

    Обучающее окно — **первые** `train_days` строк: дневное окно Вордстата
    упирается в границу 60 дней слева, и именно начало ряда гарантированно
    внутри неё.
    """
    if len(series) < train_days:
        raise ValueError(
            f"В ряде {len(series)} дней, для обучающего окна нужно {train_days}. "
            "Достраивать недостающие дни нельзя."
        )

    check_continuous(series)
    train = series.iloc[:train_days]
    holdout = series.iloc[train_days:]

    if train_days % SP:
        raise ValueError(f"Длина окна {train_days} не кратна {SP}: профиль будет смещён")

    counts = weekday_counts(train)
    if len(set(counts)) != 1:
        pairs = ", ".join(f"{name} {count}" for name, count in zip(_WEEKDAY_NAMES, counts, strict=True))
        raise ValueError(f"Дни недели представлены неодинаково ({pairs}) — профиль смещён")

    return DailyWindow(train=train, holdout=holdout)

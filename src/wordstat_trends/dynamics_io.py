"""Разбор сырого CSV представления «Динамика» Яндекс Вордстат.

Формат зафиксирован замером в `docs/DATA.md`, а не взят из справки:

* UTF-8 **с BOM** (`EF BB BF`);
* переводы строк — **только CR** (`\\r`), без `\\n`: наивное чтение по `\\n`
  видит файл одной строкой;
* разделитель полей — `;`, полей четыре, четвёртое всегда пустое
  (каждая строка данных заканчивается на `;`);
* разделитель тысяч в «Число запросов» — обычный ASCII-пробел (`997 977`);
* десятичный разделитель в долях — запятая (`0,0115`);
* период — **зависит от грануляции**, и это единственное, что различается
  между месячной и дневной выгрузкой:

  - месячная: русское название месяца в именительном падеже плюс год
    (`август 2024`), а не дата в каком-либо машинном формате;
  - дневная: дата с точками (`22.06.2026`), то есть `DD.MM.YYYY`.

  Поэтому месячный разбор периода **нельзя переиспользовать вслепую** —
  для каждой грануляции своя функция.

Отдельная ловушка дневной выгрузки: четвёртое поле заголовка (заголовок
графика) содержит конец окна, и эта метка **эксклюзивна** — она на день
опережает последнюю строку данных. Даты берём только из строк.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

MONTHS: dict[str, int] = {
    name: number
    for number, name in enumerate(
        (
            "январь февраль март апрель май июнь "
            "июль август сентябрь октябрь ноябрь декабрь"
        ).split(),
        start=1,
    )
}

_PERIOD_COLUMN = "Период"
_QUERIES_COLUMN = "Число запросов"


def _parse_period(raw: str) -> pd.Period:
    """`"август 2024"` → `Period('2024-08', 'M')`."""
    parts = raw.split()
    if len(parts) != 2:
        raise ValueError(f"Не период Вордстата: {raw!r}")
    month_name, year = parts
    try:
        month = MONTHS[month_name.lower()]
    except KeyError:
        raise ValueError(f"Неизвестный русский месяц: {month_name!r}") from None
    return pd.Period(freq="M", year=int(year), month=month)


def _parse_count(raw: str) -> int:
    """`"997 977"` → `997977`.

    Замер зафиксировал обычный ASCII-пробел, но неразрывный убираем тоже:
    замер был один, а цена ошибки — падение разбора на живых данных.
    """
    return int(raw.replace(" ", "").replace("\xa0", ""))


def _parse_day(raw: str) -> pd.Timestamp:
    """`"22.06.2026"` → `Timestamp('2026-06-22')`.

    Дневной формат периода — не тот же, что месячный: там русский месяц
    словом, здесь `DD.MM.YYYY`. `format` задан явно, чтобы `06.07.2026`
    никогда не разобралось как 7 июня по локали или эвристике pandas.
    """
    try:
        return pd.to_datetime(raw, format="%d.%m.%Y")
    except ValueError:
        raise ValueError(f"Не дневной период Вордстата (ожидался DD.MM.YYYY): {raw!r}") from None


def load_dynamics(path: str | Path, granularity: str = "monthly") -> pd.Series:
    """Читает `dynamics`-CSV в ряд числа запросов.

    `granularity="monthly"` — индекс `PeriodIndex` месячной частоты,
    `granularity="daily"` — `DatetimeIndex` по дням. Ряд сортируется по
    периоду: порядок строк в файле не считается заслуживающим доверия.
    """
    if granularity not in _PERIOD_PARSERS:
        raise ValueError(f"Неизвестная грануляция: {granularity!r}")
    parse_period = _PERIOD_PARSERS[granularity]
    text = Path(path).read_text(encoding="utf-8-sig")
    # `splitlines()` режет по CR, LF и CRLF — единственное чтение, которое
    # переживает CR-only переводы строк Вордстата.
    rows = list(csv.reader(text.splitlines(), delimiter=";"))
    if not rows:
        raise ValueError(f"Пустой файл: {path}")

    header = rows[0]
    # Проверять только вторую колонку недостаточно: у представления `regions`
    # она называется точно так же («Регион;Число запросов;…», см. docs/DATA.md).
    # Различает представления именно первая колонка, и без неё regions-файл
    # проходил бы проверку, падая позже с невнятным «Не период Вордстата».
    if len(header) < 2 or header[0].strip() != _PERIOD_COLUMN or header[1].strip() != _QUERIES_COLUMN:
        raise ValueError(f"Неожиданный заголовок dynamics: {header[:2]}")

    records: list[tuple[pd.Period, int]] = []
    for row in rows[1:]:
        # Четвёртое поле пустое во всех строках данных — это не битая строка.
        if not row or not row[0].strip():
            continue
        if len(row) < 2:
            # Иначе строка без разделителя падала бы голым IndexError — в этом
            # модуле все остальные поломки формата сообщают о себе ValueError.
            raise ValueError(f"В строке меньше двух полей: {row!r}")
        records.append((parse_period(row[0].strip()), _parse_count(row[1])))

    if not records:
        raise ValueError(f"В файле нет строк данных: {path}")

    records.sort()
    periods, counts = zip(*records, strict=True)
    index = (
        pd.PeriodIndex(periods, freq="M")
        if granularity == "monthly"
        else pd.DatetimeIndex(periods)
    )
    if index.has_duplicates:
        raise ValueError("В файле повторяются периоды")
    return pd.Series(list(counts), index=index, name="queries")


_PERIOD_PARSERS = {"monthly": _parse_period, "daily": _parse_day}

"""Окно сбора динамики: пятилетний дефолт вместо платформенных двух лет (issue #5).

Замер (#2, ``docs/DATA.md``) показал: дефолт интерфейса — ровно 24 месяца.
24 точек мало для ``sp=12`` — это лишь два сезонных цикла, на которых
сезонность неотличима от единичного артефакта. Осознанный дефолт проекта —
окно в 5 лет (60 месячных точек, ~5 циклов); полный ряд с января 2018
добирается склейкой окон (issue #6).

Рамки интерфейса Вордстата (справка + замер #2, перепроверять не нужно):
нижняя граница истории — январь 2018; окно одного запроса — от 3 до 60
календарных месяцев; дневная грануляция — только последние 60 дней.
Валидация выполняется здесь, до обращения к браузеру.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from wordstat_trends.collect.errors import WindowRangeError

#: Самая ранняя доступная история Вордстата (замер #2, подтверждено вручную).
HISTORY_FLOOR = date(2018, 1, 1)

#: Минимальное окно при месячной детализации — три календарных месяца
#: (интерфейс сам расширяет диапазон с сообщением «Диапазон дат был изменён»).
MIN_WINDOW_MONTHS = 3

#: Максимальное окно одного запроса — «до пяти лет» по справке Яндекса.
MAX_WINDOW_MONTHS = 60

#: Грануляция по умолчанию: месяцы — единственная грануляция, для которой
#: Вордстат честно отдаёт произвольное заданное окно (weekly игнорирует
#: диапазон дат — см. ``wordstat collect --help``).
DEFAULT_GRANULARITY = "monthly"


def _shift_months(d: date, months: int) -> date:
    """Первый день месяца, отстоящего от месяца ``d`` на ``months`` месяцев."""
    total = d.year * 12 + (d.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def _last_day_of_previous_month(d: date) -> date:
    return _shift_months(d, 0) - timedelta(days=1)


@dataclass(frozen=True)
class DynamicsWindow:
    """Окно запроса динамики: диапазон дат и грануляция."""

    date_from: date
    date_to: date
    granularity: str = DEFAULT_GRANULARITY

    def validate(self, *, today: date | None = None) -> None:
        """Отвергнуть недопустимое окно до браузера: рамки интерфейса Вордстата."""
        if self.date_from > self.date_to:
            raise WindowRangeError(f"дата начала {self.date_from} позже даты конца {self.date_to}")
        if self.date_from < HISTORY_FLOOR:
            raise WindowRangeError(
                f"дата начала {self.date_from} раньше нижней границы истории Вордстата ({HISTORY_FLOOR})"
            )
        if today is not None and self.date_to > today:
            raise WindowRangeError(f"дата конца {self.date_to} в будущем относительно {today}")

        months = (self.date_to.year - self.date_from.year) * 12 + (self.date_to.month - self.date_from.month) + 1
        if months < MIN_WINDOW_MONTHS:
            raise WindowRangeError(
                f"окно {self.date_from}—{self.date_to} покрывает {months} мес., "
                f"минимум интерфейса — {MIN_WINDOW_MONTHS}"
            )
        if months > MAX_WINDOW_MONTHS:
            raise WindowRangeError(
                f"окно {self.date_from}—{self.date_to} покрывает {months} мес., "
                f"максимум интерфейса — {MAX_WINDOW_MONTHS}"
            )

    def cli_flags(self) -> list[str]:
        """Аргументы ``wordstat collect``, задающие это окно."""
        return [
            "--granularity",
            self.granularity,
            "--date-from",
            self.date_from.isoformat(),
            "--date-to",
            self.date_to.isoformat(),
        ]


def default_window(*, today: date | None = None) -> DynamicsWindow:
    """Пятилетнее окно полных прошедших месяцев.

    ``date_to`` — последний день предыдущего (завершённого) месяца: неполный
    текущий месяц в ряд не попадает. ``date_from`` — первый день месяца
    60 месяцев назад, итого MAX_WINDOW_MONTHS месячных точек на окно.
    """
    today = today or date.today()
    window = DynamicsWindow(
        date_from=_shift_months(today, -MAX_WINDOW_MONTHS),
        date_to=_last_day_of_previous_month(today),
    )
    window.validate(today=today)
    return window


__all__ = [
    "DEFAULT_GRANULARITY",
    "HISTORY_FLOOR",
    "MAX_WINDOW_MONTHS",
    "MIN_WINDOW_MONTHS",
    "DynamicsWindow",
    "default_window",
]

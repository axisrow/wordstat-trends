"""Доменные ошибки сбора и склейки динамики (issue #6).

Каждая ошибка несёт данные, а не только текст: расхождение — период и оба
значения, дыра — оба края разрыва. Текст строится из них, чтобы исключение
было читаемым в логах и pytest-выводе без потери машиночитаемых полей.
"""

from __future__ import annotations


class CollectError(Exception):
    """Базовая ошибка сбора/склейки динамики."""


class WindowRangeError(CollectError):
    """Недопустимое окно запроса (issue #5).

    Отвергается до обращения к браузеру: дата раньше нижней границы истории,
    окно короче трёх или длиннее пяти лет (жёсткие рамки интерфейса Вордстата).
    """


class WindowMismatchError(CollectError):
    """Расхождение значений в зоне нахлёста двух окон.

    Усреднение запрещено сознательно (issue #6): разные числа за один месяц
    в разных окнах — сигнал о смене методики Вордстата, а не шум.
    """

    def __init__(self, period: tuple[int, int], left: int, right: int):
        self.period = period
        self.left = left
        self.right = right
        y, m = period
        super().__init__(
            f"расхождение в нахлёсте за период {m:02d}.{y}: {left} != {right} — "
            "усреднение запрещено, проверьте смену методики Вордстата"
        )


class SeriesGapError(CollectError):
    """Непрерывность итогового ряда нарушена: дыра, непокрытый край окна или
    отсутствие нахлёста между окнами.

    Дыра ломает ``sp=12`` молча, поэтому проверка обязательна (issue #6).
    """

    def __init__(self, before: tuple[int, int] | None, after: tuple[int, int] | None, detail: str = ""):
        self.before = before
        self.after = after
        parts = []
        if before is not None:
            parts.append(f"последний непрерывный период {before[1]:02d}.{before[0]}")
        if after is not None:
            parts.append(f"первый после разрыва {after[1]:02d}.{after[0]}")
        if detail:
            parts.append(detail)
        super().__init__("дыра в ряду: " + "; ".join(parts))

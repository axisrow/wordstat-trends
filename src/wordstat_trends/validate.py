"""Контрактные тесты на данные (issue #94, Фаза 7.1).

Инварианты формата — факты замера #2, зафиксированные в ``docs/DATA.md`` и
``docs/DATASET.md``. Отличие от unit-тестов ``test_loader.py``/``test_stitch.py``
(те проверяют поведение *кода* на синтетике): здесь проверяются сами *данные* —
любая выгрузка Вордстата, живая или фикстура, — против контракта. Нарушение
означает смену поведения Вордстата или сборщика, а не баг этого модуля:

- UTF-8 с BOM (``EF BB BF``), переводы строк CR-only (``\r`` без ``\n``);
- заголовок — ровно 4 поля через ``;``: «Период», «Число запросов», «Доля от
  всех запросов, %», четвёртый (длинный заголовок графика) непуст; во всех
  строках данных четвёртое поле пустое;
- число запросов — разделитель тысяч ASCII-пробел ``0x20`` (не U+00A0), доля —
  десятичная запятая; период — русский месяц текстом плюс год;
- диапазоны: ``queries ≥ 0``, ``0 ≤ доля ≤ 100``;
- непрерывность месячного ряда: без дыр и дублей (дыра ломает ``sp=12``);
- ``top_popular``/``top_related`` пусты в замере #2 (баг сборщика): пустота —
  известное состояние, непустой файл — сигнал пересмотреть контракт.

Все проверки собираются в список нарушений, а не падают на первом: отчёт
должен показать всю картину сразу. Разбор периодов переиспользует
:mod:`wordstat_trends.collect.stitch`.
"""

from __future__ import annotations

import csv
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from wordstat_trends.collect.stitch import month_key, parse_period
from wordstat_trends.loader import ManifestError, load_manifest

#: Заголовок динамики, дословно из замера #2 (docs/DATA.md).
HEADER = ["Период", "Число запросов", "Доля от всех запросов, %"]

#: Представления, пустые в замере #2 из-за бага сборщика (docs/DATA.md).
KNOWN_EMPTY_VIEWS = ("top_popular", "top_related")

#: Неразрывный пробел — как «выглядит» разделитель тысяч, скопированный из
#: веб-UI; в реальном CSV замера #2 — обычный ASCII-пробел (побайтово, DATA.md).
NBSP = " "

_Queries = re.compile(r"^\d{1,3}( \d{3})*$")
_Share = re.compile(r"^\d+(,\d+)?$")

@dataclass(frozen=True)
class Violation:
    """Одно нарушение контракта: именованная проверка + место + описание."""

    check: str
    message: str
    source: str = ""
    severity: str = "error"  # error роняет CLI; info — зафиксированный факт

    def __str__(self) -> str:
        place = f"{self.source}: " if self.source else ""
        return f"[{self.severity}] {place}{self.check}: {self.message}"

@dataclass
class RunReport:
    """Отчёт по одному run-каталогу (или одиночному CSV)."""

    source: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    def ok(self) -> bool:
        return not self.errors

    def print(self, stream=None) -> None:
        stream = stream or sys.stdout
        status = "OK" if self.ok() else f"{len(self.errors)} нарушений"
        print(f"{self.source}: {status}", file=stream)
        for v in self.violations:
            print(f"  {v}", file=stream)


def validate_dynamics_csv(data: bytes, *, source: str = "") -> list[Violation]:
    """Полный контракт одного файла dynamics.csv — от байтов до непрерывности."""
    out: list[Violation] = []

    def add(check: str, message: str, severity: str = "error") -> None:
        out.append(Violation(check, message, source, severity))

    # --- Байтовый уровень: BOM и CR-only ------------------------------------
    if not data.startswith(b"\xef\xbb\xbf"):
        add("bom", "нет UTF-8 BOM (EF BB BF) в начале файла")
    if b"\n" in data:
        add("line-endings", "переводы строк не CR-only: в файле есть \\n (ожидался только \\r)")

    # utf-8-sig снимает BOM при decode — дополнительная зачистка не нужна
    # (в loader она защита для прямых вызовов с сырым текстом; здесь вход — bytes).
    text = data.decode("utf-8-sig", errors="replace")

    # --- Заголовок: ровно 4 поля, имена дословно из замера #2 -----------------
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=";"))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        add("schema", "файл не содержит строк")
        return out
    header = [c.strip() for c in rows[0]]
    if len(header) != 4:
        add("schema", f"в заголовке {len(header)} полей, ожидалось 4: {header!r}")
    else:
        for expected, actual in zip(HEADER, header[:3]):
            if actual.lower() != expected.lower():
                add("schema", f"колонка {expected!r} называется {actual!r} — формат сменился")
        if not header[3].strip():
            add("schema", "четвёртое поле заголовка (длинный заголовок графика) пусто")
    if len(rows) < 2:
        add("schema", "нет строк данных (только заголовок)")
        return out

    # --- Строки данных: структура, форматы, диапазоны -------------------------
    seen: list[tuple[int, int, int]] = []  # (month_key, year, month)
    for row in rows[1:]:
        if len(row) != 4:
            add("schema", f"в строке {len(row)} полей, ожидалось 4 (с пустым четвёртым): {row!r}")
            continue
        if row[3].strip():
            add("schema", f"четвёртое поле не пусто: {row[3]!r}")

        try:
            year, month = parse_period(row[0])
        except ValueError:
            add("period-format", f"период не «<русский месяц> <год>»: {row[0]!r}")
            continue

        raw_queries = row[1].strip()
        nbsp = NBSP in raw_queries
        if nbsp:
            # NBSP уже зафиксирован — «не разбирается как целое» поверх не нужно.
            add("thousands-separator", f"разделитель тысяч U+00A0 вместо ASCII-пробела: {row[1]!r}")
        elif not _Queries.fullmatch(raw_queries):
            add("number-format", f"число запросов не «тысячи через ASCII-пробел»: {row[1]!r}")
        if not nbsp and not raw_queries.replace(" ", "").isdigit():
            add("range", f"число запросов не разбирается как неотрицательное целое: {row[1]!r}")

        raw_share = row[2].strip()
        if "." in raw_share:
            add("decimal-separator", f"десятичный разделитель точки вместо запятой: {row[2]!r}")
        elif not _Share.fullmatch(raw_share):
            add("number-format", f"доля не «число с десятичной запятой»: {row[2]!r}")
        else:
            share = float(raw_share.replace(",", "."))
            if not 0 <= share <= 100:
                add("range", f"доля вне [0, 100]: {row[2]!r}")

        seen.append((month_key(year, month), year, month))

    # --- Непрерывность: соседние месяцы отличаются ровно на один ---------------
    seen.sort()
    for prev, cur in zip(seen, seen[1:]):
        if cur[0] == prev[0]:
            add("continuity", f"дублирующийся период {cur[1]:04d}-{cur[2]:02d}")
        elif cur[0] - prev[0] != 1:
            add(
                "continuity",
                f"дыра в ряду: {prev[1]:04d}-{prev[2]:02d} → {cur[1]:04d}-{cur[2]:02d} "
                f"(пропущено {cur[0] - prev[0] - 1} мес.)",
            )
    return out


def validate_run(run_directory: Path) -> RunReport:
    """Контракт одного run-каталога: dynamics.csv + manifest + известные пустоты."""
    run = Path(run_directory)
    report = RunReport(source=str(run))

    if not (run / "manifest.json").is_file():
        report.violations.append(
            Violation("manifest", "нет manifest.json — это run-каталог?", str(run))
        )
    else:
        try:
            load_manifest(run)
        except ManifestError as exc:
            report.violations.append(Violation("manifest", str(exc), str(run)))

    dynamics = run / "dynamics.csv"
    if not dynamics.is_file():
        report.violations.append(
            Violation("schema", "нет dynamics.csv (нужен прогон с --keep-raw)", str(run))
        )
    else:
        report.violations.extend(
            validate_dynamics_csv(dynamics.read_bytes(), source=str(dynamics))
        )

    # Пустые топы — известный баг сборщика (docs/DATA.md): пустота не ошибка,
    # а вот появившиеся данные — сигнал, что фикс в wordstat-cli доехал и
    # контракт надо расширять, а не молча проглотить новый формат.
    for view in KNOWN_EMPTY_VIEWS:
        path = run / f"{view}.csv"
        if not path.is_file():
            continue
        rows = [
            r
            for r in csv.reader(
                io.StringIO(path.read_text(encoding="utf-8-sig")),
                delimiter=";",
            )
            if any(c.strip() for c in r)
        ]
        if len(rows) > 1:
            report.violations.append(
                Violation(
                    "known-empty-view",
                    f"{view} не пуст ({len(rows) - 1} строк данных) — баг wordstat-cli закрыт, "
                    "расширить контракт на это представление",
                    str(path),
                    severity="info",
                )
            )
    return report


def validate_dataset(paths) -> list[RunReport]:
    """Контракт по набору run-каталогов или одиночных dynamics.csv."""
    reports = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            reports.append(RunReport(str(p), [Violation("io", "путь не существует", str(p))]))
        elif (p / "dynamics.csv").is_file() or (p / "manifest.json").is_file():
            reports.append(validate_run(p))
        elif p.is_file():
            reports.append(RunReport(str(p), validate_dynamics_csv(p.read_bytes(), source=str(p))))
        else:
            # Существующий каталог без dynamics.csv и manifest.json — не run-каталог;
            # отдельное сообщение, чтобы не путать с несуществующим путём.
            report = RunReport(str(p), [Violation("io", "не run-каталог и не dynamics.csv", str(p))])
            reports.append(report)
    return reports

__all__ = [
    "HEADER",
    "KNOWN_EMPTY_VIEWS",
    "NBSP",
    "RunReport",
    "Violation",
    "validate_dynamics_csv",
    "validate_dataset",
    "validate_run",
]

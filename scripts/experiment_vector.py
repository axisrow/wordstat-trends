#!/usr/bin/env python3
"""Run the monthly (sp=12) demand-vector MVP from issue #23.

The supplied Wordstat fixtures deliberately use their production CSV format:
UTF-8 BOM, CR-only newlines, Russian month names and semicolon delimiters.
This script parses that format before fitting the model so that the experiment
can be rerun against replacement exports without a separate conversion step.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sktime.forecasting.ets import AutoETS

MONTHS = {
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

MONTH_LABELS = tuple(MONTHS)
MODEL_DESCRIPTION = "AutoETS(auto=True, sp=12)"


@dataclass(frozen=True)
class DemandSeries:
    phrase: str
    values: pd.Series


@dataclass(frozen=True)
class DemandVector:
    specification: str
    level: float
    trend: float
    seasonal: dict[str, float]
    aic: float


def read_dynamics_csv(path: Path) -> DemandSeries:
    """Read a real Wordstat monthly dynamics export into a PeriodIndex series."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: expected a header and at least one row")

    phrase_match = re.search(r"«(.+?)»", lines[0])
    if phrase_match is None:
        raise ValueError(f"{path}: phrase is missing from the dynamics header")

    observations: list[tuple[pd.Period, int]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split(";")
        if len(fields) < 2:
            raise ValueError(f"{path}:{line_number}: expected semicolon-delimited fields")
        period_text, count_text = fields[:2]
        try:
            month_name, year_text = period_text.split()
            period = pd.Period(year=int(year_text), month=MONTHS[month_name], freq="M")
            count = int(count_text.replace(" ", ""))
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid Wordstat observation") from error
        observations.append((period, count))

    values = pd.Series(
        [count for _, count in observations],
        index=pd.PeriodIndex([period for period, _ in observations], freq="M"),
        dtype="int64",
    )
    if values.index.has_duplicates or not values.index.is_monotonic_increasing:
        raise ValueError(f"{path}: periods must be unique and ordered")
    expected_index = pd.period_range(values.index[0], values.index[-1], freq="M")
    if not values.index.equals(expected_index):
        raise ValueError(f"{path}: monthly periods must be continuous")
    return DemandSeries(phrase=phrase_match.group(1), values=values)


def fit_monthly_vector(values: pd.Series) -> DemandVector:
    """Fit AutoETS and expose the selected model's final state vector."""
    forecaster = AutoETS(auto=True, sp=12)
    fitted = forecaster.fit(values)._fitted_forecaster
    final_state = fitted.states.iloc[-1]
    seasonal_by_month = {}
    if "seasonal" in fitted.states:
        seasonal_by_month = {
            MONTH_LABELS[period.month - 1]: float(seasonal)
            for period, seasonal in fitted.states["seasonal"].iloc[-12:].items()
        }
    return DemandVector(
        specification=(
            f"error={fitted.model.error}, trend={fitted.model.trend}, "
            f"seasonal={fitted.model.seasonal}, damped_trend={fitted.model.damped_trend}"
        ),
        level=float(final_state["level"]),
        trend=float(final_state.get("trend", 0.0)),
        seasonal=seasonal_by_month,
        aic=float(fitted.aic),
    )


def format_number(value: float) -> str:
    return f"{value:.2f}"


def compare_vectors(full: DemandVector, shortened: DemandVector) -> tuple[float, float, float]:
    """Return level, trend, and seasonal-profile drift relative to the full vector."""
    level_drift = abs(shortened.level - full.level) / max(abs(full.level), 1.0)
    trend_drift = abs(shortened.trend - full.trend) / max(abs(full.level), 1.0)
    seasonal_rmse = (
        sum((shortened.seasonal[month] - full.seasonal[month]) ** 2 for month in MONTH_LABELS) / len(MONTH_LABELS)
    ) ** 0.5
    seasonal_drift = seasonal_rmse / max(max(full.seasonal.values()) - min(full.seasonal.values()), 1.0)
    return level_drift, trend_drift, seasonal_drift


def render_report(series_by_file: list[tuple[Path, DemandSeries]]) -> str:
    """Render the committed, timestamp-free experiment report."""
    rows: list[tuple[Path, str, int, int, str, DemandVector | None]] = []
    for path, series in series_by_file:
        for removed_months in (0, 3, 6):
            values = series.values if removed_months == 0 else series.values.iloc[:-removed_months]
            try:
                vector = fit_monthly_vector(values)
                status = "измерен"
            except ValueError as error:
                vector = None
                status = f"не измерен: `{error}`"
            rows.append((path, series.phrase, len(values), removed_months, status, vector))

    report = [
        "# MVP: вектор параметров ETS как представление спроса",
        "",
        "Связано с #23. Это MVP только для месячного профиля `sp=12`; дневной `sp=7` не проверялся.",
        "",
        "## Что измерено",
        "",
        "Три реальные месячные выгрузки из `tests/fixtures/` прочитаны с UTF-8 BOM, CR-only переводами строк, "
        "русскими месяцами и `;`.",
        f"Для каждого ряда задана одна и та же модель: `{MODEL_DESCRIPTION}`.",
        "Вектор — финальное состояние выбранной модели: уровень, тренд и, если AutoETS выбрал сезонность, "
        "сезонные коэффициенты по календарным месяцам.",
        "",
        "## Проверка устойчивости — стоп-условие",
        "",
        "| Фраза | Точек | Удалено последних месяцев | Результат |",
        "| --- | ---: | ---: | --- |",
    ]
    for _, phrase, points, removed, status, _ in rows:
        report.append(f"| {phrase} | {points} | {removed} | {status} |")

    failed_rows = [row for row in rows if row[-1] is None]
    report.extend(["", "## Векторы", ""])
    for _, phrase, points, removed, _, vector in rows:
        if vector is None:
            continue
        seasonal = ", ".join(f"{month}: {format_number(vector.seasonal[month])}" for month in MONTH_LABELS)
        report.extend(
            [
                f"### {phrase} ({points} точек, удалено: {removed})",
                "",
                f"- выбранная спецификация: `{vector.specification}`",
                f"- AIC: {format_number(vector.aic)}",
                f"- уровень: {format_number(vector.level)}",
                f"- тренд: {format_number(vector.trend)}",
                f"- сезонные коэффициенты: {seasonal or 'модель не выбрала сезонный компонент'}",
                "",
            ]
        )

    comparisons: list[tuple[str, int, float, float, float]] = []
    for path, series in series_by_file:
        vectors = {removed: vector for row_path, _, _, removed, _, vector in rows if row_path == path}
        full = vectors[0]
        assert full is not None
        for removed in (3, 6):
            shortened = vectors[removed]
            if shortened is not None:
                comparisons.append((series.phrase, removed, *compare_vectors(full, shortened)))

    if comparisons:
        report.extend(
            [
                "## Численное сравнение с полным вектором",
                "",
                "Сезонный дрейф — RMSE 12 сезонных коэффициентов, делённый на размах сезонного профиля полного ряда. "
                "Дрейф тренда нормирован на уровень полного ряда.",
                "",
                "| Фраза | Удалено | Дрейф уровня | Дрейф тренда | Дрейф сезонного профиля |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for phrase, removed, level, trend, seasonal in comparisons:
            report.append(f"| {phrase} | {removed} | {level:.4f} | {trend:.4f} | {seasonal:.4f} |")
        report.append("")

    report.extend(
        [
            "## Вывод",
            "",
        ]
    )
    if failed_rows:
        report.extend(
            [
                "**Стоп-условие MVP сработал: гипотеза на этих фикстурах не подтверждена.** "
                "Это не измеренный провал коэффициентов, а более ранняя и жёсткая граница: при 24 точках "
                "нельзя выполнить обязательную проверку устойчивости после усечения на 3 или 6 месяцев. "
                "Следовательно, нельзя честно переходить к проверке осмысленности кластеров "
                "или к продуктовой реализации.",
                "",
                "## Чего проверить не удалось",
                "",
                "- Устойчивость вектора при усечении `−3` и `−6` месяцев — отсутствуют два полных годовых цикла.",
                "- Различение сезонного и плоского профилей — не запускалось, так как оно следует после устойчивости.",
                "- Недельный профиль `sp=7` — нет дневного ряда; он вне объёма этого MVP.",
                "",
                "Для повторного MVP нужен склеенный ряд из #6 не менее чем из 30 точек: после удаления шести "
                "месяцев должны оставаться как минимум 24 точки; более длинный ряд нужен для содержательной "
                "оценки сезонности.",
            ]
        )
    else:
        report.extend(
            [
                "Усечённые ряды обучились и численное сравнение выполнено. Интерпретация величины дрейфа "
                "и решение о переходе к кластеризации остаются отдельным шагом; этот скрипт не объявляет "
                "гипотезу успешной автоматически.",
            ]
        )
    report.extend(
        [
            "",
            "## Воспроизведение",
            "",
            "```bash",
            "uv run python scripts/experiment_vector.py --output docs/EXPERIMENT_VECTOR.md",
            "uv run --extra dev pytest tests/test_experiment_vector.py",
            "```",
            "",
        ]
    )
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fixtures = sorted(args.fixtures_dir.glob("dynamics_*.csv"))
    if not fixtures:
        raise SystemExit(f"No dynamics fixtures found in {args.fixtures_dir}")
    report = render_report([(path, read_dynamics_csv(path)) for path in fixtures])
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()

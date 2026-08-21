#!/usr/bin/env python3
"""MVP #23, variant B: test whether monthly AutoETS vectors are stable.

The script deliberately evaluates stability before semantic usefulness.  If a
truncated window cannot be fitted, or if its vector drifts beyond the declared
limits, the experiment stops without comparing phrase profiles.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sktime.forecasting.ets import AutoETS

SEASONAL_PERIOD = 12
TRUNCATIONS = (0, 3, 6)

# Declared before looking at the comparison: a fixed representation should not
# change by more than 20% in any component group when 3--6 recent observations
# are withheld.  These limits are not reached on the current 24-point fixtures,
# because seasonal AutoETS cannot initialize on the truncated series.
MAX_RELATIVE_LEVEL_DRIFT = 0.20
MAX_RELATIVE_TREND_DRIFT = 0.20
MAX_RELATIVE_SEASONAL_L2_DRIFT = 0.20

RUSSIAN_MONTHS = {
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

FIXTURES = {
    "high_freq": "dynamics_high_freq.csv",
    "mid_freq": "dynamics_mid_freq.csv",
    "seasonal": "dynamics_seasonal.csv",
}


@dataclass(frozen=True)
class ETSVector:
    """Interpretable final ETS state in calendar-month order."""

    level: float
    trend: float
    seasonal: list[float]
    error_type: str
    trend_type: str | None
    seasonal_type: str
    aicc: float


def parse_period(value: str) -> pd.Period:
    """Parse a Wordstat period such as ``август 2024``."""

    parts = value.strip().lower().split()
    if len(parts) != 2 or parts[0] not in RUSSIAN_MONTHS:
        raise ValueError(f"Unsupported Wordstat monthly period: {value!r}")
    try:
        year = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Unsupported Wordstat monthly period: {value!r}") from exc
    return pd.Period(year=year, month=RUSSIAN_MONTHS[parts[0]], freq="M")


def load_wordstat_monthly(path: Path) -> pd.Series:
    """Load a real Wordstat CSV, including BOM and CR-only line endings."""

    text = path.read_text(encoding="utf-8-sig")
    # splitlines() is intentional: the real exports use bare CR, not LF/CRLF.
    reader = csv.DictReader(io.StringIO("\n".join(text.splitlines())), delimiter=";")
    rows = list(reader)
    required = {"Период", "Число запросов"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"Missing required Wordstat columns in {path}")
    if not rows:
        raise ValueError(f"No observations in {path}")

    periods = [parse_period(row["Период"]) for row in rows]
    values = [float(row["Число запросов"].replace(" ", "")) for row in rows]
    series = pd.Series(values, index=pd.PeriodIndex(periods), name="requests", dtype=float)
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError(f"Periods must be unique and ascending in {path}")
    expected = pd.period_range(series.index[0], series.index[-1], freq="M")
    if not series.index.equals(expected):
        raise ValueError(f"Monthly series has gaps in {path}")
    return series


def fit_vector(series: pd.Series) -> ETSVector:
    """Fit automatic ETS(sp=12) and return its final state vector."""

    forecaster = AutoETS(auto=True, sp=SEASONAL_PERIOD, n_jobs=1, random_state=0)
    forecaster.fit(series)
    result = forecaster._fitted_forecaster
    model = result.model

    seasonal_state = pd.Series(result.season, index=series.index, dtype=float)
    seasonal_by_month = [
        float(seasonal_state[seasonal_state.index.month == month].iloc[-1]) for month in range(1, SEASONAL_PERIOD + 1)
    ]
    trend = float(result.slope.iloc[-1]) if model.trend is not None else 0.0
    return ETSVector(
        level=float(result.level.iloc[-1]),
        trend=trend,
        seasonal=seasonal_by_month,
        error_type=str(model.error),
        trend_type=None if model.trend is None else str(model.trend),
        seasonal_type=str(model.seasonal),
        aicc=float(result.aicc),
    )


def relative_drift(candidate: ETSVector, full: ETSVector) -> dict[str, float]:
    """Compare an aligned candidate vector with the full-window vector."""

    epsilon = np.finfo(float).eps
    level = abs(candidate.level - full.level) / max(abs(full.level), epsilon)
    trend_scale = max(abs(full.trend), abs(full.level) / SEASONAL_PERIOD, epsilon)
    trend = abs(candidate.trend - full.trend) / trend_scale
    full_seasonal = np.asarray(full.seasonal)
    candidate_seasonal = np.asarray(candidate.seasonal)
    seasonal = np.linalg.norm(candidate_seasonal - full_seasonal) / max(np.linalg.norm(full_seasonal), epsilon)
    return {
        "level_relative": float(level),
        "trend_relative": float(trend),
        "seasonal_relative_l2": float(seasonal),
    }


def run_experiment(fixtures_dir: Path) -> dict[str, Any]:
    """Run the ordered MVP, applying the stability stop-condition."""

    fits: dict[str, dict[str, Any]] = {}
    for profile, filename in FIXTURES.items():
        series = load_wordstat_monthly(fixtures_dir / filename)
        windows: dict[str, Any] = {}
        for removed in TRUNCATIONS:
            window = series if removed == 0 else series.iloc[:-removed]
            key = f"minus_{removed}"
            try:
                vector = fit_vector(window)
                windows[key] = {
                    "status": "fit",
                    "observations": len(window),
                    "start": str(window.index[0]),
                    "end": str(window.index[-1]),
                    "vector": asdict(vector),
                }
            except ValueError as exc:
                windows[key] = {
                    "status": "fit_failed",
                    "observations": len(window),
                    "start": str(window.index[0]),
                    "end": str(window.index[-1]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        fits[profile] = windows

    comparisons: dict[str, Any] = {}
    all_comparable = True
    all_within_limits = True
    for profile, windows in fits.items():
        full_data = windows["minus_0"]
        profile_comparisons: dict[str, Any] = {}
        if full_data["status"] != "fit":
            all_comparable = False
        else:
            full = ETSVector(**full_data["vector"])
            for removed in TRUNCATIONS[1:]:
                key = f"minus_{removed}"
                candidate_data = windows[key]
                if candidate_data["status"] != "fit":
                    all_comparable = False
                    profile_comparisons[key] = {"status": "not_comparable"}
                    continue
                drift = relative_drift(ETSVector(**candidate_data["vector"]), full)
                within_limits = (
                    drift["level_relative"] <= MAX_RELATIVE_LEVEL_DRIFT
                    and drift["trend_relative"] <= MAX_RELATIVE_TREND_DRIFT
                    and drift["seasonal_relative_l2"] <= MAX_RELATIVE_SEASONAL_L2_DRIFT
                )
                all_within_limits &= within_limits
                profile_comparisons[key] = {
                    "status": "compared",
                    "within_limits": within_limits,
                    **drift,
                }
        comparisons[profile] = profile_comparisons

    stability_passed = all_comparable and all_within_limits
    if stability_passed:
        status = "stability_passed"
        semantic_check = "eligible_but_not_implemented_in_this_stability_script"
    elif not all_comparable:
        status = "inconclusive_insufficient_history"
        semantic_check = "not_run_due_to_stop_condition"
    else:
        status = "stability_failed"
        semantic_check = "not_run_due_to_stop_condition"

    return {
        "configuration": {
            "model": "sktime.forecasting.ets.AutoETS",
            "auto": True,
            "sp": SEASONAL_PERIOD,
            "random_state": 0,
            "truncations_months": list(TRUNCATIONS[1:]),
            "limits": {
                "level_relative": MAX_RELATIVE_LEVEL_DRIFT,
                "trend_relative": MAX_RELATIVE_TREND_DRIFT,
                "seasonal_relative_l2": MAX_RELATIVE_SEASONAL_L2_DRIFT,
            },
        },
        "fits": fits,
        "comparisons": comparisons,
        "stability_passed": stability_passed,
        "status": status,
        "semantic_check": semantic_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tests" / "fixtures",
    )
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.fixtures_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

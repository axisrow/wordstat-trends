"""MVP #23, variant B: stability of daily AutoETS(sp=7) vectors."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEASONAL_PERIOD = 7
TRAIN_OBSERVATIONS = 56
TRUNCATIONS = (0, 7, 14)
EXPECTED_START = pd.Timestamp("2026-06-23")
EXPECTED_TRAIN_END = pd.Timestamp("2026-08-17")
EXPECTED_DATA_END = pd.Timestamp("2026-08-19")
WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# Preregistered before daily fixtures were available. See the report.
MIN_AMPLITUDE = 0.05
MIN_PROFILE_CORRELATION = 0.90
MAX_RELATIVE_RMSE = 0.25
MAX_SHIFT_TO_AMPLITUDE = 0.35

DAILY_FIXTURES = {
    "high_freq": "dynamics_daily_high_freq_mvp.csv",
    "mid_freq": "dynamics_daily_mid_freq_mvp.csv",
    "seasonal": "dynamics_daily_seasonal_mvp.csv",
}


@dataclass(frozen=True)
class ETSVector:
    backend: str
    level: float
    trend: float
    profile: list[float]
    error_type: str
    trend_type: str | None
    seasonal_type: str | None
    method: str
    aic: float
    aicc: float
    bic: float

    @property
    def seasonal_present(self) -> bool:
        return self.seasonal_type not in (None, "N")


@dataclass
class FittedVector:
    vector: ETSVector
    forecast: np.ndarray
    fitted_model: Any


def parse_daily_period(value: str) -> pd.Timestamp:
    """Parse the measured daily Wordstat period format, DD.MM.YYYY."""

    try:
        return pd.Timestamp(datetime.strptime(value.strip(), "%d.%m.%Y"))
    except ValueError as exc:
        raise ValueError(f"Unsupported Wordstat daily period: {value!r}") from exc


def parse_request_count(value: str) -> float:
    """Parse spaces as thousands separators and comma as decimal separator."""

    normalized = value.strip().replace(" ", "").replace("\N{NO-BREAK SPACE}", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported Wordstat request count: {value!r}") from exc


def load_wordstat_daily(path: Path) -> pd.Series:
    """Load a BOM/CR-only Wordstat daily dynamics CSV without using its title."""

    text = path.read_text(encoding="utf-8-sig")
    # The real export uses bare CR. splitlines(), unlike split("\n"), handles it.
    reader = csv.DictReader(io.StringIO("\n".join(text.splitlines())), delimiter=";")
    rows = list(reader)
    if reader.fieldnames is None or "Число запросов" not in reader.fieldnames:
        raise ValueError(f"Missing required Wordstat columns in {path}")
    date_fields = [field for field in ("Дата", "Период") if field in reader.fieldnames]
    if len(date_fields) != 1:
        raise ValueError(f"Expected exactly one daily date column in {path}, got {date_fields}")
    date_field = date_fields[0]
    if not rows:
        raise ValueError(f"No observations in {path}")

    dates = pd.DatetimeIndex([parse_daily_period(row[date_field]) for row in rows])
    values = [parse_request_count(row["Число запросов"]) for row in rows]
    series = pd.Series(values, index=dates, name="requests", dtype=float)
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError(f"Dates must be unique and ascending in {path}")
    expected = pd.date_range(series.index[0], series.index[-1], freq="D")
    if not series.index.equals(expected):
        raise ValueError(f"Daily series has gaps in {path}")
    # Preserve the proven daily frequency for statsmodels instead of making it
    # infer the same fact again for every fit.
    series.index = expected
    return series


def split_experiment_window(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Validate row dates and return the first 56 days plus the remaining holdout."""

    if len(series) < TRAIN_OBSERVATIONS:
        raise ValueError(f"Expected at least {TRAIN_OBSERVATIONS} real daily rows, got {len(series)}")
    if series.index[0] != EXPECTED_START or series.index[-1] != EXPECTED_DATA_END:
        raise ValueError(
            f"Expected row bounds {EXPECTED_START.date()}..{EXPECTED_DATA_END.date()}, "
            f"got {series.index[0].date()}..{series.index[-1].date()}"
        )
    train = series.iloc[:TRAIN_OBSERVATIONS]
    holdout = series.iloc[TRAIN_OBSERVATIONS:]
    if train.index[-1] != EXPECTED_TRAIN_END:
        raise ValueError(f"Expected training end {EXPECTED_TRAIN_END.date()}, got {train.index[-1].date()}")
    weekday_counts = np.bincount(train.index.dayofweek, minlength=SEASONAL_PERIOD).tolist()
    if weekday_counts != [8] * SEASONAL_PERIOD:
        raise ValueError(f"Unbalanced training weekdays: {weekday_counts}")
    return train, holdout


def _normalize_seasonal(raw: np.ndarray, seasonal_type: str | None, level: float) -> np.ndarray:
    if seasonal_type in (None, "N"):
        return np.ones(SEASONAL_PERIOD)
    raw = np.asarray(raw, dtype=float)
    if seasonal_type in ("mul", "M"):
        mean = float(np.mean(raw))
        if np.isclose(mean, 0.0):
            raise ValueError("Cannot normalize a zero-mean multiplicative seasonal profile")
        return raw / mean
    if seasonal_type in ("add", "A"):
        if np.isclose(level, 0.0):
            raise ValueError("Cannot normalize an additive seasonal profile at zero level")
        return 1.0 + (raw - np.mean(raw)) / abs(level)
    raise ValueError(f"Unsupported ETS seasonal type: {seasonal_type!r}")


def _calendar_profile(states: pd.Series, seasonal_type: str | None, level: float) -> np.ndarray:
    if seasonal_type in (None, "N"):
        return np.ones(SEASONAL_PERIOD)
    latest = np.array([states[states.index.dayofweek == day].iloc[-1] for day in range(SEASONAL_PERIOD)])
    return _normalize_seasonal(latest, seasonal_type, level)


def fit_sktime(series: pd.Series, forecast_horizon: int = 3) -> FittedVector:
    """Fit the required automatic sktime AutoETS without fixing its specification."""

    from sktime.forecasting.ets import AutoETS as SktimeAutoETS

    forecaster = SktimeAutoETS(auto=True, sp=SEASONAL_PERIOD, n_jobs=1, random_state=0)
    forecaster.fit(series)
    result = forecaster._fitted_forecaster
    model = result.model
    level = float(result.level.iloc[-1])
    trend = float(result.slope.iloc[-1]) if model.trend is not None else 0.0
    seasonal_type = None if model.seasonal is None else str(model.seasonal)
    seasonal_states = (
        pd.Series(np.ones(len(series)), index=series.index)
        if model.seasonal is None
        else pd.Series(result.season, index=series.index, dtype=float)
    )
    profile = _calendar_profile(seasonal_states, seasonal_type, level)
    vector = ETSVector(
        backend="sktime",
        level=level,
        trend=trend,
        profile=profile.tolist(),
        error_type=str(model.error),
        trend_type=None if model.trend is None else str(model.trend),
        seasonal_type=seasonal_type,
        method=f"ETS({model.error},{model.trend},{model.seasonal})",
        aic=float(result.aic),
        aicc=float(result.aicc),
        bic=float(result.bic),
    )
    forecast = np.asarray(forecaster.predict(fh=list(range(1, forecast_horizon + 1))), dtype=float)
    return FittedVector(vector=vector, forecast=forecast, fitted_model=forecaster)


def _statsforecast_profile(model: dict[str, Any], last_date: pd.Timestamp) -> np.ndarray:
    from statsforecast.ets import forecast_ets

    components = str(model["components"])
    trend_type = components[1]
    seasonal_type = components[2]
    if seasonal_type == "N":
        return np.ones(SEASONAL_PERIOD)

    seasonal_forecast = np.asarray(forecast_ets(model, h=SEASONAL_PERIOD)["mean"], dtype=float)
    baseline_model = copy.deepcopy(model)
    states = np.array(baseline_model["states"], copy=True)
    offset = 1 + int(trend_type != "N")
    states[-1, offset : offset + SEASONAL_PERIOD] = 1.0 if seasonal_type == "M" else 0.0
    baseline_model["states"] = states
    baseline_forecast = np.asarray(forecast_ets(baseline_model, h=SEASONAL_PERIOD)["mean"], dtype=float)
    level = float(model["states"][-1, 0])
    if seasonal_type == "M":
        raw = seasonal_forecast / baseline_forecast
    else:
        raw = seasonal_forecast - baseline_forecast
    ordered_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=SEASONAL_PERIOD, freq="D")
    by_weekday = np.empty(SEASONAL_PERIOD)
    for value, date in zip(raw, ordered_dates, strict=True):
        by_weekday[date.dayofweek] = value
    return _normalize_seasonal(by_weekday, seasonal_type, level)


def fit_statsforecast(series: pd.Series, forecast_horizon: int = 3) -> FittedVector:
    """Fit statsforecast AutoETS(ZZZ) and expose the independently chosen vector."""

    from statsforecast.models import AutoETS as StatsForecastAutoETS

    forecaster = StatsForecastAutoETS(season_length=SEASONAL_PERIOD)
    forecaster.fit(series.to_numpy())
    model = forecaster.model_
    components = str(model["components"])
    level = float(model["states"][-1, 0])
    trend = float(model["states"][-1, 1]) if components[1] != "N" else 0.0
    profile = _statsforecast_profile(model, series.index[-1])
    vector = ETSVector(
        backend="statsforecast",
        level=level,
        trend=trend,
        profile=profile.tolist(),
        error_type=components[0],
        trend_type=None if components[1] == "N" else components[1],
        seasonal_type=None if components[2] == "N" else components[2],
        method=str(model["method"]),
        aic=float(model["aic"]),
        aicc=float(model["aicc"]),
        bic=float(model["bic"]),
    )
    forecast = np.asarray(forecaster.predict(h=forecast_horizon)["mean"], dtype=float)
    return FittedVector(vector=vector, forecast=forecast, fitted_model=forecaster)


def statsforecast_ic_contrast(series: pd.Series) -> dict[str, Any]:
    """Diagnostic AICc contrast; it does not replace the primary automatic ZZZ fit."""

    from statsforecast.models import AutoETS as StatsForecastAutoETS

    candidates: list[dict[str, Any]] = []
    for specification in ("ZZN", "ZZA", "ZZM"):
        try:
            fitted = StatsForecastAutoETS(season_length=SEASONAL_PERIOD, model=specification).fit(series.to_numpy())
            model = fitted.model_
            candidates.append(
                {
                    "diagnostic_constraint": specification,
                    "method": str(model["method"]),
                    "components": str(model["components"]),
                    "aic": float(model["aic"]),
                    "aicc": float(model["aicc"]),
                    "bic": float(model["bic"]),
                }
            )
        except Exception as exc:  # diagnostic candidates may be inadmissible
            candidates.append({"diagnostic_constraint": specification, "error": f"{type(exc).__name__}: {exc}"})

    fitted_candidates = [candidate for candidate in candidates if "aicc" in candidate]
    nonseasonal = [candidate for candidate in fitted_candidates if candidate["components"][2] == "N"]
    seasonal = [candidate for candidate in fitted_candidates if candidate["components"][2] != "N"]
    best_nonseasonal = min(nonseasonal, key=lambda item: item["aicc"], default=None)
    best_seasonal = min(seasonal, key=lambda item: item["aicc"], default=None)
    delta = None
    if best_nonseasonal is not None and best_seasonal is not None:
        delta = float(best_seasonal["aicc"] - best_nonseasonal["aicc"])
    return {
        "candidates": candidates,
        "best_nonseasonal": best_nonseasonal,
        "best_seasonal": best_seasonal,
        "seasonal_minus_nonseasonal_aicc": delta,
    }


def stability_metrics(candidate: ETSVector, full: ETSVector) -> dict[str, Any]:
    """Apply the preregistered stability gate to aligned Mon..Sun profiles."""

    full_profile = np.asarray(full.profile)
    candidate_profile = np.asarray(candidate.profile)
    amplitude = float(np.ptp(full_profile))
    if amplitude < MIN_AMPLITUDE:
        return {
            "status": "flat_full_profile",
            "stable": False,
            "amplitude": amplitude,
            "correlation": None,
            "relative_rmse": None,
            "max_shift_to_amplitude": None,
            "seasonal_presence_changed": candidate.seasonal_present != full.seasonal_present,
        }

    delta = candidate_profile - full_profile
    correlation = float(np.corrcoef(full_profile, candidate_profile)[0, 1])
    relative_rmse = float(np.sqrt(np.mean(delta**2)) / np.sqrt(np.mean((full_profile - 1.0) ** 2)))
    max_shift = float(np.max(np.abs(delta)) / amplitude)
    presence_changed = candidate.seasonal_present != full.seasonal_present
    stable = (
        not presence_changed
        and np.isfinite(correlation)
        and correlation >= MIN_PROFILE_CORRELATION
        and relative_rmse <= MAX_RELATIVE_RMSE
        and max_shift <= MAX_SHIFT_TO_AMPLITUDE
    )
    return {
        "status": "compared",
        "stable": bool(stable),
        "amplitude": amplitude,
        "correlation": correlation if np.isfinite(correlation) else None,
        "relative_rmse": relative_rmse,
        "max_shift_to_amplitude": max_shift,
        "seasonal_presence_changed": presence_changed,
    }


def backend_comparison(sktime: ETSVector, statsforecast: ETSVector) -> dict[str, Any]:
    """Measure backend agreement descriptively; this is deliberately not a gate."""

    result: dict[str, Any] = {
        "sktime_seasonal": sktime.seasonal_present,
        "statsforecast_seasonal": statsforecast.seasonal_present,
        "same_seasonal_decision": sktime.seasonal_present == statsforecast.seasonal_present,
        "status": "not_comparable",
        "correlation": None,
        "max_shift_to_sktime_amplitude": None,
    }
    amplitude = float(np.ptp(sktime.profile))
    if sktime.seasonal_present and statsforecast.seasonal_present and amplitude >= MIN_AMPLITUDE:
        left = np.asarray(sktime.profile)
        right = np.asarray(statsforecast.profile)
        correlation = float(np.corrcoef(left, right)[0, 1])
        result.update(
            status="compared",
            correlation=correlation if np.isfinite(correlation) else None,
            max_shift_to_sktime_amplitude=float(np.max(np.abs(right - left)) / amplitude),
        )
    return result


def holdout_metrics(actual: pd.Series, forecast: np.ndarray) -> dict[str, Any]:
    """Descriptive three-day control; no pass/fail threshold was preregistered."""

    observed = actual.to_numpy(dtype=float)
    errors = np.asarray(forecast, dtype=float) - observed
    smape = np.mean(2 * np.abs(errors) / (np.abs(observed) + np.abs(forecast)))
    return {
        "dates": [str(date.date()) for date in actual.index],
        "actual": observed.tolist(),
        "forecast": np.asarray(forecast, dtype=float).tolist(),
        "mae": float(np.mean(np.abs(errors))),
        "smape": float(smape),
    }


def run_experiment(fixtures_dir: Path) -> dict[str, Any]:
    """Run the daily experiment in its preregistered order."""

    phrases: dict[str, Any] = {}
    for profile_name, filename in DAILY_FIXTURES.items():
        raw = load_wordstat_daily(fixtures_dir / filename)
        train, holdout = split_experiment_window(raw)
        fits: dict[str, Any] = {}
        fitted_objects: dict[str, dict[str, FittedVector]] = {}
        for removed in TRUNCATIONS:
            window = train if removed == 0 else train.iloc[:-removed]
            key = f"minus_{removed}"
            sktime_fit = fit_sktime(window, forecast_horizon=len(holdout))
            statsforecast_fit = fit_statsforecast(window, forecast_horizon=len(holdout))
            fitted_objects[key] = {"sktime": sktime_fit, "statsforecast": statsforecast_fit}
            fits[key] = {
                "observations": len(window),
                "start": str(window.index[0].date()),
                "end": str(window.index[-1].date()),
                "sktime": asdict(sktime_fit.vector),
                "statsforecast": asdict(statsforecast_fit.vector),
                "statsforecast_ic_contrast": statsforecast_ic_contrast(window),
                "backend_comparison": backend_comparison(sktime_fit.vector, statsforecast_fit.vector),
            }

        stability: dict[str, Any] = {}
        for backend in ("sktime", "statsforecast"):
            full_vector = fitted_objects["minus_0"][backend].vector
            stability[backend] = {
                key: stability_metrics(fitted_objects[key][backend].vector, full_vector)
                for key in ("minus_7", "minus_14")
            }
        stable = all(metric["stable"] for backend in stability.values() for metric in backend.values())
        semantic: dict[str, Any]
        if stable:
            semantic = {}
            for backend in ("sktime", "statsforecast"):
                vector = fitted_objects["minus_0"][backend].vector
                profile = np.asarray(vector.profile)
                semantic[backend] = {
                    "weekday_mean": float(np.mean(profile[:5])),
                    "weekend_mean": float(np.mean(profile[5:])),
                    "weekday_minus_weekend": float(np.mean(profile[:5]) - np.mean(profile[5:])),
                    "amplitude": float(np.ptp(profile)),
                }
        else:
            semantic = {"status": "not_run_due_to_stability_gate"}

        phrases[profile_name] = {
            "raw_observations": len(raw),
            "weekday_counts": np.bincount(train.index.dayofweek, minlength=SEASONAL_PERIOD).tolist(),
            "train": {"start": str(train.index[0].date()), "end": str(train.index[-1].date())},
            "holdout": {
                "observations": len(holdout),
                "start": str(holdout.index[0].date()),
                "end": str(holdout.index[-1].date()),
            },
            "fits": fits,
            "stability": stability,
            "stability_passed": stable,
            "semantic": semantic,
            "holdout_metrics": {
                backend: holdout_metrics(holdout, fitted_objects["minus_0"][backend].forecast)
                for backend in ("sktime", "statsforecast")
            },
        }

    return {
        "configuration": {
            "sp": SEASONAL_PERIOD,
            "train_observations": TRAIN_OBSERVATIONS,
            "holdout_definition": "all real rows after the first 56",
            "truncations_days": list(TRUNCATIONS[1:]),
            "weekday_order": list(WEEKDAY_NAMES),
            "stability_thresholds": {
                "minimum_amplitude": MIN_AMPLITUDE,
                "minimum_correlation": MIN_PROFILE_CORRELATION,
                "maximum_relative_rmse": MAX_RELATIVE_RMSE,
                "maximum_shift_to_amplitude": MAX_SHIFT_TO_AMPLITUDE,
            },
        },
        "phrases": phrases,
        "all_stability_passed": all(item["stability_passed"] for item in phrases.values()),
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

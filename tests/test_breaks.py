"""Тесты детекции структурных сдвигов (issue #7).

Синтетика — основной контур: реальные месячные фикстуры покрывают только
2024-08..2026-07, известные даты 2020 и 2022 в данных отсутствуют, поэтому
поведение детектора на COVID-подобных шоках и уходах брендов проверяется
на смоделированных рядах, а реальные фикстуры — на структурных инвариантах
(отчёт строится, сезонный декабрь не считается разрывом).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.breaks import (
    BreakPoint,
    add_dummy,
    build_breaks_report,
    classify_breaks,
    decide_handling,
    deseasonalize_log,
    known_dates_coverage,
    truncate_series,
)

FIXTURES = Path(__file__).parent / "fixtures"

SEASONAL_FACTORS = {
    1: 0.4,
    2: 0.3,
    3: 0.6,
    4: 0.8,
    5: 0.9,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.2,
    10: 1.5,
    11: 2.0,
    12: 6.0,
}


def synthetic_monthly(
    start: str = "2019-01",
    periods: int = 60,
    base: float = 10000.0,
    noise: float = 0.03,
    seed: int = 0,
    shock: dict[int, float] | None = None,
) -> pd.Series:
    """Мультипликативная сезонность + шум; shock — {индекс: множитель}."""
    rng = np.random.default_rng(seed)
    index = pd.period_range(start=start, periods=periods, freq="M")
    values = [base * SEASONAL_FACTORS[p.month] * (1 + rng.normal(0, noise)) for p in index]
    for i, mult in (shock or {}).items():
        values[i] *= mult
    return pd.Series(values, index=index, name="count")


def test_clean_series_has_no_breaks() -> None:
    """Чистый сезонный ряд без инъекций: декабрьский пик — не разрыв."""
    series = synthetic_monthly()
    assert classify_breaks(series) == []


def test_transient_spike_is_demand() -> None:
    """COVID-подобный шок: spike ×4 два месяца, возврат — demand_transient."""
    series = synthetic_monthly(shock={15: 4.0, 16: 3.0})
    breaks = classify_breaks(series)
    assert len(breaks) == 1
    kind, handling = breaks[0].kind, decide_handling(breaks)[0].handling
    assert kind == "demand_transient"
    assert handling == "keep"


def test_seam_level_shift_is_measurement() -> None:
    """Устойчивый сдвиг на шве склейки окон — разрыв измерения, обрезка."""
    seam_index = 24
    series = synthetic_monthly(shock={i: 5.0 for i in range(seam_index, 60)})
    seams = frozenset({series.index[seam_index]})
    breaks = classify_breaks(series, measurement_seams=seams)
    assert len(breaks) == 1
    assert breaks[0].kind == "measurement"
    decision = decide_handling(breaks)[0]
    assert decision.handling == "truncate"
    truncated = truncate_series(series, decision)
    assert truncated.index.min() == series.index[seam_index]
    assert len(truncated) == 60 - seam_index


def test_truncate_rejects_non_measurement() -> None:
    series = synthetic_monthly(shock={15: 4.0, 16: 3.0})
    decision = decide_handling(classify_breaks(series))[0]
    with pytest.raises(ValueError, match="truncate"):
        truncate_series(series, decision)


def test_persistent_shift_without_context_is_unresolved() -> None:
    """Устойчивый сдвиг без швов и событий: неразрешён — это честный вердикт."""
    series = synthetic_monthly(shock={i: 5.0 for i in range(30, 60)})
    breaks = classify_breaks(series)
    assert len(breaks) == 1
    assert breaks[0].kind == "unresolved"
    assert decide_handling(breaks)[0].handling == "segment"


def test_persistent_shift_at_known_event_is_demand() -> None:
    """Тот же сдвиг на известной содержательной дате — устойчивый сдвиг спроса."""
    series = synthetic_monthly(shock={i: 5.0 for i in range(30, 60)})
    event = frozenset({series.index[30]})
    breaks = classify_breaks(series, demand_events=event)
    assert breaks[0].kind == "demand_persistent"
    decision = decide_handling(breaks)[0]
    assert decision.handling == "dummy"
    framed = add_dummy(series, decision)
    assert framed["post_break"].sum() == 30
    assert framed["count"].sum() > 0


def test_known_dates_coverage_reports_missed_hypotheses() -> None:
    """Известная дата без найденного разрыва — «не подтверждена», не ошибка."""
    series = synthetic_monthly(shock={15: 4.0, 16: 3.0})
    breaks = classify_breaks(series)
    hit = breaks[0].at
    miss = series.index[-1]
    coverage = known_dates_coverage(breaks, frozenset({hit, miss}))
    assert coverage.covered == (hit,)
    assert coverage.missed == (miss,)


def test_deseasonalize_removes_month_profile() -> None:
    """Профиль месяца убирает сезонную амплитуду; уровень остаётся константой
    (он нужен детектору сдвигов, шумовые тесты инвариантны к константе)."""
    series = synthetic_monthly(noise=0.0)
    resid = deseasonalize_log(series)
    amplitude = float(np.log1p(series).max() - np.log1p(series).min())
    assert float(resid.max() - resid.min()) < 0.01 * amplitude


def test_short_series_returns_no_breaks() -> None:
    """Ряд короче 2*min_size не несёт подтверждённого разрыва."""
    index = pd.period_range("2025-01", periods=5, freq="M")
    series = pd.Series([100.0 * (2.0**i) for i in range(5)], index=index)
    assert classify_breaks(series) == []


def test_non_period_index_rejected() -> None:
    with pytest.raises(ValueError, match="PeriodIndex"):
        deseasonalize_log(pd.Series([1.0, 2.0]))


def test_real_monthly_fixture_structural_invariants() -> None:
    """Реальная сезонная фикстура: отчёт строится; типы и решения валидны;
    сезонный декабрь не помечен разрывом (десезонализация работает)."""
    from scripts.experiment_vector_d import load_dynamics_csv

    series = load_dynamics_csv(FIXTURES / "dynamics_seasonal.csv")
    breaks = classify_breaks(series)
    decisions = decide_handling(breaks)
    valid_kinds = {"demand_transient", "demand_persistent", "measurement", "unresolved"}
    valid_handlings = {"keep", "dummy", "truncate", "segment"}
    for b, d in zip(breaks, decisions, strict=True):
        assert b.kind in valid_kinds
        assert d.handling in valid_handlings
        assert d.rationale
        assert b.at.month not in (11, 12), f"сезонный пик помечен разрывом: {b}"
    report = build_breaks_report(series, decisions)
    assert report["series_len"] == len(series)
    assert report["series_range"] == [str(series.index.min()), str(series.index.max())]


def test_spike_at_series_start_uses_first_period() -> None:
    """Шок с первого месяца ряда: at = первый период (с пометкой), а не
    последний период ряда через негативный индекс."""
    index = pd.period_range("2019-01", periods=60, freq="M")
    series = synthetic_monthly(shock={0: 4.0, 1: 3.0})
    spikes = [b for b in classify_breaks(series) if b.evidence["spike"]]
    assert len(spikes) == 1
    assert spikes[0].at == index[0]
    assert spikes[0].evidence["at_series_start"] is True
    assert spikes[0].delta_log > 0


def test_whole_series_spike_does_not_crash() -> None:
    """Ряд, целиком состоящий из «шока» (уровень всюду ×50): относительно
    самого себя шока нет, replaced-нечем-кейс не падает, разрывов 0."""
    index = pd.period_range("2019-01", periods=36, freq="M")
    values = [10000.0 * 50.0 * SEASONAL_FACTORS[p.month] for p in index]
    series = pd.Series(values, index=index, name="count")
    assert classify_breaks(series) == []


def test_negative_shock_is_detected() -> None:
    """Отрицательный переходный шок (провал спроса) равноправен положительному."""
    series = synthetic_monthly(shock={15: 0.2, 16: 0.25})
    spikes = [b for b in classify_breaks(series) if b.evidence["spike"]]
    assert len(spikes) == 1
    assert spikes[0].kind == "demand_transient"
    assert spikes[0].delta_log < 0


def test_deseasonalize_tolerates_missing_month() -> None:
    """После усечения ряда в индексе может не быть отдельных месяцев —
    отсутствие наблюдений периода означает нулевой профиль, не KeyError."""
    series = synthetic_monthly()
    truncated = series.drop(index=[pd.Period("2021-05", freq="M"), pd.Period("2023-05", freq="M")])
    resid = deseasonalize_log(truncated)
    assert len(resid) == len(truncated)


def test_breakpoint_is_frozen_dataclass() -> None:
    br = BreakPoint(at=pd.Period("2024-08", freq="M"), delta_log=0.0, kind="unresolved", evidence={})
    with pytest.raises(Exception):
        br.kind = "measurement"  # type: ignore[misc]

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def load_experiment_module():
    path = Path("scripts/experiment_vector.py")
    spec = spec_from_file_location("experiment_vector", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reads_real_wordstat_csv_with_cr_only_lines_and_russian_months():
    experiment = load_experiment_module()

    series = experiment.read_dynamics_csv(Path("tests/fixtures/dynamics_seasonal.csv"))

    assert series.phrase == "новогодние подарки"
    assert len(series.values) == 24
    assert str(series.values.index[0]) == "2024-08"
    assert str(series.values.index[-1]) == "2026-07"
    assert series.values.iloc[4] == 1_234_632


def test_rejects_duplicate_periods(tmp_path):
    experiment = load_experiment_module()
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "Период;Число запросов;Динамика «фраза»\r"
        "январь 2026;1;\r"
        "январь 2026;2;\r",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="periods must be unique and ordered"):
        experiment.read_dynamics_csv(path)


def test_rejects_missing_month(tmp_path):
    experiment = load_experiment_module()
    path = tmp_path / "gap.csv"
    path.write_text(
        "Период;Число запросов;Динамика «фраза»\r"
        "январь 2026;1;\r"
        "март 2026;2;\r",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="monthly periods must be continuous"):
        experiment.read_dynamics_csv(path)


def test_report_compares_every_successful_truncation_independently(monkeypatch):
    experiment = load_experiment_module()
    full = experiment.DemandVector("test", 100.0, 10.0, {month: 1.0 for month in experiment.MONTH_LABELS}, 1.0)
    shortened = experiment.DemandVector("test", 90.0, 9.0, {month: 1.5 for month in experiment.MONTH_LABELS}, 1.0)
    values = experiment.pd.Series(
        range(30),
        index=experiment.pd.period_range("2024-01", periods=30, freq="M"),
    )
    results = iter([full, shortened, ValueError("not enough cycles")])

    def fake_fit(_):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(experiment, "fit_monthly_vector", fake_fit)
    report = experiment.render_report([(Path("one.csv"), experiment.DemandSeries("фраза", values))])

    assert "| фраза | 3 | 0.1000 | 0.0100 | 0.5000 |" in report


def test_report_keeps_same_phrase_exports_separate(monkeypatch):
    experiment = load_experiment_module()
    values = experiment.pd.Series(
        range(30),
        index=experiment.pd.period_range("2024-01", periods=30, freq="M"),
    )
    vectors = iter(
        [
            experiment.DemandVector("test", 100.0, 0.0, {month: 1.0 for month in experiment.MONTH_LABELS}, 1.0),
            experiment.DemandVector("test", 90.0, 0.0, {month: 1.0 for month in experiment.MONTH_LABELS}, 1.0),
            experiment.DemandVector("test", 80.0, 0.0, {month: 1.0 for month in experiment.MONTH_LABELS}, 1.0),
            experiment.DemandVector("test", 200.0, 0.0, {month: 1.0 for month in experiment.MONTH_LABELS}, 1.0),
            experiment.DemandVector("test", 160.0, 0.0, {month: 1.0 for month in experiment.MONTH_LABELS}, 1.0),
            experiment.DemandVector("test", 120.0, 0.0, {month: 1.0 for month in experiment.MONTH_LABELS}, 1.0),
        ]
    )
    monkeypatch.setattr(experiment, "fit_monthly_vector", lambda _: next(vectors))

    report = experiment.render_report(
        [
            (Path("first.csv"), experiment.DemandSeries("одна фраза", values)),
            (Path("second.csv"), experiment.DemandSeries("одна фраза", values)),
        ]
    )

    assert "| одна фраза | 3 | 0.1000 | 0.0000 | 0.0000 |" in report
    assert "| одна фраза | 3 | 0.2000 | 0.0000 | 0.0000 |" in report


@pytest.mark.parametrize("removed_months", [3, 6])
def test_shortened_monthly_series_cannot_initialize_two_seasonal_cycles(removed_months):
    experiment = load_experiment_module()
    series = experiment.read_dynamics_csv(Path("tests/fixtures/dynamics_seasonal.csv"))

    with pytest.raises(ValueError, match="less than two full seasonal cycles"):
        experiment.fit_monthly_vector(series.values.iloc[:-removed_months])

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


@pytest.mark.parametrize("removed_months", [3, 6])
def test_shortened_monthly_series_cannot_initialize_two_seasonal_cycles(removed_months):
    experiment = load_experiment_module()
    series = experiment.read_dynamics_csv(Path("tests/fixtures/dynamics_seasonal.csv"))

    with pytest.raises(ValueError, match="less than two full seasonal cycles"):
        experiment.fit_monthly_vector(series.values.iloc[:-removed_months])

from pathlib import Path

import pytest

from wordstat_trends.experiment_vector_b import load_wordstat_monthly, parse_period, run_experiment

FIXTURES = Path(__file__).parent / "fixtures"


def test_loads_real_cr_only_wordstat_fixture() -> None:
    raw = (FIXTURES / "dynamics_seasonal.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" in raw and b"\n" not in raw

    series = load_wordstat_monthly(FIXTURES / "dynamics_seasonal.csv")

    assert len(series) == 24
    assert str(series.index[0]) == "2024-08"
    assert str(series.index[-1]) == "2026-07"
    assert series.loc[parse_period("декабрь 2024")] == 1_234_632


@pytest.mark.parametrize("value", ["2024-08", "Foo 2024", "август"])
def test_rejects_non_wordstat_period(value: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Wordstat monthly period"):
        parse_period(value)


def test_current_fixtures_trigger_insufficient_history_stop_condition() -> None:
    result = run_experiment(FIXTURES)

    assert result["status"] == "inconclusive_insufficient_history"
    assert result["stability_passed"] is False
    assert result["semantic_check"] == "not_run_due_to_stop_condition"
    for windows in result["fits"].values():
        assert windows["minus_0"]["status"] == "fit"
        assert len(windows["minus_0"]["vector"]["seasonal"]) == 12
        for key in ("minus_3", "minus_6"):
            assert windows[key]["status"] == "fit_failed"
            assert "less than two full seasonal cycles" in windows[key]["error"]

"""Тесты бэктест-раннера по всем фразам + сводной MASE (issue #76, фаза 2.4)."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from wordstat.models import CollectionManifest, ExportSummary, WordstatView

from wordstat_trends.forecasting.backtest import (
    BASELINE_MODEL,
    SUMMARY_COLUMNS,
    TABLE_COLUMNS,
    BacktestError,
    run_backtest,
    summarize_backtest,
    write_backtest_report,
)
from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.loader import load_run

FIXTURES = Path(__file__).parent / "fixtures"

# Две фразы вместо трёх: каждая — полный 104-точечный склеенный ряд, прогон
# 4 моделей × 2 фолда небыстрый; детерминизм удваивает стоимость.
PHRASES = ["купить телефон", "новогодние подарки"]


def _make_run(tmp_path: Path, phrase: str) -> Path:
    """Run-каталог поверх фикстуры полного склеенного ряда (как в
    test_forecasting_models.py), но с разной фразой в manifest.json —
    датасет из нескольких фраз."""

    run = tmp_path / phrase.replace(" ", "_")
    run.mkdir(parents=True)
    (run / "dynamics.csv").write_bytes((FIXTURES / "dynamics_full_stitched.csv").read_bytes())
    manifest = CollectionManifest(
        phrase=phrase,
        region="Россия",
        created_at=datetime(2026, 9, 9, 0, 0, 0, tzinfo=UTC),
        source_url="https://wordstat.yandex.ru/",
        exports=[
            ExportSummary(
                view=WordstatView.DYNAMICS,
                file="dynamics.parquet",
                raw_file="dynamics.csv",
                row_count=104,
                dtypes={"Период": "string", "Число запросов": "int64"},
            )
        ],
    )
    (run / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return run


@pytest.fixture()
def series_by_phrase(tmp_path) -> dict[str, pd.Series]:
    return {
        phrase: to_monthly_series(load_run(_make_run(tmp_path, phrase)))
        for phrase in PHRASES
    }


def test_run_backtest_table_format_is_stable(series_by_phrase):
    # Формат таблицы зафиксирован: колонки, сортировка по фразе, внутри
    # фразы — порядок MODEL_FACTORIES (бейзлайн первым), mase_median > 0.
    table = run_backtest(series_by_phrase)
    assert list(table.columns) == TABLE_COLUMNS
    assert list(table["phrase"].unique()) == sorted(PHRASES)
    model_order = list(table[table["phrase"] == PHRASES[0]]["model"])
    assert model_order[0] == BASELINE_MODEL
    for phrase in PHRASES:
        assert list(table[table["phrase"] == phrase]["model"]) == model_order
    # 104 точки → 2 фолда, как в test_forecasting_models.
    assert set(table["n_folds"]) == {2}
    mase_values = table["mase_median"].to_numpy()
    assert np.all(np.isfinite(mase_values)) and np.all(mase_values > 0)
    assert len(table) == len(PHRASES) * len(model_order)


def test_run_backtest_is_deterministic(series_by_phrase):
    # Два прогона по одним фразам дают побитово одинаковую таблицу
    # (random_state моделей зафиксирован ещё в models.py).
    first = run_backtest(series_by_phrase)
    second = run_backtest(series_by_phrase)
    pd.testing.assert_frame_equal(first, second)


def test_run_backtest_empty_phrases_fail_loudly():
    with pytest.raises(BacktestError, match="пустой список фраз"):
        run_backtest({})


def test_run_backtest_short_series_fails_loudly(series_by_phrase):
    # Ряд короче initial_window + test_length не даёт ни одного фолда —
    # громкая ошибка с перечнем проблемных фраз, не тихий пустой результат.
    short = {**series_by_phrase, "короткая": series_by_phrase[PHRASES[0]].iloc[:24]}
    with pytest.raises(BacktestError, match="короткая.*24"):
        run_backtest(short)


def test_summarize_backtest_aggregates_and_win_rates(series_by_phrase):
    table = run_backtest(series_by_phrase)
    summary = summarize_backtest(table)
    assert list(summary.columns) == SUMMARY_COLUMNS
    # Порядок моделей — MODEL_FACTORIES, бейзлайн первым.
    assert summary["model"].iloc[0] == BASELINE_MODEL
    # Бейзлайн не побеждает сам себя (строгое неравенство) — 0.0.
    baseline_rate = summary.loc[summary["model"] == BASELINE_MODEL, "win_rate_vs_baseline"].iloc[0]
    assert baseline_rate == 0.0
    # Доли побед на двух фразах кратны 0.5, медианы конечны и положительны.
    for row in summary.itertuples():
        assert row.n_phrases == len(PHRASES)
        assert row.win_rate_vs_baseline in (0.0, 0.5, 1.0)
        assert np.isfinite(row.mase_median) and row.mase_median > 0
    # Сводная медиана модели = медиана её строк в таблице.
    for row in summary.itertuples():
        expected = table.loc[table["model"] == row.model, "mase_median"].median()
        assert row.mase_median == expected


def test_summarize_backtest_empty_table_fails_loudly():
    with pytest.raises(BacktestError, match="пустая таблица"):
        summarize_backtest(pd.DataFrame(columns=TABLE_COLUMNS))


def test_write_backtest_report_csv_roundtrip(series_by_phrase, tmp_path):
    # CSV пишется и читается назад без потерь: формат стабилен между
    # прогонами, кодировка utf-8-sig — как выгрузки Вордстата (docs/DATA.md).
    table = run_backtest(series_by_phrase)
    summary = summarize_backtest(table)
    table_path, summary_path = write_backtest_report(table, summary, tmp_path / "outputs")
    assert table_path.name == "backtest_mase.csv"
    assert summary_path.name == "backtest_mase_summary.csv"
    back = pd.read_csv(table_path, encoding="utf-8-sig")
    assert list(back.columns) == TABLE_COLUMNS
    pd.testing.assert_frame_equal(back, table, check_dtype=False)
    back_summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    assert list(back_summary.columns) == SUMMARY_COLUMNS

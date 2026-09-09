"""Тесты сезонного наивного бейзлайна и каркаса валидации (issue #11, фаза 2)."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from wordstat.models import CollectionManifest, ExportSummary, WordstatView

from wordstat_trends.forecasting.baseline import (
    INITIAL_WINDOW,
    SP,
    TEST_LENGTH,
    evaluate_forecaster,
    make_expanding_splitter,
    seasonal_naive,
    to_monthly_series,
)
from wordstat_trends.loader import load_run

FIXTURES = Path(__file__).parent / "fixtures"


def _make_run(tmp_path: Path, csv_name: str = "dynamics_full_stitched.csv") -> Path:
    """Собрать run-каталог поверх фикстуры полного склеенного ряда."""

    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "dynamics.csv").write_bytes((FIXTURES / csv_name).read_bytes())
    manifest = CollectionManifest(
        phrase="фаза 2 бейзлайн",
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
def y(tmp_path) -> pd.Series:
    return to_monthly_series(load_run(_make_run(tmp_path)))


def test_to_monthly_series_index_and_length(y):
    assert len(y) == 104
    assert isinstance(y.index, pd.PeriodIndex)
    assert y.index.freqstr == "M"
    assert y.index[0] == pd.Period("2018-01", freq="M")
    assert y.index[-1] == pd.Period("2026-08", freq="M")


def test_splitter_folds_are_chronological_no_shuffle(y):
    splitter = make_expanding_splitter()
    folds = list(splitter.split(y))
    # 104 точки: окно 72 + два фолда по 12 (72+12, 84+12); третий (96+12)
    # упирается в конец ряда и отбрасывается сплиттером целиком.
    assert len(folds) == 2
    for train, test in folds:
        # Хронология: обучение строго раньше теста, тест — без прореживания.
        assert train[-1] < test[0]
        assert np.all(np.diff(test) == 1)
    # Окно только расширяется, старт теста только растёт — никакой перестановки.
    assert [f[1][0] for f in folds] == sorted(f[1][0] for f in folds)
    assert folds[0][0][-1] < folds[1][0][-1]
    # Геометрия фолда: 12 последовательных точек, сдвиг на сезонный цикл.
    assert len(folds[0][0]) == INITIAL_WINDOW
    assert len(folds[0][1]) == TEST_LENGTH
    assert folds[1][1][0] - folds[0][1][0] == SP


def test_seasonal_naive_forecast_is_value_12_months_back(y):
    # Ручная проверка на одном фолде: fit по первым 72 точкам, fh=1..12.
    fc = seasonal_naive()
    fc.fit(y.iloc[:INITIAL_WINDOW])
    pred = fc.predict(list(range(1, TEST_LENGTH + 1)))
    expected = y.iloc[INITIAL_WINDOW - SP : INITIAL_WINDOW]
    assert pred is not None
    expected.index = pred.index
    pd.testing.assert_series_equal(pred, expected.astype("float64"), check_names=False)


def test_evaluate_forecaster_mase_report(y):
    result = evaluate_forecaster(seasonal_naive(), y)
    assert len(result) == 2
    assert [c for c in result.columns if c.startswith("test_")] == ["test_MeanAbsoluteScaledError"]
    # Наивный по своей природе не идеален (тренд + пики чётных декабрей),
    # но и не катастрофичен: MASE конечен, положителен и одного порядка
    # с 1 — резких выбросов на фолдах нет.
    mase = result["test_MeanAbsoluteScaledError"]
    assert np.all(np.isfinite(mase)) and np.all(mase > 0)
    assert mase.median() < 5


def test_evaluate_forecaster_deterministic(y):
    first = evaluate_forecaster(seasonal_naive(), y)["test_MeanAbsoluteScaledError"].tolist()
    second = evaluate_forecaster(seasonal_naive(), y)["test_MeanAbsoluteScaledError"].tolist()
    assert first == second

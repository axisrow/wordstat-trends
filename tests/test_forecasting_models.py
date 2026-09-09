"""Тесты ThetaForecaster и AutoETS поверх каркаса валидации (issue #74)."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from wordstat.models import CollectionManifest, ExportSummary, WordstatView

from wordstat_trends.forecasting.baseline import (
    INITIAL_WINDOW,
    evaluate_forecaster,
    make_expanding_splitter,
    seasonal_naive,
    to_monthly_series,
)
from wordstat_trends.forecasting.models import (
    auto_ets,
    ets_full_series_structure,
    evaluate_models,
    theta_model,
)
from wordstat_trends.loader import load_run

FIXTURES = Path(__file__).parent / "fixtures"


def _make_run(tmp_path: Path) -> Path:
    """Собрать run-каталог поверх фикстуры полного склеенного ряда (как в
    test_forecasting_baseline.py — общий месячный ряд для честного
    сравнения моделей и бейзлайна)."""

    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "dynamics.csv").write_bytes((FIXTURES / "dynamics_full_stitched.csv").read_bytes())
    manifest = CollectionManifest(
        phrase="фаза 2.2 модели",
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


def test_evaluate_models_runs_all_through_framework(y):
    # Обе модели + бейзлайн через ОДИН каркас: те же фолды, тот же MASE.
    report = evaluate_models(y)
    assert list(report.index) == ["seasonal_naive", "theta", "auto_ets"]
    fold_cols = [c for c in report.columns if c.startswith("fold_")]
    # 104 точки → те же 2 фолда, что у бейзлайна в test_forecasting_baseline.
    assert len(fold_cols) == 2
    values = report[fold_cols].to_numpy()
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    # Сводная колонка для сравнения с бейзлайном (MASE < 1 = лучше наива).
    assert np.all(np.isfinite(report["mase_median"]))


def test_evaluate_models_same_folds_as_baseline(y):
    # Никаких отдельных сплитов под модель: на общем сплиттере значения
    # бейзлайна из evaluate_models совпадают с прямым прогоном каркаса.
    splitter = make_expanding_splitter()
    report = evaluate_models(y, splitter)
    direct = evaluate_forecaster(seasonal_naive(), y, splitter)["test_MeanAbsoluteScaledError"]
    np.testing.assert_allclose(
        report.loc["seasonal_naive"][["fold_0", "fold_1"]].to_numpy(),
        direct.to_numpy(),
    )


def test_theta_forecast_matches_framework_geometry(y):
    # Ручной прогон Theta без каркаса не падает и даёт конечный прогноз
    # на те же 12 точек (fh как у сплиттера).
    fc = theta_model()
    fc.fit(y.iloc[:INITIAL_WINDOW])
    pred = fc.predict(list(range(1, 13)))
    assert pred is not None and len(pred) == 12
    assert np.all(np.isfinite(pred.to_numpy()))


def test_models_deterministic(y):
    # Два прогона каждой модели через каркас дают побитово одинаковый MASE
    # (random_state зафиксирован; Theta детерминирован по построению).
    for fc in (theta_model(), auto_ets()):
        first = evaluate_forecaster(fc, y)["test_MeanAbsoluteScaledError"].tolist()
        second = evaluate_forecaster(fc, y)["test_MeanAbsoluteScaledError"].tolist()
        assert first == second


def test_ets_full_series_structure_pinned(y):
    # Зафиксированная структура ETS на полном ряду (требование #74 после
    # вывода #32: смена has_seasonal между окнами — структуры не менять
    # молча). spec — три буквы ETS: ошибка/тренд (d = демпфированный)/
    # сезонность; начальное окно 72 месяца выше порога детекции из #34.
    structure = ets_full_series_structure(y)
    assert set(structure) >= {"spec", "has_seasonal", "error", "trend", "seasonal", "damped_trend"}
    assert isinstance(structure["has_seasonal"], bool)
    # spec вида AAdA / ANN / ... : буква ошибки, тренд (N/A/Ad), сезонность (N/A/M).
    assert isinstance(structure["spec"], str) and len(structure["spec"]) in (3, 4)
    assert structure["spec"][0] in "AM"
    assert structure["spec"][-1] in "NAM"
    # Сезонность выбрана: последний символ — A или M, и только тогда.
    assert (structure["spec"][-1] in "AM") == structure["has_seasonal"]
    # Детерминированность выбора спецификации на полном ряду.
    assert ets_full_series_structure(y) == structure

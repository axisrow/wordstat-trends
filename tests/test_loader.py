"""Тесты загрузчика run-каталога (issue #61)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from wordstat.models import CollectionManifest, ExportSummary, WordstatView

from wordstat_trends.loader import (
    EmptyDatasetError,
    LoaderError,
    ManifestError,
    load_run,
    parse_dynamics_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _make_run(tmp_path: Path, csv_name: str = "dynamics_high_freq.csv") -> Path:
    """Собрать минимальный run-каталог поверх реальной фикстуры замера #2."""

    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "dynamics.csv").write_bytes((FIXTURES / csv_name).read_bytes())
    manifest = CollectionManifest(
        phrase="купить телефон",
        region="Россия",
        created_at=datetime(2026, 8, 21, 1, 40, 38, tzinfo=UTC),
        source_url="https://wordstat.yandex.ru/",
        exports=[
            ExportSummary(
                view=WordstatView.DYNAMICS,
                file="dynamics.parquet",
                raw_file="dynamics.csv",
                row_count=24,
                dtypes={"Период": "string", "Число запросов": "int64"},
            )
        ],
    )
    (run / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return run


def test_load_run_dataframe_and_metadata(tmp_path):
    run = _make_run(tmp_path)
    df = load_run(run)

    assert len(df) == 24
    assert list(df.columns) == ["period", "queries", "share_pct"]
    # Первая строка фикстуры: «август 2024;997 977;0,0115;» (docs/DATA.md).
    assert df.iloc[0]["period"] == "2024-08"
    assert df.iloc[0]["queries"] == 997977
    assert df.iloc[0]["share_pct"] == pytest.approx(0.0115)
    # Декабрьский пик сезонной фразы — «1 234 632» из дословного примера.
    seasonal = load_run(_make_run(tmp_path / "s", "dynamics_seasonal.csv"))
    december = seasonal[seasonal["period"] == "2024-12"]
    assert december.iloc[0]["queries"] == 1234632
    assert df["queries"].dtype == "int64"

    assert df.attrs["phrase"] == "купить телефон"
    assert df.attrs["region"] == "Россия"
    assert df.attrs["created_at"] == datetime(2026, 8, 21, 1, 40, 38, tzinfo=UTC)
    assert df.attrs["dtypes"]["Число запросов"] == "int64"


def test_rows_sorted_by_period(tmp_path):
    df = load_run(_make_run(tmp_path))
    assert list(df["period"]) == sorted(df["period"])


def test_cr_only_and_bom_are_handled():
    # Фикстура — UTF-8 с BOM и CR-only; если бы читали по \n, увидели бы одну строку.
    rows = parse_dynamics_rows((FIXTURES / "dynamics_seasonal.csv").read_text(encoding="utf-8-sig"))
    assert len(rows) == 24
    assert rows[0][0] == "2024-08"


def test_empty_dataset_rejected():
    with pytest.raises(EmptyDatasetError):
        parse_dynamics_rows("Период;Число запросов;Доля от всех запросов, %;\r")


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(ManifestError):
        load_run(tmp_path)


def test_invalid_manifest_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_run(tmp_path)


def test_missing_csv_rejected(tmp_path):
    run = _make_run(tmp_path)
    (run / "dynamics.csv").unlink()
    with pytest.raises(LoaderError, match="keep-raw"):
        load_run(run)


def test_unsupported_view_rejected(tmp_path):
    with pytest.raises(LoaderError, match="top_popular|dynamics"):
        load_run(_make_run(tmp_path), view="top_popular")

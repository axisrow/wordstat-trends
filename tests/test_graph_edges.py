"""Провайдер рёбер поверх экспортов wordstat-cli (issue #82).

Фикстуры повторяют реальную раскладку прогона ``wordstat collect``:
``runs/<dir>/manifest.json`` + ``top_related.parquet`` с локализованными
колонками (живой прогон 2026-09-09, см. docstring провайдера). Браузера
и сети нет — только файлы.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from wordstat_trends.graph.edges import ExportEdgeProvider
from wordstat_trends.graph.traverse import EdgesNotAvailable, RelatedPhrase

TITLE = "Запросы, похожие на «{phrase}», 08.08.2026 — 08.09.2026, Россия, все устройства"


def write_run(
    runs_root: Path,
    phrase: str,
    neighbors: list[tuple[str, int]],
    *,
    created_at: str = "2026-09-09T10:00:00Z",
    run_name: str | None = None,
    with_related: bool = True,
) -> Path:
    run_dir = runs_root / (run_name or f"20260909T100000Z-{phrase.replace(' ', '-')}")
    run_dir.mkdir(parents=True)
    related = "top_related.parquet"
    exports = [{"view": "top_related", "file": related, "row_count": len(neighbors)}] if with_related else []
    (run_dir / "manifest.json").write_text(
        json.dumps({"phrase": phrase, "created_at": created_at, "exports": exports}, ensure_ascii=False),
        encoding="utf-8",
    )
    if with_related:
        frame = pd.DataFrame(
            {
                "Запросы со словами": [n[0] for n in neighbors],
                "Число запросов": [n[1] for n in neighbors],
                TITLE.format(phrase=phrase): [""] * len(neighbors),
            }
        )
        frame.to_parquet(run_dir / related, index=False)
    return run_dir


def test_neighbors_from_real_layout(tmp_path):
    runs = tmp_path / "results" / "runs"
    write_run(runs, "ремонт квартир", [("отделка", 1470978), ("ремонтные работы", 160387)])

    provider = ExportEdgeProvider(runs)

    assert provider.neighbors("ремонт квартир") == [
        RelatedPhrase(phrase="отделка", frequency=1470978),
        RelatedPhrase(phrase="ремонтные работы", frequency=160387),
    ]
    assert provider.has_export("ремонт квартир")


def test_uncollected_phrase_waits_not_dead_ends(tmp_path):
    """Нет экспорта ≠ тупик: кандидат должен ждать сбора во фронтире."""
    provider = ExportEdgeProvider(tmp_path / "runs")

    assert not provider.has_export("нет такой фразы")
    with pytest.raises(EdgesNotAvailable):
        provider.neighbors("нет такой фразы")


def test_latest_run_wins(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "фраза", [("старый сосед", 10)], created_at="2026-09-01T00:00:00Z", run_name="old")
    write_run(runs, "фраза", [("новый сосед", 20)], created_at="2026-09-08T00:00:00Z", run_name="new")

    neighbors = ExportEdgeProvider(runs).neighbors("фраза")

    assert [n.phrase for n in neighbors] == ["новый сосед"]


def test_run_without_related_export_is_not_an_edge_source(tmp_path):
    runs = tmp_path / "runs"
    run_dir = write_run(runs, "фраза", [], with_related=False)

    provider = ExportEdgeProvider(runs)

    assert not provider.has_export("фраза")
    with pytest.raises(EdgesNotAvailable):
        provider.neighbors("фраза")
    assert not (run_dir / "top_related.parquet").exists()


def test_manifest_pointing_to_missing_file(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "фраза", [("сосед", 5)])
    (runs / "20260909T100000Z-фраза" / "top_related.parquet").unlink()

    provider = ExportEdgeProvider(runs)

    assert not provider.has_export("фраза")
    with pytest.raises(EdgesNotAvailable):
        provider.neighbors("фраза")


def test_broken_manifest_is_skipped_not_fatal(tmp_path):
    runs = tmp_path / "runs"
    good = write_run(runs, "хорошая фраза", [("сосед", 7)])
    bad = runs / "20260909T110000Z-битая"
    bad.mkdir()
    (bad / "manifest.json").write_text("{не json", encoding="utf-8")

    provider = ExportEdgeProvider(runs)

    assert [n.phrase for n in provider.neighbors("хорошая фраза")] == ["сосед"]
    assert good.is_dir()


def test_empty_phrase_rows_are_skipped(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "фраза", [("", 0), ("сосед", 3)])

    neighbors = ExportEdgeProvider(runs).neighbors("фраза")

    assert neighbors == [RelatedPhrase(phrase="сосед", frequency=3)]


def test_unexpected_columns_give_no_edges(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "20260909T100000Z-фраза"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        '{"phrase": "фраза", "created_at": "2026-09-09T10:00:00Z", '
        '"exports": [{"view": "top_related", "file": "top_related.parquet", "row_count": 1}]}',
        encoding="utf-8",
    )
    pd.DataFrame({"что-то": ["иное"]}).to_parquet(run_dir / "top_related.parquet", index=False)

    assert ExportEdgeProvider(runs).neighbors("фраза") == []


def test_non_numeric_frequency_row_is_skipped(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "20260909T100000Z-фраза"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        '{"phrase": "фраза", "created_at": "2026-09-09T10:00:00Z", '
        '"exports": [{"view": "top_related", "file": "top_related.parquet", "row_count": 2}]}',
        encoding="utf-8",
    )
    # частотность ушла во float с NaN — битая строка не роняет остальные рёбра
    pd.DataFrame(
        {
            "Запросы со словами": ["хороший сосед", "битый сосед"],
            "Число запросов": [7.0, float("nan")],
        }
    ).to_parquet(run_dir / "top_related.parquet", index=False)

    neighbors = ExportEdgeProvider(runs).neighbors("фраза")

    assert neighbors == [RelatedPhrase(phrase="хороший сосед", frequency=7)]


def test_malformed_exports_entries_give_no_source(tmp_path):
    runs = tmp_path / "runs"
    for name, exports in (
        ("a", ["не dict"]),
        ("b", "не список"),
        ("c", [{"view": "top_related", "file": 42}]),
    ):
        run_dir = runs / f"20260909T10000{name}Z-фраза"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"phrase": "фраза", "created_at": "2026-09-09T10:00:00Z", "exports": exports}),
            encoding="utf-8",
        )

    provider = ExportEdgeProvider(runs)

    assert not provider.has_export("фраза")
    with pytest.raises(EdgesNotAvailable):
        provider.neighbors("фраза")


def test_non_object_manifest_is_skipped(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "20260909T100000Z-фраза"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('["список", "не объект"]', encoding="utf-8")

    provider = ExportEdgeProvider(runs)

    assert not provider.has_export("фраза")

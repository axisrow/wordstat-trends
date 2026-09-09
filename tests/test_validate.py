"""Контрактные тесты на данные (issue #94, Фаза 7.1).

Позитивная часть — дословные выгрузки замера #2 (``tests/fixtures/dynamics_*.csv``)
проходят контракт целиком; негативная — по одному нарушению на каждый пункт
контракта из ``docs/DATA.md``/``docs/DATASET.md``.
"""

from datetime import UTC, datetime
from pathlib import Path

from wordstat.models import CollectionManifest, ExportSummary, WordstatView

from wordstat_trends.__main__ import main as cli_main
from wordstat_trends.validate import (
    NBSP,
    validate_dataset,
    validate_dynamics_csv,
    validate_run,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Эталонный заголовок из замера #2 (docs/DATA.md): 4 поля, четвёртое —
# длинный текст заголовка графика.
_HEADER = (
    "Период;Число запросов;Доля от всех запросов, %;"
    "Динамика частотности запросов «ф», по месяцам, 01.08.2024 — 31.07.2026, Россия"
)
_ROW = "август 2024;997 977;0,0115;"
_ROW2 = "сентябрь 2024;87 798;0,00095;"


def _csv(*rows: str, header: str = _HEADER, bom: bool = True, crlf: bool = False) -> bytes:
    """Собрать валидный dynamics.csv, отклоняясь от контракта только там, где просят."""
    lines = [header, *rows]
    sep = "\r\n" if crlf else "\r"
    text = sep.join(lines) + sep
    return ("﻿" if bom else "").encode() + text.encode()


def _checks(violations) -> set[str]:
    return {v.check for v in violations if v.severity == "error"}


def _make_run(tmp_path: Path, csv_name: str = "dynamics_high_freq.csv", *, tops: str = "empty") -> Path:
    """Run-каталог поверх реальной фикстуры; tops — состояние top-представлений."""
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "dynamics.csv").write_bytes((FIXTURES / csv_name).read_bytes())
    if tops == "empty":
        for view in ("top_popular", "top_related"):
            (run / f"{view}.csv").write_text(
                "Поисковый запрос;Число запросов, мес.;Доля от всех запросов, %;\r",
                encoding="utf-8-sig",
            )
    elif tops == "nonempty":
        (run / "top_popular.csv").write_bytes(
            "﻿Запрос;Число запросов;Доля;\rкупить телефон;997 977;0,0115;\r".encode()
        )
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


# --- Позитивная часть: дословные выгрузки замера #2 проходят контракт --------


def test_all_fixtures_pass_contract():
    for fixture in FIXTURES.glob("dynamics_*.csv"):
        # mvp-фикстуры дневные (issue #23), stitched — синтетика для тестов
        # склейки (заголовок без длинного четвёртого поля): контракт дословных
        # месячных выгрузок — на трёх фикстурах замера #2.
        if "mvp" in fixture.name or "stitched" in fixture.name:
            continue
        violations = validate_dynamics_csv(fixture.read_bytes(), source=fixture.name)
        assert _checks(violations) == set(), (fixture.name, violations)


def test_daily_mvp_fixtures_report_period_format_only():
    # Дневные фикстуры MVP (#23) — тот же формат файла, но период не месячный:
    # контракт обязан это зафиксировать как «период», а не молча принять.
    for name in ("dynamics_daily_high_freq_mvp.csv", "dynamics_daily_mid_freq_mvp.csv",
                 "dynamics_daily_seasonal_mvp.csv"):
        violations = validate_dynamics_csv((FIXTURES / name).read_bytes(), source=name)
        assert "period-format" in _checks(violations), (name, violations)


def test_valid_synthetic_passes():
    assert validate_dynamics_csv(_csv(_ROW, _ROW2)) == []


def test_validate_run_ok(tmp_path):
    report = validate_run(_make_run(tmp_path))
    assert report.ok(), report.violations
    assert report.violations == []  # пустые топы — известный факт, не нарушение


def test_cli_ok(tmp_path, capsys):
    assert cli_main([str(_make_run(tmp_path))]) == 0
    assert "OK" in capsys.readouterr().out


# --- Негативная часть: по одному нарушению на пункт контракта -----------------


def test_no_bom():
    assert _checks(validate_dynamics_csv(_csv(_ROW, bom=False))) == {"bom"}


def test_lf_line_endings():
    assert _checks(validate_dynamics_csv(_csv(_ROW, crlf=True))) == {"line-endings"}


def test_renamed_column():
    bad = _HEADER.replace("Число запросов", "Показов")  # ловушка синтетической фикстуры
    assert _checks(validate_dynamics_csv(_csv(_ROW, header=bad))) == {"schema"}


def test_three_field_header():
    assert _checks(validate_dynamics_csv(_csv(_ROW, header="Период;Число запросов;Доля"))) == {"schema"}


def test_nonempty_fourth_field_in_data_row():
    assert _checks(validate_dynamics_csv(_csv("август 2024;997 977;0,0115;лишнее"))) == {"schema"}


def test_row_with_wrong_field_count():
    assert _checks(validate_dynamics_csv(_csv("август 2024;997 977"))) == {"schema"}


def test_non_month_period():
    assert _checks(validate_dynamics_csv(_csv("01.2024;997 977;0,0115;"))) == {"period-format"}


def test_nbsp_thousands_separator():
    row = "август 2024;997" + NBSP + "977;0,0115;"
    assert _checks(validate_dynamics_csv(_csv(row))) == {"thousands-separator"}


def test_negative_queries():
    assert _checks(validate_dynamics_csv(_csv("август 2024;-100;0,0115;"))) == {"number-format", "range"}


def test_share_decimal_point():
    assert _checks(validate_dynamics_csv(_csv("август 2024;997 977;0.0115;"))) == {"decimal-separator"}


def test_share_out_of_range():
    assert _checks(validate_dynamics_csv(_csv("август 2024;997 977;150,5;"))) == {"range"}


def test_gap_in_series():
    # август 2024 → октябрь 2024: сентября нет.
    assert _checks(validate_dynamics_csv(_csv(_ROW, "октябрь 2024;211 226;0,00202;"))) == {"continuity"}


def test_duplicate_period():
    assert _checks(validate_dynamics_csv(_csv(_ROW, _ROW))) == {"continuity"}


def test_empty_data_rows():
    assert _checks(validate_dynamics_csv(_csv())) == {"schema"}


# --- Run-каталог и CLI --------------------------------------------------------


def test_run_without_manifest(tmp_path):
    run = _make_run(tmp_path)
    (run / "manifest.json").unlink()
    report = validate_run(run)
    assert _checks(report.violations) == {"manifest"}
    assert not report.ok()


def test_run_without_dynamics(tmp_path):
    run = _make_run(tmp_path)
    (run / "dynamics.csv").unlink()
    assert _checks(validate_run(run).violations) == {"schema"}


def test_nonempty_top_view_is_info_not_error(tmp_path):
    report = validate_run(_make_run(tmp_path, tops="nonempty"))
    assert report.ok(), report.violations  # info не роняет
    info = [v for v in report.violations if v.check == "known-empty-view"]
    assert len(info) == 1
    assert info[0].severity == "info"
    assert "расширить контракт" in info[0].message


def test_cli_fails_on_violation(tmp_path, capsys):
    run = _make_run(tmp_path)
    (run / "dynamics.csv").write_bytes(_csv("август 2024;-100;0,0115;"))
    assert cli_main([str(run)]) == 1
    captured = capsys.readouterr()
    assert "range" in captured.out
    assert "не прошли контракт" in captured.err


def test_cli_no_args():
    assert cli_main([]) == 2


def test_cli_missing_path(tmp_path, capsys):
    assert cli_main([str(tmp_path / "нет такого")]) == 1
    assert "не существует" in capsys.readouterr().out


def test_existing_dir_without_run_markers(tmp_path, capsys):
    # Существующий каталог без dynamics.csv и manifest.json — не «не найден»,
    # а отдельное сообщение: путь есть, но это не run-каталог и не выгрузка.
    empty = tmp_path / "каталог"
    empty.mkdir()
    assert cli_main([str(empty)]) == 1
    out = capsys.readouterr().out
    assert "не run-каталог" in out
    assert "не найден" not in out


def test_validate_dataset_mixes_run_dirs_and_csv(tmp_path):
    run = _make_run(tmp_path / "a")
    csv = tmp_path / "b.csv"
    csv.write_bytes((FIXTURES / "dynamics_seasonal.csv").read_bytes())
    reports = validate_dataset([run, csv])
    assert [r.ok() for r in reports] == [True, True]

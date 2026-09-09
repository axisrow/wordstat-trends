"""Бэктест-раннер по всем фразам датасета (issue #76, фаза 2.4).

Прогоняет бейзлайн и все модели (``MODEL_FACTORIES`` из
``wordstat_trends.forecasting.models``) через общий каркас валидации
``baseline.py`` по каждому run-каталогу сборщика (``manifest.json`` +
``dynamics.csv``, формат — см. docs/DATA.md), агрегирует MASE по фразам
(медиана, доля побед над бейзлайном) и пишет два CSV в ``outputs/``:

- ``outputs/backtest_mase.csv`` — таблица ``фраза × модель × MASE``;
- ``outputs/backtest_mase_summary.csv`` — сводная по моделям.

Запуск:

    python scripts/backtest_forecasters.py runs/run_ph1 runs/run_ph2 ...

Run-каталоги передаются аргументами командной строки; без аргументов скрипт
падает с понятной ошибкой — молча прогонять «ничего» нельзя (см.
BacktestError в ``wordstat_trends.forecasting.backtest``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ограничить BLAS/OMP-потоки ДО импорта numpy/statsmodels — как в
# scripts/experiment_vector_d.py: несколько независимых прогонов не должны
# делить все ядра машины.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wordstat_trends.forecasting.backtest import (  # noqa: E402
    BacktestError,
    run_backtest,
    summarize_backtest,
    write_backtest_report,
)
from wordstat_trends.forecasting.baseline import to_monthly_series  # noqa: E402
from wordstat_trends.forecasting.models import RANDOM_STATE  # noqa: E402
from wordstat_trends.loader import load_run  # noqa: E402
from wordstat_trends.run_meta import run_metadata  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_series_by_phrase(run_directories: list[Path]) -> dict[str, pd.Series]:
    """Run-каталоги → {фраза: месячный ряд}. Фраза берётся из manifest.json
    (``df.attrs``); дубликат фразы — ошибка: двум разным прогонам одной фразы
    нет места в одной таблице ``фраза × модель``."""

    series_by_phrase: dict[str, pd.Series] = {}
    for run in run_directories:
        frame = load_run(run)
        phrase = str(frame.attrs["phrase"])
        if phrase in series_by_phrase:
            raise BacktestError(f"фраза {phrase!r} встречается дважды (run {run}) — дубликаты не поддерживаются")
        series_by_phrase[phrase] = to_monthly_series(frame)
    return series_by_phrase


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(
            "usage: python scripts/backtest_forecasters.py <run-dir> [<run-dir> ...]\n"
            "каждый run-dir — каталог с manifest.json и dynamics.csv (прогон с --keep-raw)",
            file=sys.stderr,
        )
        return 2

    series_by_phrase = load_series_by_phrase([Path(a) for a in args])
    table = run_backtest(series_by_phrase)
    summary = summarize_backtest(table)
    table_path, summary_path = write_backtest_report(table, summary, OUTPUTS_DIR)

    # Метаданные прогона (issue #95) рядом с CSV: seed=RANDOM_STATE —
    # единственная стохастика прогона (init-розыгрыш AutoETS, аудит в
    # wordstat_trends/seeds.py). Входы — dynamics.csv каждого run-каталога.
    run_dirs = [Path(a) / "dynamics.csv" for a in args]
    meta_path = OUTPUTS_DIR / "backtest_run_meta.json"
    meta_path.write_text(
        json.dumps(run_metadata(inputs=tuple(run_dirs), seed=RANDOM_STATE), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(summary.to_string(index=False))
    print(f"\nтаблица:   {table_path}")
    print(f"сводная:   {summary_path}")
    print(f"метаданные: {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

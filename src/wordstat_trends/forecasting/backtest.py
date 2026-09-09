"""Бэктест-раннер по всем фразам датасета + сводная MASE (issue #76, фаза 2.4).

Одна фраза — один ряд; одиночный прогон :func:`evaluate_models` ничего не
говорит о поведении модели на датасете. Здесь он оборачивается в цикл по
фразам: каждая фраза прогоняется через ТОТ ЖЕ каркас валидации из
``baseline.py`` (те же фолды, тот же MASE), результаты агрегируются по фразам:

- таблица ``фраза × модель × mase_median`` (:func:`run_backtest`);
- сводная по моделям (:func:`summarize_backtest`): медиана MASE по фразам и
  доля побед над сезонным наивным бейзлайном — доля фраз, где медианный MASE
  модели СТРОГО меньше медианного MASE бейзлайна на той же фразе.

Победа над бейзлайном сравнивается с фактическим MASE бейзлайна на той же
фразе и тех же фолдах, а не с константой 1.0: MASE масштабируется наивом
на обучающей части, и на тестовых фолдах бейзлайн сам может быть и лучше,
и хуже единицы.

Никаких тихих пропусков: пустой список фраз, слишком короткий ряд (нет ни
одного фолда) или отсутствие фразы бейзлайна в результате — ``BacktestError``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wordstat_trends.forecasting.baseline import INITIAL_WINDOW, TEST_LENGTH
from wordstat_trends.forecasting.models import MODEL_FACTORIES, evaluate_models

#: Имя бейзлайна в MODEL_FACTORIES — первый по порядку, с ним сравниваются остальные.
BASELINE_MODEL = "seasonal_naive"

#: Колонки длинной таблицы ``фраза × модель × MASE`` — стабильный формат CSV.
TABLE_COLUMNS = ["phrase", "model", "mase_median", "n_folds"]

#: Колонки сводной таблицы по моделям.
SUMMARY_COLUMNS = ["model", "mase_median", "win_rate_vs_baseline", "n_phrases"]


class BacktestError(ValueError):
    """Бэктест не может быть честно выполнен: пустой список фраз, короткий
    ряд (ни одного фолда) или отсутствие бейзлайна в результате."""


def run_backtest(
    series_by_phrase: dict[str, pd.Series],
    splitter=None,
) -> pd.DataFrame:
    """Прогнать все модели по ВСЕМ фразам через общий каркас валидации.

    Для каждой фразы вызывается :func:`~wordstat_trends.forecasting.models.evaluate_models`
    (бейзлайн + Theta + AutoETS + AutoARIMA на одних фолдах); медианный MASE
    каждой модели попадает в строку длинной таблицы ``фраза × модель``.
    Возвращает DataFrame с колонками :data:`TABLE_COLUMNS`, отсортированный
    по фразе; внутри фразы модели идут в порядке ``MODEL_FACTORIES``
    (бейзлайн первым).
    """

    if not series_by_phrase:
        raise BacktestError("пустой список фраз: бэктест не имеет смысла без данных")

    # Слишком короткий ряд не даст ни одного фолда — ловим ДО прогона, с
    # перечнем всех проблемных фраз сразу, а не первой по счёту.
    too_short = {
        phrase: len(series)
        for phrase, series in series_by_phrase.items()
        if len(series) < INITIAL_WINDOW + TEST_LENGTH
    }
    if too_short:
        details = ", ".join(f"{p!r}: {n} < {INITIAL_WINDOW + TEST_LENGTH}" for p, n in sorted(too_short.items()))
        raise BacktestError(f"ряды короче минимального окна ({INITIAL_WINDOW}+{TEST_LENGTH} точек): {details}")

    rows: list[dict] = []
    for phrase in sorted(series_by_phrase):
        per_model = evaluate_models(series_by_phrase[phrase], splitter)
        if BASELINE_MODEL not in per_model.index:
            raise BacktestError(
                f"фраза {phrase!r}: бейзлайн {BASELINE_MODEL!r} отсутствует в результате evaluate_models"
            )
        fold_cols = [c for c in per_model.columns if c.startswith("fold_")]
        for model in per_model.index:  # порядок MODEL_FACTORIES: бейзлайн первым
            rows.append(
                {
                    "phrase": phrase,
                    "model": model,
                    "mase_median": float(per_model.loc[model, "mase_median"]),  # type: ignore[arg-type]
                    "n_folds": len(fold_cols),
                }
            )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def summarize_backtest(table: pd.DataFrame) -> pd.DataFrame:
    """Сводная по моделям: медиана MASE по фразам + доля побед над бейзлайном.

    Победа на фразе — ``mase_median`` модели строго меньше ``mase_median``
    бейзлайна (``seasonal_naive``) на той же фразе, из тех же фолдов. У самого
    бейзлайна доля побед по построению 0.0 (строгое неравенство с самим собой
    никогда не выполняется). Модели идут в порядке ``MODEL_FACTORIES``.
    """

    if table.empty:
        raise BacktestError("пустая таблица бэктеста: нечего суммаризировать")
    baseline = table[table["model"] == BASELINE_MODEL].set_index("phrase")["mase_median"]
    if baseline.empty:
        raise BacktestError(f"в таблице нет строк бейзлайна {BASELINE_MODEL!r} — доля побед не определена")
    # Каждая фраза таблицы обязана иметь строку бейзлайна: без неё доля побед
    # не определена, и тихий NaN скрыл бы кривую таблицу (собранную не
    # run_backtest). Поэтому проверка — громкая, а не NaN в результате.
    orphaned = sorted(set(table["phrase"]) - set(baseline.index))
    if orphaned:
        raise BacktestError(f"фразы без строки бейзлайна {BASELINE_MODEL!r}: {orphaned}")

    rows: list[dict] = []
    for model in MODEL_FACTORIES:
        part = table[table["model"] == model]
        if part.empty:
            continue
        wins = sum(
            float(row.mase_median) < float(baseline.loc[row.phrase])  # type: ignore[arg-type]
            for row in part.itertuples()
        )
        rows.append(
            {
                "model": model,
                "mase_median": float(part["mase_median"].median()),  # type: ignore[arg-type]
                "win_rate_vs_baseline": wins / len(part),
                "n_phrases": int(len(part)),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def write_backtest_report(table: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Записать таблицу и сводную в CSV (utf-8-sig, как выгрузки Вордстата).

    Возвращает пути ``(таблица, сводная)``. Каталог создаётся при отсутствии.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "backtest_mase.csv"
    summary_path = out_dir / "backtest_mase_summary.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return table_path, summary_path


__all__ = [
    "BASELINE_MODEL",
    "SUMMARY_COLUMNS",
    "TABLE_COLUMNS",
    "BacktestError",
    "run_backtest",
    "summarize_backtest",
    "write_backtest_report",
]

"""Генерация таблицы результатов бэктеста для README (issue #97, фаза 7.4).

Числа в README не рукописные: скрипт читает артефакт бэктеста
``outputs/backtest_mase_summary.csv`` (пишет ``scripts/backtest_forecasters.py``)
и заменяет в README блок между маркерами ``<!-- results:begin -->`` и
``<!-- results:end -->`` на Markdown-таблицу «модель × MASE» с главным ответом
(лучшая модель и её преимущество над наивной). Блок идемпотентен: повторный
запуск по тому же артефакту не меняет README, по новому — обновляет числа,
так README не разъезжается с фактом.

Всё, что не выводится из сводной одной строки (проигрыши на отдельных фолдах,
ограничения датасета), остаётся в README рукописным текстом со ссылкой на
честный отчёт ``docs/ISSUE_11_FORECAST_REPORT.md`` — скрипт его не трогает.

Запуск (после ``scripts/backtest_forecasters.py <run-dir> ...``):

    python scripts/render_readme_results.py

Stdlib только: артефакт — маленький CSV, pandas не нужен.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Как в scripts/backtest_forecasters.py: скрипт должен работать без установки
# пакета в окружение — достаточно дерева репозитория.
sys.path.insert(0, str(REPO_ROOT / "src"))

from wordstat_trends.forecasting.backtest import BASELINE_MODEL, SUMMARY_COLUMNS  # noqa: E402

DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "backtest_mase_summary.csv"
DEFAULT_README = REPO_ROOT / "README.md"

BEGIN_MARKER = "<!-- results:begin -->"
END_MARKER = "<!-- results:end -->"

#: Ключи MODEL_FACTORIES → человеческие имена моделей в таблице (issue #97).
MODEL_LABELS = {
    "seasonal_naive": "Naive seasonal (бейзлайн)",
    "theta": "Theta",
    "auto_ets": "AutoETS",
    "auto_arima": "AutoARIMA",
}


class RenderError(ValueError):
    """Артефакт или README не позволяют собрать блок честно."""


def load_summary(path: Path) -> list[dict[str, str]]:
    """Прочитать сводную CSV (utf-8-sig, как пишут выгрузки)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RenderError(f"пустая сводная {path}: сначала прогоните scripts/backtest_forecasters.py")
    missing = set(SUMMARY_COLUMNS) - set(rows[0])
    if missing:
        raise RenderError(f"в сводной {path} нет колонок {sorted(missing)}")
    return rows


def render_block(summary: list[dict[str, str]]) -> str:
    """Сводная → Markdown-блок: таблица моделей + главный ответ первым абзацем.

    Лучшая модель — минимальная медиана MASE (жирным в таблице); её
    преимущество над наивной — отношение медианы бейзлайна к её медиане.
    """

    def number(row: dict[str, str], column: str) -> float:
        try:
            return float(row[column])
        except (TypeError, ValueError) as exc:
            raise RenderError(
                f"модель {row.get('model')!r}: колонка {column!r} не число — {row.get(column)!r}"
            ) from exc

    def mase(row: dict[str, str]) -> float:
        return number(row, "mase_median")

    baseline = next((r for r in summary if r["model"] == BASELINE_MODEL), None)
    if baseline is None:
        raise RenderError(f"в сводной нет строки бейзлайна {BASELINE_MODEL!r}")
    best = min(summary, key=mase)
    if best["model"] == BASELINE_MODEL:
        raise RenderError("бейзлайн оказался лучшей моделью — «преимущество над наивной» не определено")
    ratio = mase(baseline) / mase(best)
    best_label = MODEL_LABELS.get(best["model"], best["model"])

    lines = [
        f"Лучшая модель — **{best_label}**: медианный MASE {mase(best):.3f} против "
        f"{mase(baseline):.3f} у сезонного наива — **в {ratio:.1f} раза меньше ошибка** "
        f"на доступном срезе.",
        "",
        "| Модель | Медиана MASE | Доля побед над бейзлайном |",
        "|---|---:|---:|",
    ]
    for row in summary:
        label = MODEL_LABELS.get(row["model"], row["model"])
        name = f"**{label}**" if row is best else label
        lines.append(f"| {name} | {mase(row):.3f} | {number(row, 'win_rate_vs_baseline'):.1f} |")
    return "\n".join(lines)


def replace_block(readme_text: str, block: str) -> str:
    """Заменить содержимое между маркерами; маркеры остаются в файле."""
    begin = readme_text.find(BEGIN_MARKER)
    end = readme_text.find(END_MARKER)
    if begin == -1 or end == -1 or end < begin:
        raise RenderError(f"в README нет пары маркеров {BEGIN_MARKER} … {END_MARKER}")
    head = readme_text[: begin + len(BEGIN_MARKER)]
    tail = readme_text[end:]
    return f"{head}\n{block}\n{tail}"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 2:
        print("usage: python scripts/render_readme_results.py [summary.csv] [README.md]", file=sys.stderr)
        return 2
    summary_path = Path(args[0]) if args else DEFAULT_SUMMARY
    readme_path = Path(args[1]) if len(args) > 1 else DEFAULT_README

    summary = load_summary(summary_path)
    updated = replace_block(readme_path.read_text(encoding="utf-8"), render_block(summary))
    readme_path.write_text(updated, encoding="utf-8")
    print(f"README обновлён из {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Тесты для scripts/render_readme_results.py (issue #97, фаза 7.4)."""

import pytest

from scripts.render_readme_results import (
    BEGIN_MARKER,
    END_MARKER,
    RenderError,
    load_summary,
    render_block,
    replace_block,
)

# Мини-копия сводной из outputs/backtest_mase_summary.csv (прогон 2026-09-09,
# docs/ISSUE_11_FORECAST_REPORT.md): порядок и имена моделей — MODEL_FACTORIES.
SUMMARY_CSV = "model,mase_median,win_rate_vs_baseline,n_phrases\n" + "\n".join(
    [
        "seasonal_naive,0.7591332566755493,0.0,2",
        "theta,0.6767000702281589,1.0,2",
        "auto_ets,0.3025162983917731,1.0,2",
        "auto_arima,0.32087266776069095,1.0,2",
    ]
) + "\n"

README_TEMPLATE = f"""# wordstat-trends

{BEGIN_MARKER}
старый блок
{END_MARKER}

## Дальше
"""


@pytest.fixture()
def summary_file(tmp_path):
    path = tmp_path / "backtest_mase_summary.csv"
    path.write_text(SUMMARY_CSV, encoding="utf-8-sig")
    return path


def test_render_block_best_model_and_ratio(summary_file):
    # Главный ответ: лучшая модель, её MASE, MASE бейзлайна и отношение —
    # всё из артефакта, ничего рукописного.
    block = render_block(load_summary(summary_file))
    assert "**AutoETS**" in block
    assert "0.303" in block and "0.759" in block
    assert "в 2.5 раза" in block
    # Таблица: все четыре модели, бейзлайн первым, лучшая — жирным.
    assert "| Naive seasonal (бейзлайн) | 0.759 | 0.0 |" in block
    assert "| Theta | 0.677 | 1.0 |" in block
    assert "| **AutoETS** | 0.303 | 1.0 |" in block
    assert "| AutoARIMA | 0.321 | 1.0 |" in block


def test_replace_block_is_idempotent():
    # Повторная замена того же блока не меняет файл — README можно
    # регенерировать после каждого бэктеста без диффа-шума.
    once = replace_block(README_TEMPLATE, "новый блок")
    assert replace_block(once, "новый блок") == once
    assert "старый блок" not in once
    assert BEGIN_MARKER in once and END_MARKER in once


def test_replace_block_without_markers_fails_loudly():
    # README без пары маркеров — ошибка, а не тихое «нечего обновлять».
    with pytest.raises(RenderError, match="маркеров"):
        replace_block("# README без маркеров\n", "блок")


def test_load_summary_rejects_bad_artifact(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("model,mase_median\n", encoding="utf-8-sig")
    with pytest.raises(RenderError, match="пустая сводная"):
        load_summary(empty)
    wrong_cols = tmp_path / "wrong.csv"
    wrong_cols.write_text("a,b\n1,2\n", encoding="utf-8-sig")
    with pytest.raises(RenderError, match="колонок"):
        load_summary(wrong_cols)


def test_render_block_without_baseline_fails_loudly(tmp_path):
    no_baseline = tmp_path / "no_baseline.csv"
    no_baseline.write_text(
        "model,mase_median,win_rate_vs_baseline,n_phrases\ntheta,0.5,1.0,1\n", encoding="utf-8-sig"
    )
    with pytest.raises(RenderError, match="бейзлайна"):
        render_block(load_summary(no_baseline))


def test_render_block_garbage_number_fails_with_model_named(tmp_path):
    # Мусор в числовой колонке — RenderError с моделью и колонкой,
    # а не голый ValueError без контекста.
    garbage = tmp_path / "garbage.csv"
    garbage.write_text(
        "model,mase_median,win_rate_vs_baseline,n_phrases\n"
        "seasonal_naive,0.8,0.0,1\ntheta,не-число,1.0,1\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(RenderError, match="'theta'.*'mase_median'.*'не-число'"):
        render_block(load_summary(garbage))

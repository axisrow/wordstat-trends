"""Тесты шаблонов объяснений (issue #20, фаза 5).

Подстановку проверяем на фактических структурах выхода ranking/growth:
фикстуры tests/fixtures → phrase_record → rank_showcase → explain_phrase.
Числа в шаблонах — уже отформатированные по локали (format_ratio на
реальном ``GrowthScore.score``), падежи и порядок слов держит шаблон.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.explain import ExplainError, explain_phrase, localized_class
from wordstat_trends.forecasting.baseline import to_monthly_series
from wordstat_trends.i18n import format_ratio
from wordstat_trends.loader import parse_dynamics_rows
from wordstat_trends.trends.ranking import (
    PhraseRecord,
    TrendClass,
    phrase_record,
    rank_showcase,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_series(csv_name: str) -> pd.Series:
    rows = parse_dynamics_rows((FIXTURES / csv_name).read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows, columns=["period", "queries", "share_pct"])
    return to_monthly_series(frame)


def _showcase_with_growth() -> dict[str, tuple]:
    """Витрина фикстур issue #85 + скор детекции каждой фразы."""

    records: list[PhraseRecord] = [
        phrase_record(_fixture_series("dynamics_seasonal.csv"), "новогодние подарки"),
        phrase_record(_fixture_series("dynamics_high_freq.csv"), "купить телефон"),
    ]
    pattern = _fixture_series("dynamics_seasonal.csv").iloc[:12].to_numpy()
    flat = pd.Series(
        np.tile(pattern.astype(float), 3),
        index=pd.PeriodIndex(pd.period_range("2024-01", periods=36, freq="M"), freq="M"),
        name="queries",
    )
    flat.iloc[-3:] = flat.iloc[-3:] * 2.0
    records.append(phrase_record(flat, "растущая синтетика"))

    showcase = rank_showcase(records)
    growths = {r.phrase: r.growth for r in records}
    return {row.phrase: (row, growths[row.phrase]) for row in showcase}


def test_explain_growing_real_ratio_both_locales():
    ranked, growth = _showcase_with_growth()["растущая синтетика"]
    assert ranked.klass is TrendClass.GROWING
    ru = explain_phrase(ranked, growth, "ru")
    zh = explain_phrase(ranked, growth, "zh")
    # Скор реальной синтетики — ровно 2.0; число заходит в шаблон уже
    # отформатированным по локали, сырого float в тексте нет.
    assert f"в {format_ratio(growth.score, 'ru')} раза" in ru
    assert f"增长{format_ratio(growth.score, 'zh')}倍" in zh
    assert "2,0" in ru and "2.0" in zh


def test_explain_every_class_renders_template():
    for phrase, (ranked, growth) in _showcase_with_growth().items():
        for locale in ("ru", "zh"):
            text = explain_phrase(ranked, growth, locale)
            assert text, (phrase, locale)
            # «рост идёт {months} мес.» — возраст из RankedPhrase.components
            if ranked.klass is TrendClass.GROWING:
                assert str(ranked.components.growth_age_months) in text


def test_explain_falling_uses_ratio_from_growth_score():
    ranked, growth = _showcase_with_growth()["купить телефон"]
    if ranked.klass is not TrendClass.FALLING:
        pytest.skip("фикстура «купить телефон» стабильна — падения нет")
    assert format_ratio(growth.score, "ru") in explain_phrase(ranked, growth, "ru")


def test_localized_class_labels_match_trend_values():
    for klass in TrendClass:
        assert localized_class(klass, "ru") == klass.value
        assert localized_class(klass, "zh")  # метка непустая


def test_missing_key_raises(tmp_path):
    ranked, growth = _showcase_with_growth()["растущая синтетика"]
    with pytest.raises(ExplainError, match="explain.growing"):
        explain_phrase(ranked, growth, "ru", messages={})


def test_missing_placeholder_raises_explain_error():
    # Ревью PR #108: GROWING с components=None (допускается guard-ом) —
    # плейсхолдер {months} без значения обязан дать ExplainError,
    # а не сырой KeyError из str.format.
    ranked, growth = _showcase_with_growth()["растущая синтетика"]
    stripped = replace(ranked, components=None)
    messages = {"explain.growing": "рост к прогнозу в {ratio} раза, рост идёт {months} мес."}
    with pytest.raises(ExplainError, match="months"):
        explain_phrase(stripped, growth, "ru", messages=messages)


def test_format_ratio_locale_separators_on_real_scores():
    for _, growth in _showcase_with_growth().values():
        ru = format_ratio(growth.score, "ru")
        zh = format_ratio(growth.score, "zh")
        assert ru.replace(",", ".") == zh
        assert "," in ru or "." in ru

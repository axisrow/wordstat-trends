"""Детекция, отсев и ранжирование трендов поверх прогноза (фаза 3, группа Б — issue #12).

Детекция, отсев и ранжирование живут здесь, поверх моделей фазы 2
(:mod:`wordstat_trends.forecasting`) — модели не детектируют ничего
сами.
"""

from wordstat_trends.trends.growth import (
    GROWTH_RATIO_THRESHOLD,
    MIN_HISTORY,
    SCORE_WINDOW,
    GrowthScore,
    InsufficientHistoryError,
    score_frame,
    score_growth,
)
from wordstat_trends.trends.ranking import (
    ANOMALY_WEIGHT,
    CLASS_ORDER,
    DECLINE_RATIO_THRESHOLD,
    FRESHNESS_WEIGHT,
    SEASONAL_MAX_MIN_RATIO,
    VOLUME_TOP,
    VOLUME_WEIGHT,
    PhraseRecord,
    RankedPhrase,
    ScoreComponents,
    TrendClass,
    classify,
    growth_age_months,
    growth_score_components,
    phrase_record,
    phrase_record_from_frame,
    rank_showcase,
    seasonal_max_min_ratio,
)

__all__ = [
    "ANOMALY_WEIGHT",
    "CLASS_ORDER",
    "DECLINE_RATIO_THRESHOLD",
    "FRESHNESS_WEIGHT",
    "GROWTH_RATIO_THRESHOLD",
    "MIN_HISTORY",
    "SCORE_WINDOW",
    "SEASONAL_MAX_MIN_RATIO",
    "VOLUME_TOP",
    "VOLUME_WEIGHT",
    "GrowthScore",
    "InsufficientHistoryError",
    "PhraseRecord",
    "RankedPhrase",
    "ScoreComponents",
    "TrendClass",
    "classify",
    "growth_age_months",
    "growth_score_components",
    "phrase_record",
    "phrase_record_from_frame",
    "rank_showcase",
    "score_frame",
    "score_growth",
    "seasonal_max_min_ratio",
]

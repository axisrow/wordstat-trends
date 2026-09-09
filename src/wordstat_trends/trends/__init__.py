"""Детекция трендов поверх прогноза (фаза 3, группа Б — issue #12).

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

__all__ = [
    "GROWTH_RATIO_THRESHOLD",
    "MIN_HISTORY",
    "SCORE_WINDOW",
    "GrowthScore",
    "InsufficientHistoryError",
    "score_frame",
    "score_growth",
]

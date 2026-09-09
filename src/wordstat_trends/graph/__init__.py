"""Обход графа запросов Вордстата (issue #81, Фаза 3 / группа А).

Ядро на абстрактном провайдере рёбер; интеграция со сбором — #82.
"""

from wordstat_trends.graph.traverse import (
    SEED_FREQUENCY,
    EdgeProvider,
    FrontierEntry,
    RelatedPhrase,
    TraversalLimits,
    TraversalResult,
    TraversalState,
    run_traversal,
)

__all__ = [
    "EdgeProvider",
    "FrontierEntry",
    "RelatedPhrase",
    "SEED_FREQUENCY",
    "TraversalLimits",
    "TraversalResult",
    "TraversalState",
    "run_traversal",
]

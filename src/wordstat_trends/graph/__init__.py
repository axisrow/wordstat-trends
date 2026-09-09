"""Обход графа запросов Вордстата (issue #81, Фаза 3 / группа А).

Ядро на абстрактном провайдере рёбер; интеграция со сбором — #82: реальный
провайдер поверх экспортов wordstat-cli, персистентное состояние и бюджет
фраз (``edges``/``state``/``schedule``).
"""

from wordstat_trends.graph.edges import ExportEdgeProvider
from wordstat_trends.graph.schedule import GRAPH_MAX_DEPTH, GRAPH_MAX_PHRASES, pick_pending
from wordstat_trends.graph.state import GraphStateError, GraphStateFile
from wordstat_trends.graph.traverse import (
    SEED_FREQUENCY,
    EdgeProvider,
    EdgesNotAvailable,
    FrontierEntry,
    RelatedPhrase,
    TraversalLimits,
    TraversalResult,
    TraversalState,
    run_traversal,
)

__all__ = [
    "EdgeProvider",
    "EdgesNotAvailable",
    "ExportEdgeProvider",
    "FrontierEntry",
    "GRAPH_MAX_DEPTH",
    "GRAPH_MAX_PHRASES",
    "GraphStateError",
    "GraphStateFile",
    "RelatedPhrase",
    "SEED_FREQUENCY",
    "TraversalLimits",
    "TraversalResult",
    "TraversalState",
    "pick_pending",
    "run_traversal",
]

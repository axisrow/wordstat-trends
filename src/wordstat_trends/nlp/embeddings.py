"""Эмбеддинги фраз моделью sergeyzh/BERTA (issue #69, Фаза 4.2).

BERTA — дистилляция ai-forever/FRIDA (0.1B, 768 dim, MIT): близкое качество
при на порядок меньшем размере. Модель instruction-tuned: вход обязан нести
префикс задачи (`clustering: <фраза>` / `categorize: <фраза>`) — без него
качество заметно падает, поэтому префикс не деталь вызова, а обязательный
параметр `task`.

`sentence-transformers`/`torch` (~2 ГБ) живут только в extra `nlp`:
импорт ленивый, базовый набор зависимостей библиотеку не тянет. Это локальные
модели, а не LLM-клиенты — гейт `scripts/check_no_llm.py` их сознательно
не запрещает (комментарий у FORBIDDEN_PACKAGES).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np

Task = Literal["clustering", "categorize"]

MODEL_NAME = "sergeyzh/BERTA"
EMBEDDING_DIM = 768

# Префикс задачи — часть контракта модели, не оформление строки.
_TASK_PREFIXES: dict[Task, str] = {"clustering": "clustering: ", "categorize": "categorize: "}

# Защита от деления на ноль для нулевых строк (пустые фразы после фильтров).
_EPS = 1e-12


@lru_cache(maxsize=1)
def _model():
    """Единственный экземпляр модели на процесс (загрузка весов — дорогое)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - тривиальная ветка, проверена тестом
        raise ImportError(
            "эмуляции нужны sentence-transformers и torch из extra `nlp`: "
            "uv sync --extra nlp"
        ) from error
    return SentenceTransformer(MODEL_NAME)


def embed_phrases(phrases: list[str], task: Task) -> np.ndarray:
    """Эмбеддинги фраз с префиксом задачи: матрица (len(phrases), 768), строки L2-нормированы.

    Пустой список легален и модели не касается: (0, 768) — корректный вход
    для последующих шагов (PCA, кластеризация) без особых случаев у них.
    """
    if task not in _TASK_PREFIXES:
        raise ValueError(f"неизвестная задача {task!r}: ожидается одна из {sorted(_TASK_PREFIXES)}")
    if not phrases:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    prefixed = [_TASK_PREFIXES[task] + phrase for phrase in phrases]
    embeddings = np.asarray(_model().encode(prefixed), dtype=np.float32)
    if embeddings.shape != (len(phrases), EMBEDDING_DIM):
        raise ValueError(
            f"модель вернула неожиданную форму {embeddings.shape}: "
            f"ожидалась ({len(phrases)}, {EMBEDDING_DIM})"
        )

    # L2-нормализация здесь, а не в encode(normalize_embeddings=True):
    # контракт «единичные строки» принадлежит модулю, а не настройке вызова.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, _EPS)

"""Тесты эмбеддингов BERTA (issue #69, Фаза 4.2) — без сети.

`SentenceTransformer.encode` подменяется фейком через sys.modules: настоящий
sentence-transformers (~2 ГБ вместе с torch) в CI не ставится — он в extra
`nlp`. Сетевой smoke — отдельный маркер `network`, в CI не входит.
"""

import os
import sys
import types

import numpy as np
import pytest

from wordstat_trends.nlp.embeddings import EMBEDDING_DIM, MODEL_NAME, embed_phrases


class _FakeSentenceTransformer:
    """Запоминает вход encode и отдаёт детерминированные ненормированные векторы."""

    def __init__(self, model_name: str):
        assert model_name == MODEL_NAME
        self.encoded: list[str] | None = None

    def encode(self, sentences: list[str]) -> np.ndarray:
        self.encoded = list(sentences)
        # Ненормированные векторы: нормализация — обязанность embed_phrases.
        rng = np.random.default_rng(42)
        return rng.random((len(sentences), EMBEDDING_DIM), dtype=np.float32) * 10.0


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch):
    """Подменяет sentence_transformers в sys.modules до ленивого импорта."""
    instance = _FakeSentenceTransformer(MODEL_NAME)
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = lambda name: instance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    # _model кешируется на процесс — сбрасываем до и после, чтобы фейк
    # не протёк в другие тесты, а чужой закешированный экземпляр — в эти.
    import wordstat_trends.nlp.embeddings as emb

    emb._model.cache_clear()
    yield instance
    emb._model.cache_clear()


def test_task_prefix_passed_to_model(fake_model):
    embed_phrases(["купить телефон", "курсы английского"], task="clustering")
    assert fake_model.encoded == [
        "clustering: купить телефон",
        "clustering: курсы английского",
    ]


def test_categorize_prefix_used_for_categorize_task(fake_model):
    embed_phrases(["новогодние подарки"], task="categorize")
    assert fake_model.encoded == ["categorize: новогодние подарки"]


def test_result_shape_is_n_phrases_by_dim(fake_model):
    result = embed_phrases(["a", "b", "c"], task="clustering")
    assert result.shape == (3, EMBEDDING_DIM)


def test_rows_are_l2_normalized(fake_model):
    result = embed_phrases(["a", "b", "c", "d"], task="clustering")
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_normalization_actually_changes_vectors(fake_model):
    # Фейк отдаёт векторы с нормой ~10*sqrt(768/3): если нормализация
    # сломается, строки не станут единичными — проверка выше это поймает,
    # а здесь фиксируем, что вход модели действительно был ненормирован.
    raw = _FakeSentenceTransformer(MODEL_NAME).encode(["x"])
    assert abs(np.linalg.norm(raw[0]) - 1.0) > 1.0


def test_empty_phrase_list_gives_empty_matrix_without_model(fake_model):
    result = embed_phrases([], task="clustering")
    assert result.shape == (0, EMBEDDING_DIM)
    assert fake_model.encoded is None


def test_unknown_task_rejected_before_model_call(fake_model):
    with pytest.raises(ValueError, match="неизвестная задача"):
        embed_phrases(["фраза"], task="summarize")  # type: ignore[arg-type]
    assert fake_model.encoded is None


def test_missing_extra_raises_helpful_import_error(monkeypatch: pytest.MonkeyPatch):
    import wordstat_trends.nlp.embeddings as emb

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # None -> ImportError при импорте
    emb._model.cache_clear()
    with pytest.raises(ImportError, match="extra `nlp`"):
        embed_phrases(["фраза"], task="clustering")
    emb._model.cache_clear()


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("WORDSTAT_TRENDS_NLP_SMOKE") != "1",
    reason="сетевой smoke: WORDSTAT_TRENDS_NLP_SMOKE=1 и extra nlp (uv sync --extra nlp)",
)
def test_network_smoke_real_model():
    """Живой прогон sergeyzh/BERTA: форма, нормализация — против Hugging Face."""
    result = embed_phrases(["купить телефон", "новогодние подарки"], task="clustering")
    assert result.shape == (2, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)

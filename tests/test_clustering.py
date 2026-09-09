"""Тесты кластеризации фраз в ниши (issue #71, Фаза 4.4) — без сети.

BERTA-конвейер проверяется на фейке вместо sentence-transformers (как в
tests/test_embeddings.py): фейковый encode строит детерминированные векторы
из лемм (хеш леммы + сумма), поэтому фразы с общими леммами сближаются —
можно проверить группировку end-to-end, не качая ~2 ГБ extra `nlp`.
TF-IDF-конвейер полностью настоящий (sklearn + pymorphy3 — основные
зависимости) и группирует настоящие фразы без каких-либо фейков.
"""

import hashlib
import sys
import types

import numpy as np
import pytest

from wordstat_trends.nlp.clustering import DEFAULT_K_GRID, SEED, Cluster, ClusteringResult, cluster_phrases
from wordstat_trends.nlp.embeddings import EMBEDDING_DIM
from wordstat_trends.nlp.lemmatize import lemmatized_phrase

# Три профильные seed-фразы фикстур Фазы 1 (dynamics_*.csv) — три разные темы.
FIXTURE_PHRASES = ["новогодние подарки", "купить телефон", "курсы английского"]

# Синтетические ниши: общая лемма («английский») должна собрать их в одну
# тему рядом с «курсы английского» из фикстур.
ENGLISH_NICHE = ["репетитор английского", "выучить английский"]

# Хеш леммы → 8 байт, растирание в EMBEDDING_DIM повторением блока.
_LEMMA_BYTES = 8


class _FakeSentenceTransformer:
    """encode → детерминированные векторы из лемм: хеш леммы + сумма по фразе.

    Не претендует на качество BERTA — только на детерминизм и содержательную
    близость фраз с общими леммами (её достаточно для проверки группировки).
    """

    def __init__(self, model_name: str):
        self.encoded: list[str] | None = None

    @staticmethod
    def _lemma_vector(lemma: str) -> np.ndarray:
        digest = hashlib.md5(lemma.encode("utf-8")).digest()[:_LEMMA_BYTES]  # noqa: S324 — не криптография
        return np.tile(
            np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0,
            EMBEDDING_DIM // _LEMMA_BYTES,
        )

    def encode(self, sentences: list[str]) -> np.ndarray:
        self.encoded = list(sentences)
        vectors = np.zeros((len(sentences), EMBEDDING_DIM), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            for token in sentence.split():
                vectors[i] += self._lemma_vector(token)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch):
    """Подменяет sentence_transformers в sys.modules до ленивого импорта."""
    import wordstat_trends.nlp.embeddings as emb

    instance = _FakeSentenceTransformer("sergeyzh/BERTA")
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = lambda name: instance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    emb._model.cache_clear()
    yield instance
    emb._model.cache_clear()


def phrases_of(result: ClusteringResult) -> list[list[str]]:
    return [cluster.phrases for cluster in result.clusters]


class TestTfidfPipeline:
    """Лёгкий fallback: sklearn + pymorphy3, без extra `nlp` и без фейков."""

    def test_english_niche_merges_with_fixture_phrase(self):
        # «репетитор/выучить английский» + «курсы английского» — одна тема;
        # покупать телефон и новогодние подарки — не в ней.
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="tfidf")
        groups = phrases_of(result)
        english = [group for group in groups if "курсы английского" in group]
        assert len(english) == 1
        assert set(ENGLISH_NICHE) <= set(english[0])
        assert "купить телефон" not in english[0]

    def test_fixture_phrases_end_up_in_different_clusters(self):
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="tfidf")
        flat = [phrase for group in phrases_of(result) for phrase in group]
        assert set(flat) == set(FIXTURE_PHRASES + ENGLISH_NICHE)
        # фразы без общих лемм не должны попасть в один кластер
        assert result.n_clusters >= 2

    def test_topic_label_is_top_lemmas(self):
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="tfidf")
        english = next(cluster for cluster in result.clusters if "курсы английского" in cluster.phrases)
        assert "английский" in english.top_lemmas
        assert english.topic == ", ".join(english.top_lemmas)

    def test_seed_fixed_result_is_deterministic(self):
        phrases = FIXTURE_PHRASES + ENGLISH_NICHE
        first = cluster_phrases(phrases, pipeline="tfidf")
        second = cluster_phrases(phrases, pipeline="tfidf")
        assert phrases_of(first) == phrases_of(second)
        assert [c.top_lemmas for c in first.clusters] == [c.top_lemmas for c in second.clusters]


class TestBertaPipeline:
    """Основной конвейер: проверяется на детерминированном фейке encode."""

    def test_task_prefix_passed_to_model(self, fake_model):
        cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="berta")
        # вход модели — лемматизированные фразы с префиксом задачи clustering
        assert fake_model.encoded == ["clustering: " + lemmatized_phrase(p) for p in FIXTURE_PHRASES + ENGLISH_NICHE]

    def test_english_niche_merges_on_fake_embeddings(self, fake_model):
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="berta")
        groups = phrases_of(result)
        english = [group for group in groups if "курсы английского" in group]
        assert len(english) == 1
        assert set(ENGLISH_NICHE) <= set(english[0])

    def test_silhouette_is_reported(self, fake_model):
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="berta")
        assert result.n_clusters >= 2
        assert result.silhouette is not None
        assert -1.0 <= result.silhouette <= 1.0

    def test_default_pipeline_is_berta(self, fake_model):
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE)
        assert result.pipeline == "berta"


class TestKSelection:
    """Число кластеров — подбор по силуэту по сетке."""

    def test_best_k_beats_others_by_silhouette(self, fake_model):
        phrases = FIXTURE_PHRASES + ENGLISH_NICHE
        result = cluster_phrases(phrases, pipeline="berta")
        scores = {}
        for k in DEFAULT_K_GRID:
            if not 2 <= k <= len(phrases) - 1:
                continue
            scores[k] = cluster_phrases(phrases, pipeline="berta", k_grid=(k,)).silhouette
        # выбранный силуэт — максимум по сетке (равные k дают одинаковый k-means)
        assert result.silhouette == pytest.approx(max(scores.values()))

    def test_custom_grid_respected(self, fake_model):
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="berta", k_grid=(2,))
        assert result.n_clusters == 2

    def test_seed_defaults_to_fixed_value(self):
        assert SEED == 42


class TestDegenerateInputs:
    def test_empty_phrases(self):
        result = cluster_phrases([], pipeline="tfidf")
        assert result.n_phrases == 0
        assert result.n_clusters == 0
        assert result.clusters == []
        assert result.silhouette is None

    def test_single_phrase(self):
        result = cluster_phrases(["курсы английского"], pipeline="tfidf")
        assert result.n_clusters == 1
        assert result.clusters[0].phrases == ["курсы английского"]
        assert result.silhouette is None

    def test_two_phrases_single_cluster(self):
        # сетка 2..8 обрезается до пустой (k <= n-1) → один кластер
        result = cluster_phrases(["купить телефон", "курсы английского"], pipeline="tfidf")
        assert result.n_clusters == 1
        assert result.clusters[0].phrases == ["купить телефон", "курсы английского"]

    def test_identical_phrases_single_cluster(self):
        # одинаковые признаки: k-means может вернуть <2 уникальных меток,
        # силуэт не определён — это один кластер, а не ValueError
        result = cluster_phrases(["курсы английского"] * 5, pipeline="tfidf")
        assert result.n_phrases == 5
        assert result.n_clusters == 1
        assert result.clusters[0].phrases == ["курсы английского"] * 5
        assert result.silhouette is None

    def test_identical_phrases_berta_fake(self, fake_model):
        result = cluster_phrases(["курсы английского"] * 5, pipeline="berta")
        assert result.n_clusters == 1
        assert result.silhouette is None

    def test_phrases_without_lemmas_excluded(self):
        # цифры/пунктуация: лемматизация пуста — не «empty vocabulary», а фильтр
        result = cluster_phrases(["123", "!!!", "купить телефон"], pipeline="tfidf")
        assert result.n_phrases == 1
        assert result.clusters[0].phrases == ["купить телефон"]

    def test_all_phrases_without_lemmas_empty_result(self):
        result = cluster_phrases(["123", "???"], pipeline="tfidf")
        assert result.n_phrases == 0
        assert result.n_clusters == 0
        assert result.clusters == []

    def test_unknown_pipeline_rejected(self):
        with pytest.raises(ValueError, match="неизвестный конвейер"):
            cluster_phrases(FIXTURE_PHRASES, pipeline="fasttext")  # type: ignore[arg-type]


class TestMissingExtra:
    def test_berta_without_nlp_extra_raises_helpful_import_error(self, monkeypatch: pytest.MonkeyPatch):
        import wordstat_trends.nlp.embeddings as emb

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # None -> ImportError при импорте
        emb._model.cache_clear()
        with pytest.raises(ImportError, match="extra `nlp`"):
            cluster_phrases(FIXTURE_PHRASES, pipeline="berta")
        emb._model.cache_clear()

    def test_tfidf_works_without_nlp_extra(self, monkeypatch: pytest.MonkeyPatch):
        # лёгкий fallback не должен касаться sentence-transformers вовсе
        import wordstat_trends.nlp.clustering as clustering_module

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        monkeypatch.setattr(clustering_module, "_berta_features", None)  # если бы вызывался — упал бы TypeError
        result = cluster_phrases(FIXTURE_PHRASES + ENGLISH_NICHE, pipeline="tfidf")
        assert result.n_clusters >= 1


class TestClusterDataclass:
    def test_topic_joins_top_lemmas(self):
        cluster = Cluster(top_lemmas=["английский", "курс"], phrases=["курсы английского"])
        assert cluster.topic == "английский, курс"

    def test_member_phrases_keep_input_order(self, fake_model):
        phrases = ENGLISH_NICHE + FIXTURE_PHRASES
        result = cluster_phrases(phrases, pipeline="berta")
        flat = [phrase for cluster in result.clusters for phrase in cluster.phrases]
        assert sorted(flat) == sorted(phrases)
        # внутри кластера фразы идут в порядке появления во входе
        for cluster in result.clusters:
            positions = [phrases.index(phrase) for phrase in cluster.phrases]
            assert positions == sorted(positions)

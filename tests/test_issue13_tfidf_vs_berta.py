"""Тесты эксперимента TF-IDF vs BERTA (issue #70, Фаза 4.3) — без сети.

BERTA подменяется детерминированным фейком (embed_fn инъекцируется в
run_berta_pipeline): настоящий sentence-transformers (~2 ГБ) в CI не
ставится — он в extra `nlp`. TF-IDF-конвейер гоняется честно, целиком.
"""

import numpy as np

from scripts.issue13_tfidf_vs_berta import (
    EMBEDDING_DIM_FAKE,
    build_dataset,
    fake_embed,
    run_berta_pipeline,
    run_tfidf_pipeline,
)

EXPECTED_N_GROUPS = 5
EXPECTED_GROUP_SIZE = 10


def test_dataset_shape_is_expected():
    phrases, expected = build_dataset()
    assert len(phrases) == EXPECTED_N_GROUPS * EXPECTED_GROUP_SIZE
    # Ожидаемые метки — блоки по группе, порядок групп детерминирован.
    assert len(set(expected)) == EXPECTED_N_GROUPS
    assert all(expected.count(group) == EXPECTED_GROUP_SIZE for group in set(expected))


def test_tfidf_pipeline_output_shape():
    phrases, expected = build_dataset()
    result = run_tfidf_pipeline(phrases, expected)
    clustering = result["clustering"]
    assert clustering["k"] == EXPECTED_N_GROUPS
    assert len(clustering["labels"]) == len(phrases)
    assert len(set(clustering["labels"])) == EXPECTED_N_GROUPS
    assert -1.0 <= clustering["silhouette"] <= 1.0
    assert -1.0 <= clustering["ari_vs_expected"] <= 1.0
    # Стабильность: ARI всех пар seed'ов — среднее и минимум.
    assert 0.0 <= clustering["ari_between_seeds_mean"] <= 1.0
    assert 0.0 <= clustering["ari_between_seeds_min"] <= 1.0
    assert len(result["top_terms_by_cluster"]) == EXPECTED_N_GROUPS
    assert len(result["top_phrases_by_cluster"]) == EXPECTED_N_GROUPS


def test_tfidf_pipeline_is_deterministic():
    phrases, expected = build_dataset()
    first = run_tfidf_pipeline(phrases, expected)
    second = run_tfidf_pipeline(phrases, expected)
    assert first["clustering"] == second["clustering"]
    assert first["top_terms_by_cluster"] == second["top_terms_by_cluster"]
    assert first["top_phrases_by_cluster"] == second["top_phrases_by_cluster"]


def test_berta_pipeline_with_fake_embeddings_shape():
    phrases, expected = build_dataset()
    result = run_berta_pipeline(phrases, expected, fake_embed)
    clustering = result["clustering"]
    assert clustering["k"] == EXPECTED_N_GROUPS
    assert len(clustering["labels"]) == len(phrases)
    assert -1.0 <= clustering["silhouette"] <= 1.0
    assert -1.0 <= clustering["ari_vs_expected"] <= 1.0
    assert len(result["top_phrases_by_cluster"]) == EXPECTED_N_GROUPS


def test_berta_pipeline_with_fake_embeddings_is_deterministic():
    phrases, expected = build_dataset()
    assert run_berta_pipeline(phrases, expected, fake_embed) == run_berta_pipeline(phrases, expected, fake_embed)


def test_fake_embeddings_depend_on_phrase_content():
    # Фейк обязан различать фразы — иначе тесты формы/детерминизма проходили
    # бы и на сломанном (константном) представлении.
    a = fake_embed(["курсы английского"], "clustering")
    b = fake_embed(["доставка пиццы"], "clustering")
    assert a.shape == (1, EMBEDDING_DIM_FAKE)
    assert not np.allclose(a, b)

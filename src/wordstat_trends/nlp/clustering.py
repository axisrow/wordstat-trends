"""Кластеризация фраз-кандидатов в ниши (issue #71, Фаза 4.4).

Витрина должна показывать темы, а не сотни отдельных фраз: модуль группирует
найденное в кластеры, каждому даёт метку-тему из топ-лемм состава.

Вход — просто ``list[str]`` фраз. Интеграция с автооткрытием трендов
(issue #12) — интерфейс на будущее: обход графа Wordstat пока заблокирован
багом ``top_related`` в wordstat-cli, поэтому здесь нет ничего, что знает
о происхождении фраз (шов — точка подключения #12 без переделок).

Оба конвейера представления за общим интерфейсом, выбор — одна строка
конфигурации (``pipeline=...``):

- ``"berta"`` (основной, docs/ISSUE_13_TFIDF_VS_BERTA.md: ARI vs expected
  0.948 против 0.470 у TF-IDF) — эмбеддинги sergeyzh/BERTA из extra `nlp`
  через ленивый импорт в ``wordstat_trends.nlp.embeddings``;
- ``"tfidf"`` — лёгкий fallback без extra: TF-IDF по леммам (sklearn +
  pymorphy3, всё в основных зависимостях).

Число кластеров — подбор по силуэту по сетке; seed фиксирован.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from wordstat_trends.nlp.lemmatize import lemmatize

Pipeline = Literal["berta", "tfidf"]

# Сетка k по умолчанию: 2..8 — разумный диапазон тем витрины для одного
# сеанса автооткрытия; обрезается сверху числом фраз (k-means и силуэт
# требуют k <= n-1).
DEFAULT_K_GRID: tuple[int, ...] = tuple(range(2, 9))

# Метка-тема: столько самых частотных лемм состава кластера.
TOP_LABEL_LEMMAS = 3

SEED = 42


@dataclass(frozen=True)
class Cluster:
    """Ниша: метка-тема из топ-лемм и состав (фразы в исходном порядке)."""

    top_lemmas: list[str]
    phrases: list[str]

    @property
    def topic(self) -> str:
        """Метка-тема для витрины: топ-леммы через запятую."""
        return ", ".join(self.top_lemmas)


@dataclass(frozen=True)
class ClusteringResult:
    """Итог кластеризации: выбранное k, силуэт и сами кластеры.

    Силуэт — главный критерий выбора k; None означает вырожденный случай
    (один кластер, силуэт не определён).
    """

    pipeline: Pipeline
    n_phrases: int
    n_clusters: int
    silhouette: float | None
    clusters: list[Cluster] = field(default_factory=list)


def cluster_phrases(
    phrases: list[str],
    pipeline: Pipeline = "berta",
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    seed: int = SEED,
) -> ClusteringResult:
    """Сгруппировать фразы в ниши: выбор конвейера и k — параметры вызова.

    Одинаковый k-means и подбор k по силуэту для обоих представлений;
    метки-тем из топ-лемм состава — независимо от конвейера (у BERTA
    представление не в пространстве лемм, см. ISSUE_13_TFIDF_VS_BERTA.md).
    """
    if not phrases:
        return ClusteringResult(pipeline=pipeline, n_phrases=0, n_clusters=0, silhouette=None)

    lemmatized = [" ".join(lemmatize(phrase)) for phrase in phrases]
    if pipeline == "berta":
        features = _berta_features(lemmatized)
    elif pipeline == "tfidf":
        features = _tfidf_features(lemmatized)
    else:
        raise ValueError(f"неизвестный конвейер {pipeline!r}: ожидается один из ['berta', 'tfidf']")

    labels = _best_kmeans_labels(features, k_grid, seed)
    clusters = _build_clusters(phrases, lemmatized, labels)
    return ClusteringResult(
        pipeline=pipeline,
        n_phrases=len(phrases),
        n_clusters=len(clusters),
        silhouette=_silhouette(features, labels),
        clusters=clusters,
    )


def _berta_features(lemmatized: list[str]) -> np.ndarray:
    """BERTA-эмбеддинги лемматизированных фраз, task="clustering".

    Ленивый импорт extra `nlp` живёт в embeddings.embed_phrases: без extra
    летит ImportError с подсказкой про `uv sync --extra nlp` — здесь добавлять
    к нему нечего, исключение пробрасывается вызывающему (можно уйти
    на pipeline="tfidf").
    """
    from wordstat_trends.nlp.embeddings import embed_phrases

    return embed_phrases(lemmatized, "clustering")


def _tfidf_features(lemmatized: list[str]):
    """TF-IDF по леммам: лемматизация отдельно, векторайзер по пробелам.

    sublinear_tf — типовая настройка для коротких фраз; min_df=1, потому что
    наборы кандидатов бывают маленькими и min_df=2 выкинул бы все признаки.
    """
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=1, sublinear_tf=True)
    return vectorizer.fit_transform(lemmatized)


def _best_kmeans_labels(features, k_grid: tuple[int, ...], seed: int) -> np.ndarray:
    """k-means для каждого валидного k из сетки, итог — метки лучшего по силуэту.

    Детерминизм: один seed (по умолчанию фиксированный SEED), n_init=10,
    tie-break — минимальный k (первый максимум в порядке возрастания сетки).
    Вырожденные входы (меньше трёх фраз, пустая после обрезки сетка, все
    признаки совпадают) честно дают один кластер — это не ошибка.
    """
    n = features.shape[0]
    valid = sorted({k for k in k_grid if 2 <= k <= n - 1})
    if not valid:
        return np.zeros(n, dtype=int)

    best_labels = np.zeros(n, dtype=int)
    best_score = -np.inf
    for k in valid:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:  # строго больше: при равенстве остаётся меньший k
            best_score, best_labels = score, labels
    return best_labels


def _silhouette(features, labels: np.ndarray) -> float | None:
    """Силуэт итогового разбиения; None, если разбиение вырождено (k=1)."""
    if len(set(labels.tolist())) < 2:
        return None
    return float(silhouette_score(features, labels))


def _build_clusters(phrases: list[str], lemmatized: list[str], labels: np.ndarray) -> list[Cluster]:
    """Кластеры по меткам: состав в исходном порядке, тема — топ-леммы состава.

    Частотность лемм внутри кластера: метка темы должна читаться человеком
    («курс, английский, язык»), а не быть артефактом представления.
    Детерминизм top-лемм — сортировка по (-частота, лемма).
    """
    by_cluster: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_cluster.setdefault(label, []).append(index)

    clusters: list[Cluster] = []
    for label in sorted(by_cluster):
        members = by_cluster[label]
        lemma_counts: Counter[str] = Counter()
        for index in members:
            lemma_counts.update(lemmatized[index].split())
        ranked = sorted(lemma_counts.items(), key=lambda item: (-item[1], item[0]))
        top_lemmas = [lemma for lemma, _ in ranked[:TOP_LABEL_LEMMAS]]
        clusters.append(Cluster(top_lemmas=top_lemmas, phrases=[phrases[index] for index in members]))
    return clusters

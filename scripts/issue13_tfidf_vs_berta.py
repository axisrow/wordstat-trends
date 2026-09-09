"""Честное сравнение TF-IDF + k-means против BERTA-эмбеддингов + k-means
(issue #70, Фаза 4.3).

Один набор фраз — два конвейера представления, одинаковый k-means:

1. TF-IDF по леммам (sklearn + pymorphy3, всё в основных зависимостях);
2. BERTA-эмбеддинги (src/wordstat_trends/nlp/embeddings.py, extra `nlp`,
   префикс задачи "clustering: ").

Набор фраз: три seed-фразы фикстур Фазы 1 (новогодние подарки / купить
телефон / курсы английского) + словарь похожих и разных фраз, сгруппированных
по ожидаемым темам. Ожидаемая разметка известна — она используется как
ground truth для ARI-vs-expected (само сравнение конвейеров честное: ни один
из них не видит метки при обучении, k-means без supervised-сигнала).

Метрики:
- силуэт (silhouette_score) на представлении с итоговыми метками;
- ARI между seed'ами k-means (стабильность разбиения к инициализации);
- ARI итоговых меток против ожидаемых групп;
- интерпретируемость: топ-леммы кластера (TF-IDF, по центроидам) и
  топ-фразы кластера (оба конвейера, по близости к центроиду).

Отрицательный результат — тоже результат: если TF-IDF не хуже, витрина
остаётся на лёгком конвейере (см. docs/ISSUE_13_TFIDF_VS_BERTA.md).

Запуск:
- python scripts/issue13_tfidf_vs_berta.py          -> оба конвейера (BERTA требует extra nlp)
- python scripts/issue13_tfidf_vs_berta.py --tfidf  -> только лёгкий конвейер

Вывод — JSON в stdout (машиночитаемый контракт, как в experiment_vector_d).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

# Ограничить BLAS-потоки до импорта numpy/sklearn — несколько независимых
# прогонов на одной машине не должны делить все ядра (как experiment_vector_d).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wordstat_trends.nlp.lemmatize import lemmatized_phrase

# Seed-фразы фикстур Фазы 1 (dynamics_*.csv) + словарь похожих/разных фраз.
# Ключ — имя ожидаемой темы (= ground truth), порядок детерминирован.
PHRASE_GROUPS: dict[str, list[str]] = {
    "новогодние подарки": [
        "новогодние подарки",
        "подарки на новый год",
        "что подарить на новый год",
        "новогодние подарки коллегам",
        "подарки близким на новый год",
        "упаковка для новогодних подарков",
        "новогодние сувениры",
        "подарочные наборы на новый год",
        "новогодний адвент календарь",
        "подарки сотрудникам на новый год",
    ],
    "смартфоны": [
        "купить телефон",
        "купить смартфон",
        "смартфон айфон купить",
        "телефон самсунг цена",
        "купить телефон в рассрочку",
        "смартфон xiaomi купить",
        "б у телефон купить",
        "новые смартфоны 2026",
        "чехол для смартфона",
        "защитное стекло для телефона",
    ],
    "изучение английского": [
        "курсы английского",
        "курсы английского языка",
        "выучить английский язык",
        "репетитор по английскому",
        "английский для начинающих",
        "разговорный английский курс",
        "онлайн курсы английского",
        "английский язык для детей",
        "подготовка к ielts",
        "школа английского языка",
    ],
    "мебель": [
        "купить диван",
        "диван в рассрочку",
        "угловой диван купить",
        "шкаф купе на заказ",
        "кухонный уголок",
        "стол обеденный купить",
        "кровать с матрасом",
        "мебель для спальни",
        "мягкая мебель цена",
        "корпусная мебель на заказ",
    ],
    "доставка еды": [
        "доставка пиццы",
        "заказать пиццу рядом",
        "доставка суши",
        "доставка еды на дом",
        "роллы доставка",
        "доставка обедов в офис",
        "еда на ночь доставка",
        "доставка борща",
        "суши вок заказать",
        "фаст фуд доставка",
    ],
}

SEEDS = (0, 1, 2, 42)
TOP_TERMS = 5
TOP_PHRASES = 3

# Фейк для тестов и фикстурного прогона без extra `nlp`: детерминированный
# вектор из лемм фразы. Он НЕ претендует на качество BERTA — только на
# корректность формы контракта embed_fn и детерминизм конвейера.
EMBEDDING_DIM_FAKE = 64


def fake_embed(phrases: list[str], task: str) -> np.ndarray:
    """Детерминированные псевдо-эмбеддинги: seed вектора леммы — её md5.

    Похожесть фраз растёт с общими леммами (сумма векторов), то есть фейк
    содержательно ближе к мешку лемм, чем к трансформеру — это оговорено
    в отчёте и не позволяет выдавать его результат за BERTA.
    """
    import hashlib

    def _lemma_vector(lemma: str) -> np.ndarray:
        digest = hashlib.md5(lemma.encode("utf-8")).digest()[: EMBEDDING_DIM_FAKE // 8]  # noqa: S324 — не криптография
        raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        # Растянуть 8 байт в EMBEDDING_DIM_FAKE координат повторением блоков.
        return np.tile(raw, EMBEDDING_DIM_FAKE // raw.shape[0])

    vectors = np.zeros((len(phrases), EMBEDDING_DIM_FAKE), dtype=np.float32)
    for i, phrase in enumerate(phrases):
        for token in phrase.split():
            vectors[i] += _lemma_vector(token)
    assert task == "clustering"
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def build_dataset() -> tuple[list[str], list[str]]:
    """Фразы и ожидаемые метки тем (ground truth), детерминированный порядок."""
    phrases: list[str] = []
    expected: list[str] = []
    for group, members in PHRASE_GROUPS.items():
        phrases.extend(member.strip() for member in members)
        expected.extend([group] * len(members))
    return phrases, expected


def tfidf_features(phrases: list[str]):
    """TF-IDF по леммам: лемматизация отдельно, векторайзер по пробелам.

    sublinear_tf + min_df=2: типовые настройки для коротких фраз — редкие
    опечатки не должны получать раздутый вес.
    """
    lemmatized = [lemmatized_phrase(phrase) for phrase in phrases]
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=2, sublinear_tf=True)
    return vectorizer.fit_transform(lemmatized), vectorizer, lemmatized


def berta_features(phrases: list[str], embed_fn: Callable[[list[str], str], np.ndarray]) -> np.ndarray:
    """BERTA-эмбеддинги лемматизированных фраз с task="clustering".

    embed_fn инъекцируется: прод — wordstat_trends.nlp.embeddings.embed_phrases
    (ленивый импорт sentence-transformers), тесты подставляют детерминированный
    фейк без extra `nlp`.
    """
    lemmatized = [lemmatized_phrase(phrase) for phrase in phrases]
    return embed_fn(lemmatized, "clustering")


def run_kmeans(features, k: int, seeds: tuple[int, ...] = SEEDS) -> dict:
    """k-means по нескольким seed'ам: итоговые метки, силуэт, ARI-стабильность.

    Итоговое разбиение — seed с максимальным силуэтом (детерминированный
    tie-break: первый максимум в порядке seeds). ARI считается для всех пар
    seed'ов — метрика стабильности к инициализации, а не качества.
    """
    runs: list[dict] = []
    for seed in seeds:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
        silhouette = float(silhouette_score(features, labels))
        runs.append({"seed": seed, "labels": labels.tolist(), "silhouette": silhouette})

    pairwise_ari = [
        float(adjusted_rand_score(runs[i]["labels"], runs[j]["labels"]))
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]

    best = max(runs, key=lambda run: run["silhouette"])
    return {
        "k": k,
        "seeds": list(seeds),
        "runs_silhouettes": [round(run["silhouette"], 6) for run in runs],
        "best_seed": best["seed"],
        "labels": best["labels"],
        "silhouette": round(best["silhouette"], 6),
        "ari_between_seeds_mean": round(float(np.mean(pairwise_ari)), 6),
        "ari_between_seeds_min": round(float(np.min(pairwise_ari)), 6),
    }


def cluster_interpretability_tfidf(vectorizer, labels: list[int], lemmatized: list[str]) -> list[dict]:
    """Топ-леммы каждого кластера по среднему TF-IDF-весу внутри кластера.

    Считается по строкам матрицы TF-IDF: средний вес леммы среди фраз кластера —
    не центроид k-means, а простая интерпретируемая агрегация.
    """
    feature_names = vectorizer.get_feature_names_out()
    matrix = vectorizer.transform(lemmatized).toarray()
    result = []
    for cluster in sorted(set(labels)):
        rows = matrix[[i for i, lab in enumerate(labels) if lab == cluster]]
        mean_weights = rows.mean(axis=0)
        top_idx = np.argsort(mean_weights)[::-1][:TOP_TERMS]
        result.append(
            {
                "cluster": cluster,
                "n_phrases": int(rows.shape[0]),
                "top_terms": [str(feature_names[i]) for i in top_idx],
            }
        )
    return result


def cluster_interpretability_phrases(features, phrases: list[str], labels: list[int]) -> list[dict]:
    """Топ-фразы кластера по близости к центроиду (общая для обоих конвейеров).

    Для TF-IDF матрица разреженная — центроид и дистанции считаются через
    dense-представление (набор из десятков фраз это позволяет).
    """
    dense = np.asarray(features.todense()) if hasattr(features, "todense") else np.asarray(features)
    centroids = np.vstack(
        [dense[[i for i, lab in enumerate(labels) if lab == cluster]].mean(axis=0) for cluster in sorted(set(labels))]
    )
    result = []
    for cluster in sorted(set(labels)):
        members = [i for i, lab in enumerate(labels) if lab == cluster]
        dists = np.linalg.norm(dense[members] - centroids[cluster], axis=1)
        top_idx = [members[i] for i in np.argsort(dists)[:TOP_PHRASES]]
        result.append({"cluster": cluster, "n_phrases": len(members), "top_phrases": [phrases[i] for i in top_idx]})
    return result


def run_tfidf_pipeline(phrases: list[str], expected: list[str]) -> dict:
    """Конвейер 1: TF-IDF по леммам + k-means."""
    features, vectorizer, lemmatized = tfidf_features(phrases)
    clustering = run_kmeans(features, k=len(PHRASE_GROUPS))
    labels = clustering["labels"]
    clustering["ari_vs_expected"] = round(float(adjusted_rand_score(expected, labels)), 6)
    return {
        "pipeline": "tfidf_lemmas",
        "clustering": clustering,
        "top_terms_by_cluster": cluster_interpretability_tfidf(vectorizer, labels, lemmatized),
        "top_phrases_by_cluster": cluster_interpretability_phrases(features, phrases, labels),
    }


def run_berta_pipeline(phrases: list[str], expected: list[str], embed_fn) -> dict:
    """Конвейер 2: BERTA-эмбеддинги (task=clustering) + k-means."""
    features = berta_features(phrases, embed_fn)
    clustering = run_kmeans(features, k=len(PHRASE_GROUPS))
    labels = clustering["labels"]
    clustering["ari_vs_expected"] = round(float(adjusted_rand_score(expected, labels)), 6)
    return {
        "pipeline": "berta_embeddings",
        "clustering": clustering,
        # Топ-лемм для BERTA нет: представление не в пространстве лемм —
        # интерпретируемость только по фразам.
        "top_phrases_by_cluster": cluster_interpretability_phrases(features, phrases, labels),
    }


def _default_embed_fn():
    from wordstat_trends.nlp.embeddings import embed_phrases

    return embed_phrases


def main() -> int:
    phrases, expected = build_dataset()
    report: dict = {"n_phrases": len(phrases), "n_expected_groups": len(PHRASE_GROUPS), "pipelines": {}}

    report["pipelines"]["tfidf_lemmas"] = run_tfidf_pipeline(phrases, expected)

    if "--tfidf" in sys.argv:
        report["pipelines"]["berta_embeddings"] = "SKIPPED — запуск с --tfidf"
    else:
        try:
            embed_fn = _default_embed_fn()
            # Прогрев на одной фразе: импорт модуля эмбеддингов ленивый и без
            # extra `nlp` проходит — ImportError реально летит при первой
            # загрузке модели (embed_phrases -> _model). Отсутствие extra —
            # ожидаемый исход на машинах без модели: репортится честно,
            # а не валит весь прогон.
            embed_fn(["тестовая фраза"], "clustering")
        except ImportError as error:
            report["pipelines"]["berta_embeddings"] = f"SKIPPED — {error}"
        else:
            report["pipelines"]["berta_embeddings"] = run_berta_pipeline(phrases, expected, embed_fn)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

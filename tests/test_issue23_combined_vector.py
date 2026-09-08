"""Тесты для scripts/issue23_combined_vector.py (итерация 3 MVP #23:
объединённый вектор sp=7 + sp=12 как представление спроса)."""

import math

import numpy as np
import pytest

from scripts.experiment_vector_d import DAILY_FIXTURES, FIXTURES, TRAIN_WINDOW_DAYS
from scripts.issue23_combined_vector import (
    DAILY_REFIT_DROPS,
    PHRASES,
    CombinedVector,
    block_defined,
    build_combined_vector,
    nearest_neighbors,
    neighbor_preserved,
    pairwise_structure,
    pearson_or_nan,
    similarity,
    structure_fingerprint,
    zscore_block,
)


def _make_combined(
    seasonal7: np.ndarray,
    seasonal12: np.ndarray,
    has7: bool = True,
    has12: bool = True,
    label: str = "test",
) -> CombinedVector:
    return CombinedVector(
        label=label,
        spec7="mul/add/mul",
        spec12="mul/None/mul",
        has_seasonal7=has7,
        has_seasonal12=has12,
        z7=zscore_block(seasonal7, has7),
        z12=zscore_block(seasonal12, has12),
    )


def test_zscore_block_normalizes_mean_and_std():
    block = zscore_block(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), has_seasonal=True)
    assert block.mean() == pytest.approx(0.0, abs=1e-12)
    assert block.std() == pytest.approx(1.0, abs=1e-12)


def test_zscore_block_without_seasonality_is_zero_vector_by_construction():
    # has_seasonal=False — модель не оценивала сезонность (не «плоский
    # измеренный профиль»): нулевой вектор-маркер, z-скоринг неприменим.
    block = zscore_block(np.zeros(7), has_seasonal=False)
    assert np.array_equal(block, np.zeros(7))


def test_zscore_block_constant_profile_degenerates_to_zero():
    # Константный, но заявленно сезонный профиль: std == 0, после
    # центрирования нули — pearson_or_nan обязан трактовать сходство с ним
    # как NaN (см. следующий тест), не как 0 или 1.
    block = zscore_block(np.full(12, 2.5), has_seasonal=True)
    assert np.array_equal(block, np.zeros(12))


def test_pearson_or_nan_returns_nan_for_constant_vector():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert math.isnan(pearson_or_nan(a, np.zeros(4)))


def test_pearson_or_nan_perfect_positive_and_negative():
    a = np.array([1.0, 2.0, 3.0])
    assert pearson_or_nan(a, a * 3 + 1) == pytest.approx(1.0)
    assert pearson_or_nan(a, -a) == pytest.approx(-1.0)


def test_combined_vector_is_concatenation_of_blocks():
    vec = _make_combined(np.arange(7, dtype=float), np.arange(12, dtype=float))
    assert vec.combined.shape == (19,)
    assert np.array_equal(vec.combined[:7], vec.z7)
    assert np.array_equal(vec.combined[7:], vec.z12)


def test_has_combined_signal_false_only_when_both_blocks_absent():
    alive = _make_combined(np.arange(7, dtype=float), np.zeros(12), has12=False)
    dead = _make_combined(np.zeros(7), np.zeros(12), has7=False, has12=False)
    assert alive.has_combined_signal is True
    assert dead.has_combined_signal is False


def test_nearest_neighbor_undefined_when_all_similarities_nan():
    # Все три фразы без сезонности ни в одном блоке: у seasonal нет ни
    # одного определённого сходства — neighbor=None с флагом undefined,
    # не «сама себе ближайшая» и не argmax по NaN.
    vectors = {p: _make_combined(np.zeros(7), np.zeros(12), has7=False, has12=False, label=p) for p in PHRASES}
    nn = nearest_neighbors(vectors)
    for repr_name in ("sp7", "sp12", "combined"):
        for phrase in PHRASES:
            assert nn[repr_name][phrase]["undefined"] is True
            assert nn[repr_name][phrase]["neighbor"] is None


def test_nearest_neighbor_picks_max_over_defined_similarities():
    a = _make_combined(np.array([1.0, 2, 3, 4, 5, 6, 7]), np.arange(12, dtype=float), label="a")
    b = _make_combined(np.array([7.0, 6, 5, 4, 3, 2, 1]), np.arange(12, dtype=float), label="b")
    c = _make_combined(np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]), np.arange(12, dtype=float), label="c")
    nn = nearest_neighbors({"a": a, "b": b, "c": c})
    assert nn["combined"]["a"]["neighbor"] == "c"
    assert nn["combined"]["b"]["neighbor"] is not None


def test_combined_similarity_nan_when_any_participating_block_absent():
    # Инвариант (ревью PR #47): combined-similarity — NaN, если хотя бы
    # один из ЧЕТЫРЁХ участвующих блоков не определён. Частично живой
    # combined (один блок есть) не даёт определённого числа: нулевой
    # блок-маркер не подмешивается в живую конкатенацию.
    full = _make_combined(np.arange(7, dtype=float), np.arange(12, dtype=float), label="full")
    no_weekly = _make_combined(np.zeros(7), np.arange(12, dtype=float), has7=False, label="no_weekly")
    assert math.isnan(similarity(full, no_weekly, "combined"))
    assert math.isnan(similarity(no_weekly, full, "combined"))


def test_combined_similarity_defined_when_all_four_blocks_present():
    a = _make_combined(np.arange(7, dtype=float), np.arange(12, dtype=float), label="a")
    b = _make_combined(np.arange(7, dtype=float) * 2, np.arange(12, dtype=float) * 2, label="b")
    assert similarity(a, b, "combined") == pytest.approx(1.0)


def test_block_defined_false_for_absent_and_degenerate_blocks():
    alive = _make_combined(np.arange(7, dtype=float), np.arange(12, dtype=float), label="alive")
    absent = _make_combined(np.zeros(7), np.arange(12, dtype=float), has7=False, label="absent")
    degenerate = _make_combined(np.full(7, 3.0), np.arange(12, dtype=float), label="degenerate")
    assert block_defined(alive, "sp7") is True
    assert block_defined(absent, "sp7") is False
    assert block_defined(degenerate, "sp7") is False


def test_pairwise_structure_combined_is_null_with_block_absent():
    full = _make_combined(np.arange(7, dtype=float), np.arange(12, dtype=float), label="full")
    no_weekly = _make_combined(np.zeros(7), np.arange(12, dtype=float), has7=False, label="no_weekly")
    structure = pairwise_structure({"full": full, "no_weekly": no_weekly})
    assert structure["combined"]["full <-> no_weekly"] is None
    assert structure["sp12"]["full <-> no_weekly"] == pytest.approx(1.0)


def test_neighbor_preserved_undefined_in_both_is_not_counted_as_preserved():
    undefined = {"neighbor": None, "sim": None, "undefined": True}
    result = neighbor_preserved(undefined, undefined)
    assert result["preserved"] is False
    assert result["undefined_in_both"] is True
    assert result["undefined_in_either"] is True


def test_neighbor_preserved_change_and_stability_of_defined_neighbors():
    before_mid = {"neighbor": "mid", "sim": 0.9, "undefined": False}
    after_mid = {"neighbor": "mid", "sim": 0.8, "undefined": False}
    after_phone = {"neighbor": "phone", "sim": 0.8, "undefined": False}
    undefined_neighbor = {"neighbor": None, "sim": None, "undefined": True}
    assert neighbor_preserved(before_mid, after_mid)["preserved"] is True
    assert neighbor_preserved(before_mid, after_phone)["preserved"] is False
    # defined→undefined — тоже не «сохранилось»: сосед перестал существовать.
    assert neighbor_preserved(before_mid, undefined_neighbor)["preserved"] is False
    assert neighbor_preserved(before_mid, undefined_neighbor)["undefined_in_both"] is False


def test_structure_fingerprint_orders_by_similarity_desc_and_puts_none_last():
    structure = {"p1": 0.1, "p2": None, "p3": 0.9}
    assert structure_fingerprint(structure) == ["p3", "p1", "p2"]


def test_refit_drops_match_iteration_2_convention():
    # −1 и −2 недели, как в итерации 2 (docs/EXPERIMENT_VECTOR_D.md):
    # «устойчивость структуры» обязана измеряться на той же сетке рефитов,
    # что уже измеренная устойчивость коэффициентов sp=7-блока.
    assert DAILY_REFIT_DROPS == (7, 14)


def test_phrases_cover_exactly_the_three_fixture_pairs():
    assert len(PHRASES) == 3
    for phrase in PHRASES:
        assert phrase in DAILY_FIXTURES
        assert phrase in FIXTURES


def test_build_combined_vector_seasonal_phrase_has_both_blocks():
    label = "seasonal (новогодние подарки)"
    daily = load_daily_train_window(label)
    monthly = FIXTURES[label].exists()
    assert monthly
    vec = build_combined_vector(label, daily, load_monthly(label))
    assert vec.has_seasonal7 is True
    assert vec.has_seasonal12 is True
    # Конвенция раздела 1 обеих итераций EXPERIMENT_VECTOR_D (aic):
    # у сезонной фразы обе спеки мультипликативно-сезонные.
    assert "mul" in vec.spec7
    assert "mul" in vec.spec12
    assert vec.combined.shape == (19,)


def load_daily_train_window(label: str):
    from scripts.experiment_vector_d import load_daily_dynamics_csv, truncate_series

    series = load_daily_dynamics_csv(DAILY_FIXTURES[label])
    assert len(series) >= TRAIN_WINDOW_DAYS
    return truncate_series(series, len(series) - TRAIN_WINDOW_DAYS)


def load_monthly(label: str):
    from scripts.experiment_vector_d import load_dynamics_csv

    return load_dynamics_csv(FIXTURES[label])


def test_build_combined_vector_high_freq_has_no_weekly_block():
    # Зафиксировано итерацией 2 и #32: у high_freq основной бэкенд (aic,
    # окно 56) сезонность по sp=7 не включает — блок отсутствует по
    # построению, а не из-за сломанного парсинга или потерянных данных.
    label = "high_freq (купить телефон)"
    vec = build_combined_vector(label, load_daily_train_window(label), load_monthly(label))
    assert vec.has_seasonal7 is False
    assert vec.has_seasonal12 is True
    assert np.array_equal(vec.z7, np.zeros(7))

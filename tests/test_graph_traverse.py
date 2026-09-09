"""Тесты ядра обхода графа запросов (issue #81).



Провайдер рёбер — синтетический: реального источника нет, пока не закрыт
wordstat-cli#62 (``top_related`` отдаёт 0 строк). Все графы ниже —
dict «фраза → соседи с частотностями», требования #81: детерминированность,
идемпотентность, лимиты глубины и объёма, дедупликация.
"""

from __future__ import annotations

import pytest

from wordstat_trends.graph import RelatedPhrase, TraversalLimits, run_traversal


class DictProvider:
    """Синтетический граф: фиксированный dict рёбер + счётчик вызовов."""

    def __init__(self, edges: dict[str, list[tuple[str, int]]]) -> None:
        self.edges = edges
        self.calls: list[str] = []

    def neighbors(self, phrase: str) -> list[RelatedPhrase]:
        self.calls.append(phrase)
        return [RelatedPhrase(phrase=p, frequency=f) for p, f in self.edges.get(phrase, [])]


class TestBasic:
    def test_chain_is_walked_with_depths(self):
        provider = DictProvider({"a": [("b", 10)], "b": [("c", 5)], "c": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=3))
        assert result.phrases == ["a", "b", "c"]
        assert result.state.visited == {"a": 0, "b": 1, "c": 2}
        assert not result.truncated_by_phrases
        assert not result.truncated_by_depth

    def test_isolated_seed_is_dead_end(self):
        provider = DictProvider({"a": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=3))
        assert result.phrases == ["a"]
        assert provider.calls == ["a"]

    def test_cycle_terminates(self):
        # a -> b -> a: уже посещённая фраза не переустанавливается
        provider = DictProvider({"a": [("b", 10)], "b": [("a", 20)]})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=5))
        assert result.phrases == ["a", "b"]
        assert provider.calls == ["a", "b"]

    def test_diamond_deduplicates_shared_vertex(self):
        # ромб: a -> b, a -> c; b -> d, c -> d — d один раз на глубине 2
        provider = DictProvider({"a": [("b", 10), ("c", 9)], "b": [("d", 1)], "c": [("d", 2)], "d": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=5))
        assert result.phrases == ["a", "b", "c", "d"]
        assert provider.calls.count("d") == 1

    def test_self_loop_is_not_an_edge(self):
        provider = DictProvider({"a": [("a", 100), ("b", 1)], "b": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=2))
        assert result.phrases == ["a", "b"]

    def test_duplicate_edges_are_deduplicated(self):
        provider = DictProvider({"a": [("b", 10), ("b", 7)], "b": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=2))
        assert result.phrases == ["a", "b"]


class TestPriority:
    def test_highest_frequency_expanded_first(self):
        # приоритет ГЛОБАЛЬНЫЙ по фронтиру, не послойный: после раскрытия «a»
        # во фронтире «c»(10) и «b»(5) — «c» идёт первым; но найденный через
        # «b» «d»(1000) обгоняет всё остальное, потому что частотнее
        provider = DictProvider(
            {"a": [("c", 10), ("b", 5)], "b": [("d", 1000)], "c": [("e", 3)], "d": [], "e": []}
        )
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=2))
        assert result.phrases == ["a", "b", "c", "d", "e"]
        assert provider.calls == ["a", "c", "b", "d", "e"]

    def test_equal_frequencies_tie_break_is_lexicographic(self):
        provider = DictProvider({"a": [("z", 5), ("m", 5), ("b", 5)]})
        run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=1))
        assert provider.calls == ["a", "b", "m", "z"]

    def test_deterministic_regardless_of_neighbor_order(self):
        edges_one = {"a": [("b", 5), ("c", 5)], "b": [], "c": []}
        edges_two = {"a": [("c", 5), ("b", 5)], "b": [], "c": []}
        first = run_traversal(["a"], DictProvider(edges_one), TraversalLimits(max_phrases=10, max_depth=1))
        second = run_traversal(["a"], DictProvider(edges_two), TraversalLimits(max_phrases=10, max_depth=1))
        assert first.phrases == second.phrases
        assert first.provider_calls == second.provider_calls


class TestSeeds:
    def test_seeds_start_at_depth_zero(self):
        provider = DictProvider({"a": [("b", 1)], "b": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=1))
        assert result.state.visited == {"a": 0, "b": 1}

    def test_duplicate_seeds_counted_once(self):
        provider = DictProvider({"a": []})
        result = run_traversal(["a", "a"], provider, TraversalLimits(max_phrases=10, max_depth=1))
        assert result.phrases == ["a"]
        assert provider.calls == ["a"]

    def test_seeds_sorted_regardless_of_input_order(self):
        provider = DictProvider({"c": [], "a": [], "b": []})
        run_traversal(["c", "a", "b"], provider, TraversalLimits(max_phrases=10, max_depth=0))
        assert provider.calls == ["a", "b", "c"]

    def test_multiple_seeds_disjoint_components(self):
        provider = DictProvider({"a": [("b", 1)], "x": [("y", 2)], "b": [], "y": []})
        result = run_traversal(["a", "x"], provider, TraversalLimits(max_phrases=10, max_depth=2))
        assert result.phrases == ["a", "b", "x", "y"]


class TestLimits:
    def test_phrase_cap_with_only_depth_leftovers_flags_depth_not_phrases(self):
        # пограничный случай из ревью: словарь достиг max_phrases, но во
        # фронтире остались ТОЛЬКО depth-обрезанные записи — обрезки по
        # фразам не было, флаг должен это отражать
        provider = DictProvider({"a": [("b", 10)], "b": [("c", 5)], "c": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=2, max_depth=1))
        assert result.phrases == ["a", "b"]
        assert len(result.state.visited) == 2
        assert result.truncated_by_depth
        assert not result.truncated_by_phrases

    def test_max_phrases_truncates_and_flags(self):
        provider = DictProvider({"a": [("b", 10), ("c", 9), ("d", 8)]})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=3, max_depth=5))
        assert result.phrases == ["a", "b", "c"]
        assert result.truncated_by_phrases
        assert not result.truncated_by_depth

    def test_max_depth_truncates_and_flags(self):
        provider = DictProvider({"a": [("b", 10)], "b": [("c", 10)], "c": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=1))
        assert result.phrases == ["a", "b"]
        assert result.truncated_by_depth
        assert not result.truncated_by_phrases

    def test_max_depth_zero_keeps_seeds_only(self):
        provider = DictProvider({"a": [("b", 10)]})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=0))
        assert result.phrases == ["a"]
        assert result.truncated_by_depth

    def test_invalid_limits_raise(self):
        provider = DictProvider({})
        with pytest.raises(ValueError, match="max_phrases"):
            run_traversal(["a"], provider, TraversalLimits(max_phrases=0, max_depth=1))
        with pytest.raises(ValueError, match="max_depth"):
            run_traversal(["a"], provider, TraversalLimits(max_phrases=1, max_depth=-1))

    def test_truncated_candidates_survive_for_continuation(self):
        # обрезанные по max_phrases кандидаты остаются во фронтире состояния:
        # продолжение с большим лимитом добирает их без новых вызовов
        # провайдера по уже раскрытым фразам
        provider = DictProvider({"a": [("b", 10), ("c", 9)], "b": [], "c": []})
        first = run_traversal(["a"], provider, TraversalLimits(max_phrases=2, max_depth=5))
        assert first.phrases == ["a", "b"]
        assert first.truncated_by_phrases

        second = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=5), state=first.state)
        assert second.phrases == ["a", "b", "c"]
        assert second.provider_calls == 1  # только «c»: «a» и «b» уже раскрыты


class TestIdempotency:
    def test_rerun_on_exhausted_state_calls_no_provider(self):
        provider = DictProvider({"a": [("b", 1)], "b": []})
        first = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=2))
        assert provider.calls == ["a", "b"]

        frozen = provider.calls.copy()
        second = run_traversal(["a"], provider, TraversalLimits(max_phrases=10, max_depth=2), state=first.state)
        assert provider.calls == frozen
        assert second.phrases == first.phrases
        assert not second.truncated_by_phrases

    def test_rerun_does_not_exceed_limits(self):
        # первый прогон: словарь полон на «a» + «c» (частотнее «b»),
        # «b» осталась во фронтире; повторный прогон с теми же лимитами
        # не добавляет её и не обращается к провайдеру
        provider = DictProvider({"a": [("b", 1), ("c", 2)], "b": [], "c": []})
        first = run_traversal(["a"], provider, TraversalLimits(max_phrases=2, max_depth=5))
        assert first.phrases == ["a", "c"]
        frozen = provider.calls.copy()
        second = run_traversal(["a"], provider, TraversalLimits(max_phrases=2, max_depth=5), state=first.state)
        assert second.phrases == ["a", "c"]
        assert provider.calls == frozen
        assert len(second.state.visited) == 2

    def test_continuation_from_partial_state(self):
        # имитация дозаписи #82: первое состояние остановлено по лимиту фраз,
        # продолжение с теми же лимитами после «расширения бюджета»
        provider = DictProvider({"a": [("b", 10), ("c", 9), ("d", 8)]})
        first = run_traversal(["a"], provider, TraversalLimits(max_phrases=2, max_depth=5))
        second = run_traversal([], provider, TraversalLimits(max_phrases=4, max_depth=5), state=first.state)
        assert second.phrases == ["a", "b", "c", "d"]


class TestProviderContract:
    def test_seed_expanded_before_any_found_phrase(self):
        # у seed больше нет «бесконечной частотности» (inf не сериализуется
        # в стандартный JSON): первенство даёт явный is_seed, поэтому даже
        # сосед с гигантской частотностью раскрывается после seed
        provider = DictProvider({"b": [("z", 10**9)], "z": []})
        result = run_traversal(["b"], provider, TraversalLimits(max_phrases=10, max_depth=1))
        assert provider.calls == ["b", "z"]
        assert result.phrases == ["b", "z"]

    def test_state_is_standard_json_serializable(self):
        # состояние попадёт в персистентность #82 как есть: ни inf, ни
        # прочих значений, которые json.dumps кодирует нестандартно
        import json

        provider = DictProvider({"a": [("b", 10), ("c", 9)], "b": [], "c": []})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=2, max_depth=5))
        payload = json.dumps(
            {
                "visited": result.state.visited,
                "frontier": [
                    {"phrase": e.phrase, "frequency": e.frequency, "depth": e.depth, "is_seed": e.is_seed}
                    for e in result.state.frontier
                ],
            }
        )
        assert "Infinity" not in payload
        assert "NaN" not in payload


    def test_provider_ignoring_unknown_phrase(self):
        # провайдер может не знать фразу (нет рёбер) — это тупик, не ошибка
        provider = DictProvider({})
        result = run_traversal(["a"], provider, TraversalLimits(max_phrases=5, max_depth=2))
        assert result.phrases == ["a"]
        assert not result.truncated_by_phrases

    def test_result_phrases_sorted_deterministically(self):
        provider = DictProvider({"m": [("z", 1), ("a", 2)], "z": [], "a": []})
        result = run_traversal(["m"], provider, TraversalLimits(max_phrases=10, max_depth=1))
        assert result.phrases == ["a", "m", "z"]

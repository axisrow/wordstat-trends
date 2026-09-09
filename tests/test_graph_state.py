"""Персистентное состояние обхода между прогонами (issue #82)."""

import json

import pytest

from wordstat_trends.graph.state import GraphStateError, GraphStateFile
from wordstat_trends.graph.traverse import FrontierEntry, TraversalState


def make_state() -> TraversalState:
    return TraversalState(
        visited={"seed фраза": 0, "сосед": 1},
        frontier=[
            FrontierEntry(phrase="кандидат", frequency=123.0, depth=1),
            FrontierEntry(phrase="seed фраза", frequency=0.0, depth=0, is_seed=True),
        ],
    )


def test_missing_file_is_first_run(tmp_path):
    store = GraphStateFile(tmp_path / "graph_state.json")

    assert store.load() == (None, None)


def test_roundtrip_state_and_budget(tmp_path):
    path = tmp_path / "graph_state.json"
    store = GraphStateFile(path)
    state = make_state()
    budget = {"date": "2026-09-09", "spent": 3}

    store.save(state, budget)
    loaded_state, loaded_budget = store.load()

    assert loaded_state is not None
    assert loaded_state.visited == {"seed фраза": 0, "сосед": 1}
    assert loaded_state.frontier == state.frontier
    assert loaded_state.phrases == ["seed фраза", "сосед"]
    assert loaded_budget == budget


def test_budget_is_optional(tmp_path):
    path = tmp_path / "graph_state.json"
    store = GraphStateFile(path)
    store.save(make_state(), None)

    _, budget = store.load()

    assert budget is None


def test_save_replaces_previous_version(tmp_path):
    path = tmp_path / "graph_state.json"
    store = GraphStateFile(path)
    store.save(make_state(), {"date": "2026-09-09", "spent": 1})

    state = make_state()
    state.visited["ещё одна"] = 2
    store.save(state, {"date": "2026-09-09", "spent": 2})

    loaded, budget = store.load()
    assert loaded is not None and "ещё одна" in loaded.visited
    assert budget == {"date": "2026-09-09", "spent": 2}
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_no_tmp_files_left_behind(tmp_path):
    store = GraphStateFile(tmp_path / "graph_state.json")
    store.save(make_state(), None)

    assert [p.name for p in tmp_path.iterdir()] == ["graph_state.json"]


def test_corrupt_json_raises(tmp_path):
    path = tmp_path / "graph_state.json"
    path.write_text("{не json", encoding="utf-8")

    with pytest.raises(GraphStateError):
        GraphStateFile(path).load()


def test_wrong_schema_raises(tmp_path):
    path = tmp_path / "graph_state.json"
    path.write_text(json.dumps({"visited": {"а": 0}, "frontier": [{"phrase": "б"}]}), encoding="utf-8")

    with pytest.raises(GraphStateError):
        GraphStateFile(path).load()


def test_wrong_version_raises(tmp_path):
    path = tmp_path / "graph_state.json"
    path.write_text(json.dumps({"version": 99, "visited": {}, "frontier": []}), encoding="utf-8")

    with pytest.raises(GraphStateError, match="версия"):
        GraphStateFile(path).load()

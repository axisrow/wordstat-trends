"""Интеграция обхода графа с конвейером сбора (issue #82).

Сбор подменён заглушкой subprocess, которая пишет прогоны в раскладке
wordstat-cli (manifest.json + top_related.parquet) по заранее заданному
графу — так воспроизводится главный контракт интеграции: собранные фразы
становятся рёбрами, рёбра пополняют расписание следующего дня. Браузера
и сети нет.
"""

import json
import threading
from datetime import date
from pathlib import Path

import pandas as pd

from wordstat_trends.collect_service import CollectService, Config
from wordstat_trends.graph.state import GraphStateFile

TODAY = date(2026, 9, 9)


class _FixedDate(date):
    """«Сегодня» для сервиса: бюджеты и даты состояния не зависят от реального дня."""

    @classmethod
    def today(cls) -> date:
        return TODAY


TITLE = "Запросы, похожие на «{p}», 08.08.2026 — 08.09.2026, Россия, все устройства"


def make_service(tmp_path, monkeypatch, *, graph_enabled=True, budget=10, graph=None):
    cfg = Config(
        phrases_file=tmp_path / "phrases.txt",
        results_dir=tmp_path / "results",
        git_dir=tmp_path / "repo",  # без .git — коммит пропускается
        phrase_delay_s=0,
        graph_enabled=graph_enabled,
        graph_state_file=tmp_path / "graph_state.json",
        graph_daily_budget=budget,
    )
    cfg.phrases_file.write_text("seed один\nseed два\n", encoding="utf-8")
    svc = CollectService(cfg)

    collected: list[str] = []
    lock = threading.Lock()
    graph = graph if graph is not None else {}
    run_no = [0]

    class FakeProc:
        returncode = 0

    def fake_run(cmd, **_):
        assert cmd[0] == "wordstat" and cmd[1] == "collect"
        phrase = cmd[2]
        with lock:
            collected.append(phrase)
            run_no[0] += 1
            n = run_no[0]
        neighbors = graph.get(phrase, [])
        run_dir = cfg.results_dir / "runs" / f"20260909T{n:06d}Z-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "phrase": phrase,
                    "created_at": f"2026-09-09T{n // 100:02d}:{n % 100:02d}:00Z",
                    "exports": [
                        {"view": "top_related", "file": "top_related.parquet", "row_count": len(neighbors)}
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "Запросы со словами": [p for p, _ in neighbors],
                "Число запросов": [f for _, f in neighbors],
                TITLE.format(p=phrase): [""] * len(neighbors),
            }
        ).to_parquet(run_dir / "top_related.parquet", index=False)
        return FakeProc()

    monkeypatch.setattr("wordstat_trends.collect_service.subprocess.run", fake_run)
    monkeypatch.setattr(CollectService, "_ensure_chrome_alive", lambda self: None)
    monkeypatch.setattr("wordstat_trends.collect_service.date", _FixedDate)  # фиксируем «сегодня»
    return svc, collected


def state_dict(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_day1_collects_only_seeds_and_expands_them(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед а", 500), ("сосед б", 300)], "seed два": []}
    svc, collected = make_service(tmp_path, monkeypatch, graph=graph)

    svc.scheduler_once()

    assert collected == ["seed один", "seed два"]
    raw = state_dict(tmp_path / "graph_state.json")
    assert raw["visited"] == {"seed один": 0, "seed два": 0}
    assert {e["phrase"] for e in raw["frontier"]} == {"сосед а", "сосед б"}


def test_day2_picks_by_frequency_then_depth(tmp_path, monkeypatch):
    graph = {
        "seed один": [("частый", 900), ("редкий", 100)],
        "частый": [("внучок частого", 50)],
        "редкий": [],
    }
    svc, collected = make_service(tmp_path, monkeypatch, graph=graph)

    svc.scheduler_once()
    day1 = list(collected)
    svc.scheduler_once()

    assert day1 == ["seed один", "seed два"]
    # более частотное ребро раскрывается первым, глубина не меняет порядок
    assert collected == ["seed один", "seed два", "seed один", "seed два", "частый", "редкий"]
    raw = state_dict(tmp_path / "graph_state.json")
    assert raw["visited"] == {"seed один": 0, "seed два": 0, "частый": 1, "редкий": 1}
    assert {e["phrase"] for e in raw["frontier"]} == {"внучок частого"}


def test_daily_budget_limits_new_phrases(tmp_path, monkeypatch):
    graph = {"seed один": [("а", 300), ("б", 200), ("в", 100)]}
    svc, collected = make_service(tmp_path, monkeypatch, graph=graph, budget=2)

    svc.scheduler_once()
    svc.scheduler_once()

    # день 1 — seed-набор; день 2 — seed-набор + не больше двух новых фраз
    assert collected == ["seed один", "seed два", "seed один", "seed два", "а", "б"]
    raw = state_dict(tmp_path / "graph_state.json")
    assert raw["budget"] == {"date": TODAY.isoformat(), "spent": 2}
    assert {e["phrase"] for e in raw["frontier"]} == {"в"}


def test_budget_survives_restart_same_day(tmp_path, monkeypatch):
    graph = {"seed один": [("а", 300), ("б", 200)]}
    svc, collected = make_service(tmp_path, monkeypatch, graph=graph, budget=2)

    svc.scheduler_once()
    # «перезапуск»: новый экземпляр сервиса на тех же файлах состояния и фраз
    svc2, collected2 = make_service(tmp_path, monkeypatch, graph=graph, budget=2)
    svc2.scheduler_once()

    assert collected == ["seed один", "seed два"]
    assert collected2 == ["seed один", "seed два", "а", "б"]
    # третий прогон того же дня: бюджет исчерпан, только seed
    svc3, collected3 = make_service(tmp_path, monkeypatch, graph=graph, budget=2)
    svc3.scheduler_once()
    assert collected3 == ["seed один", "seed два"]


def test_graph_disabled_collects_seeds_only(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед", 500)]}
    svc, collected = make_service(tmp_path, monkeypatch, graph_enabled=False, graph=graph)

    svc.scheduler_once()

    assert collected == ["seed один", "seed два"]
    assert not (tmp_path / "graph_state.json").exists()


def test_http_triggered_run_does_not_touch_graph(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед", 500)]}
    svc, collected = make_service(tmp_path, monkeypatch, graph=graph)

    assert svc.try_start(["какая-то фраза"])
    for _ in range(200):
        if not svc.is_running():
            break
        threading.Event().wait(0.01)
    assert not svc.is_running()

    assert collected == ["какая-то фраза"]
    assert not (tmp_path / "graph_state.json").exists()


def test_repeated_advance_is_idempotent(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед а", 500)]}
    svc, _ = make_service(tmp_path, monkeypatch, graph=graph)

    svc.scheduler_once()
    first = state_dict(tmp_path / "graph_state.json")
    svc._graph_advance(["seed один", "seed два"])
    second = state_dict(tmp_path / "graph_state.json")

    assert first["visited"] == second["visited"]
    assert first["frontier"] == second["frontier"]


def test_depth_limit_keeps_deep_candidates_pending(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед", 500)], "сосед": [("внук", 5)]}
    svc, _ = make_service(tmp_path, monkeypatch, graph=graph)

    svc.scheduler_once()  # seed → сосед во фронтире
    svc.scheduler_once()  # сосед собран и раскрыт → внук на глубине 2

    raw = state_dict(tmp_path / "graph_state.json")
    assert raw["visited"] == {"seed один": 0, "seed два": 0, "сосед": 1}
    # GRAPH_MAX_DEPTH = 2: внук достижим, но не собран бюджетом этого дня
    assert "внук" in {e["phrase"] for e in raw["frontier"]}


def test_corrupt_state_disables_expansion_but_not_seed_collection(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед", 500)]}
    svc, collected = make_service(tmp_path, monkeypatch, graph=graph)
    (tmp_path / "graph_state.json").write_text("{битый", encoding="utf-8")

    svc.scheduler_once()

    assert collected == ["seed один", "seed два"]


def test_unwritable_state_file_skips_expansion_not_seed_run(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед", 500)]}
    svc, collected = make_service(tmp_path, monkeypatch, graph=graph)
    svc.scheduler_once()  # день 1: состояние создано, сосед во фронтире

    # «диск сломался»: запись состояния падает — расширение дня 2 пропускается
    monkeypatch.setattr(
        "wordstat_trends.collect_service.GraphStateFile.save",
        lambda self, state, budget: (_ for _ in ()).throw(OSError("нет места")),
    )
    svc.scheduler_once()

    # seed-набор собран, фразы из фронтира не запланированы
    assert collected == ["seed один", "seed два", "seed один", "seed два"]
    state, budget = GraphStateFile(tmp_path / "graph_state.json").load()
    assert budget == {"date": TODAY.isoformat(), "spent": 0}
    assert svc.status()["last_error"].startswith("OSError")


def test_store_roundtrip_through_service_file(tmp_path, monkeypatch):
    graph = {"seed один": [("сосед а", 500)]}
    svc, _ = make_service(tmp_path, monkeypatch, graph=graph)

    svc.scheduler_once()

    state, budget = GraphStateFile(tmp_path / "graph_state.json").load()
    assert state is not None
    assert state.visited == {"seed один": 0, "seed два": 0}
    assert budget == {"date": TODAY.isoformat(), "spent": 0}

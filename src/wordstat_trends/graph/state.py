"""Персистентное состояние обхода между прогонами сбора (issue #82).

``TraversalState`` ядра (#81) — плоские данные именно для этого файла: JSON
как есть, без ORM и отдельной схемы. Рядом со словарём живёт суточный бюджет
фраз (``{"date": "YYYY-MM-DD", "spent": N}``): бюджет — часть состояния
обхода, а не сервиса, поэтому и персистится с ним — перезапуск контейнера
среди дня не даёт обходу потратить больше лимита.

Запись атомарна (временный файл в том же каталоге + ``os.replace``): падение
посреди записи не оставляет наполовину переписанное состояние — обход либо
видит прошлый цельный файл, либо новый цельный.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from wordstat_trends.graph.traverse import FrontierEntry, TraversalState

log = logging.getLogger("wordstat_trends.graph.state")

#: Версия схемы файла состояния: изменение формата повышает число, старые
#: файлы честно отвергаются, а не читаются молча неверно.
STATE_VERSION = 1


class GraphStateError(Exception):
    """Файл состояния есть, но не читается как валидное состояние обхода."""


def _entry_to_dict(entry: FrontierEntry) -> dict:
    return {
        "phrase": entry.phrase,
        "frequency": entry.frequency,
        "depth": entry.depth,
        "is_seed": entry.is_seed,
    }


def _entry_from_dict(raw: dict) -> FrontierEntry:
    return FrontierEntry(
        phrase=raw["phrase"],
        frequency=float(raw["frequency"]),
        depth=int(raw["depth"]),
        is_seed=bool(raw.get("is_seed", False)),
    )


class GraphStateFile:
    """Загрузка/сохранение (состояние обхода + суточный бюджет) одним JSON."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> tuple[TraversalState | None, dict | None]:
        """Прочитать состояние; ``(None, None)`` — файла ещё нет (первый запуск).

        Битый файл — ``GraphStateError``: молча начать обход заново значило бы
        тихо потерять весь прогресс словаря, решение о сбросе — за оператором.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None, None
        except (OSError, json.JSONDecodeError) as exc:
            raise GraphStateError(f"файл состояния обхода не читается: {self.path} ({exc})") from exc
        try:
            return self._parse(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphStateError(f"файл состояния обхода имеет неверный формат: {self.path} ({exc})") from exc

    def _parse(self, raw: dict) -> tuple[TraversalState, dict | None]:
        version = raw.get("version")
        if version != STATE_VERSION:
            raise GraphStateError(f"версия файла состояния {version!r} != {STATE_VERSION}")
        state = TraversalState(
            visited={str(k): int(v) for k, v in raw["visited"].items()},
            frontier=[_entry_from_dict(e) for e in raw["frontier"]],
        )
        budget = raw.get("budget")
        if budget is not None:
            budget = {"date": str(budget["date"]), "spent": int(budget["spent"])}
        return state, budget

    def save(self, state: TraversalState, budget: dict | None) -> None:
        payload = {
            "version": STATE_VERSION,
            "visited": dict(state.visited),
            "frontier": [_entry_to_dict(e) for e in state.frontier],
            "budget": budget,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Атомарная замена: tmp в том же каталоге (одна ФС), затем os.replace.
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".graph_state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        log.info("состояние обхода сохранено: %s (словарь %d, фронтир %d)",
                 self.path, len(state.visited), len(state.frontier))

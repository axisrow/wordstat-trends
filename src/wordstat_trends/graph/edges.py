"""Провайдер рёбер поверх экспортов wordstat-cli (issue #82, Фаза 3 / А2).

Ядро обхода (#81) знает рёбра через абстрактный ``EdgeProvider``; здесь —
реальная реализация: ``wordstat collect`` для каждой фразы пишет прогон
``<RESULTS_DIR>/runs/<ts>-<slug>/`` с ``manifest.json`` и ``top_related.parquet``
(полные экспорты появились после wordstat-cli#63). Файлы манифеста читаются
как простой JSON без pydantic-моделей wordstat: манифест пишется той версией
cli, что стоит в контейнере, и жёсткая схема здесь связывала бы релизы.

Формат ``top_related.parquet`` (подтверждён живым прогоном 2026-09-09,
``wordstat collect "ремонт квартир"``): колонки с локализованными именами —
первая начинается с ``Запросы со словами`` (фраза), вторая с ``Число
запросов`` (частотность); третья колонка — заголовок таблицы, значений не
несёт. Имена колонок содержат период и регион, поэтому ищутся по префиксу,
а не по точному равенству.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from wordstat_trends.graph.traverse import EdgesNotAvailable, RelatedPhrase

log = logging.getLogger("wordstat_trends.graph.edges")

_PHRASE_PREFIX = "Запросы со словами"
_FREQUENCY_PREFIX = "Число запросов"


def _related_file_from_manifest(manifest: object) -> str | None:
    """Имя файла ``top_related`` из манифеста прогона (обычно дефолтное).

    Манифест — внешний файл, полной схемы у него здесь нет: не-dict в
    ``exports`` или не-строка в ``file`` просто не дают источника рёбер,
    а не роняют обход.
    """
    if not isinstance(manifest, dict):
        return None
    exports = manifest.get("exports")
    if not isinstance(exports, list):
        return None
    for export in exports:
        if isinstance(export, dict) and export.get("view") == "top_related":
            file = export.get("file")
            if isinstance(file, str) and file:
                return file
    return None


class ExportEdgeProvider:
    """Рёбра графа из прогонов сбора: фраза → соседи по ``top_related``.

    Индекс «фраза → самый свежий прогон» строится один раз при создании:
    ``neighbors`` вызывается по разу на каждую раскрываемую фразу, а скан
    каталога прогонов на каждый вызов перечитывал бы все манифесты заново.
    Для одной фразы берётся прогон с максимальным ``created_at`` — рёбра
    свежего окна приоритетнее старых. Прогон без ``top_related``-файла
    (пустой экспорт, неполный прогон) рёбер не даёт: ``neighbors`` про
    такую фразу поднимает ``EdgesNotAvailable``, и фраза остаётся
    нераскрытой, а не становится тупиком.
    """

    def __init__(self, runs_root: Path):
        self._runs_root = runs_root
        self._latest: dict[str, tuple[str, Path]] = {}
        for run_dir in self._iter_run_dirs():
            try:
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("прогон без читаемого манифеста, пропущен: %s (%s)", run_dir.name, exc)
                continue
            if not isinstance(manifest, dict):
                log.warning("манифест не является объектом, прогон пропущен: %s", run_dir.name)
                continue
            phrase = manifest.get("phrase")
            created_at = manifest.get("created_at")
            related = _related_file_from_manifest(manifest)
            if not isinstance(phrase, str) or not phrase:
                continue
            if related is None:
                continue
            if not isinstance(created_at, str):
                created_at = ""
            current = self._latest.get(phrase)
            if current is None or created_at > current[0]:
                self._latest[phrase] = (created_at, run_dir / related)

    def _iter_run_dirs(self) -> list[Path]:
        if not self._runs_root.is_dir():
            return []
        return sorted(p for p in self._runs_root.iterdir() if p.is_dir())

    def has_export(self, phrase: str) -> bool:
        """Есть ли у фразы прогон с ``top_related``-экспортом."""
        entry = self._latest.get(phrase)
        return entry is not None and entry[1].is_file()

    def neighbors(self, phrase: str) -> list[RelatedPhrase]:
        """Соседей по ``top_related``; ``EdgesNotAvailable`` — экспорта ещё нет.

        Различие принципиально для ядра: пустой список значит «у фразы нет
        похожих запросов» (тупик), отсутствие экспорта — «фраза ещё не
        собрана» (кандидат ждёт во фронтире расписания сбора).
        """
        entry = self._latest.get(phrase)
        if entry is None or not entry[1].is_file():
            raise EdgesNotAvailable(f"прогон с top_related для <{phrase}> не найден")
        if not self.has_export(phrase):  # pragma: no cover — индекс и проверка согласны
            raise EdgesNotAvailable(f"экспорт top_related для <{phrase}> не найден")
        frame = pd.read_parquet(entry[1])
        phrase_col = _find_column(frame, _PHRASE_PREFIX)
        freq_col = _find_column(frame, _FREQUENCY_PREFIX)
        if phrase_col is None or freq_col is None:
            log.warning("неожиданные колонки top_related у <%s>: %s", phrase, list(frame.columns))
            return []
        result: list[RelatedPhrase] = []
        for text, freq in zip(frame[phrase_col], frame[freq_col], strict=True):
            text = str(text).strip()
            if not text:
                continue
            try:
                frequency = int(freq)
            except (TypeError, ValueError):
                # битая строка экспорта (не-число/NaN в частотности) не должна
                # отменять раскрытие остальных рёбер фразы
                log.warning("нечисловая частотность у строки <%s> в экспорте <%s>, строка пропущена", text, phrase)
                continue
            result.append(RelatedPhrase(phrase=text, frequency=frequency))
        return result


def _find_column(frame: pd.DataFrame, prefix: str) -> str | None:
    for column in frame.columns:
        if str(column).startswith(prefix):
            return column
    return None

"""Ядро обхода графа запросов (issue #81, Фаза 3 / группа А).

«Похожие запросы» Вордстата — рёбра графа фраз. Ядро расширяет словарь от
seed-набора обходом в ширину с приоритетом по частотности: из фронтира
сначала раскрывается самая частотная фраза (не самая близкая к seed) —
частотные рёбра ``top_related`` дают больше новых фраз на один запрос к
источнику, а каждый запрос дорог (живой браузер, ``phrase_delay_s``).

Границы задачи, зафиксированные в issue #81:

- **Источник рёбер — абстрактный провайдер** (``EdgeProvider``: фраза →
  соседи с частотностями). Реального источника сейчас нет: ``top_related``
  возвращает 0 строк данных из-за бага в wordstat-cli (wordstat-cli#62),
  замер #2, docs/DATA.md. Ядро разрабатывается и тестируется офлайн на
  синтетических графах — интеграция со сбором это отдельная задача #82.
- **Лимиты обязательны, не настройка для подбора**: без ограничения глубины
  и объёма обход взрывается — фразы Вордстата ветвятся очень широко, а
  словарь нужен для витрины трендов, а не как полный слепок Яндекса.
- **Детерминированность**: при равных частотностях порядок раскрытия —
  по лексикографической фразе. Один и тот же граф даёт один и тот же
  словарь в одном и том же порядке.
- **Идемпотентность**: повторный запуск на том же состоянии ничего не
  добавляет и не обращается к провайдеру — персистентность состояния и
  дозапись между прогонами сбора это #82, ядро лишь обязано уметь
  продолжать с остановленного состояния без дублей.

Состояние (``TraversalState``) — плоские данные (dict/list): #82
персистит его как есть, ядро не знает о файлах и расписании.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Protocol

#: Частотность seed-фразы. У seed нет родительского ребра, поэтому в порядке
#: фронтира они идут первыми — но не «бесконечной частотой» (``inf`` в
#: сериализуемом состоянии дала бы нестандартный JSON ``Infinity``), а
#: явным флагом ``FrontierEntry.is_seed``. Значение конечно и в сравнении
#: не участвует.
SEED_FREQUENCY = 0.0


@dataclass(frozen=True)
class RelatedPhrase:
    """Сосед по графу: фраза + её частотность из рёбер ``top_related``."""

    phrase: str
    frequency: int


class EdgeProvider(Protocol):
    """Источник рёбер графа: фраза → соседи с частотностями.

    Единственная точка контакта ядра с внешним миром. Реальная реализация
    (чтение ``top_related`` из прогонов сбора) — #82. Пустой список —
    легальный ответ (тупик); «данных ещё нет» — ``EdgesNotAvailable``
    (фраза остаётся во фронтире, а не становится тупиком).
    """

    def neighbors(self, phrase: str) -> list[RelatedPhrase]:
        """Вернуть соседей фразы. Пустой список — легальный ответ (тупик)."""
        ...


class EdgesNotAvailable(Exception):
    """Рёбер для фразы пока нет: экспорт не собран (интеграция со сбором, #82).

    Ядро отвечает на это сохранением кандидата во фронтире: фраза не
    попадает в ``visited`` и дождётся сбора, вместо того чтобы навсегда
    стать «раскрытым тупиком».
    """


@dataclass(frozen=True)
class TraversalLimits:
    """Лимиты обхода: защита от взрыва объёма (требование #81).

    ``max_phrases`` — жёсткий потолок словаря; ``max_depth`` — сколько
    рёбер от seed допускается пройти (глубина 0 — сами seed). Оба — явные
    значения, задаваемые вызывающим кодом (#82 подставит свои константы с
    обоснованием под бюджет сбора), ядро проверяет их корректность.
    """

    max_phrases: int
    max_depth: int

    def validate(self) -> None:
        if self.max_phrases < 1:
            raise ValueError(f"max_phrases должен быть >= 1, получено {self.max_phrases}")
        if self.max_depth < 0:
            raise ValueError(f"max_depth должен быть >= 0, получено {self.max_depth}")


@dataclass(order=False)
class FrontierEntry:
    """Фраза во фронтире: кандидат на раскрытие.

    Порядок в куче: seed раньше найденных фраз (``is_seed``), затем по
    убыванию частотности, при равенстве по фразе (лексикографически) —
    детерминированный порядок без зависимости от того, в каком порядке
    провайдер вернул соседей. Все поля сериализуемы в стандартный JSON.
    """

    phrase: str
    frequency: float
    depth: int
    is_seed: bool = False

    def __lt__(self, other: FrontierEntry) -> bool:
        return (not self.is_seed, -self.frequency, self.phrase) < (
            not other.is_seed,
            -other.frequency,
            other.phrase,
        )


@dataclass
class TraversalState:
    """Состояние обхода: посещённые фразы + фронтир.

    ``visited`` — фраза → глубина появления; наличие ключа значит «фраза в
    словаре». Фронтир хранится списком: повторные записи одной фразы
    легальны (одно и то же ребро приходит из разных вершин),
    дедуплицируются при извлечении проверкой ``visited``. Плоские
    структуры намеренны — состояние сериализуется как есть (персистентность
    в #82), ядро про файлы и расписание не знает.
    """

    visited: dict[str, int] = field(default_factory=dict)
    frontier: list[FrontierEntry] = field(default_factory=list)

    @property
    def phrases(self) -> list[str]:
        """Посещённые фразы в детерминированном порядке (лексикография)."""
        return sorted(self.visited)


@dataclass(frozen=True)
class TraversalResult:
    """Итог обхода: словарь + честная пометка, чем он обрезан."""

    state: TraversalState
    provider_calls: int
    truncated_by_phrases: bool
    truncated_by_depth: bool

    @property
    def phrases(self) -> list[str]:
        return self.state.phrases


def run_traversal(
    seeds: list[str],
    provider: EdgeProvider,
    limits: TraversalLimits,
    state: TraversalState | None = None,
) -> TraversalResult:
    """Расширить ``state`` обходом от ``seeds`` в пределах ``limits``.

    Первый запуск (``state=None``): seed попадают во фронтир на глубине 0
    (флаг ``is_seed``, раскрываются раньше найденных фраз) и проходят через
    общий цикл — visited у ядра
    значит «в словаре И раскрыта», поэтому seed не может быть отсечён
    проверкой дубля. Продолжение (передано состояние прошлого запуска):
    уже посещённые seed не переустанавливаются, обход идёт дальше с
    сохранённого фронтира. Повторный запуск на исчерпанном состоянии не
    обращается к провайдеру — идемпотентность из #81.

    Кандидаты, не помещавшиеся в ``max_phrases`` или глубже ``max_depth``,
    остаются в ``state.frontier``: продолжение с расширенными лимитами
    подберёт их без повторных запросов к провайдеру по уже раскрытым
    фразам.

    ``truncated_by_phrases``/``truncated_by_depth`` — факты об обрезке, а
    не ошибки: расширять ли лимиты, решает #82.
    """
    limits.validate()
    state = state if state is not None else TraversalState()

    in_frontier = {entry.phrase for entry in state.frontier}
    for seed in sorted(set(seeds)):
        if seed not in state.visited and seed not in in_frontier:
            state.frontier.append(FrontierEntry(phrase=seed, frequency=SEED_FREQUENCY, depth=0, is_seed=True))

    heap: list[FrontierEntry] = list(state.frontier)
    heapq.heapify(heap)
    state.frontier = []

    provider_calls = 0
    truncated_by_depth = False
    truncated_by_phrases = False

    while heap:
        entry = heapq.heappop(heap)
        if entry.phrase in state.visited:
            continue  # дубликат ребра: фраза уже в словаре
        if entry.depth > limits.max_depth:
            truncated_by_depth = True
            state.frontier.append(entry)
            continue
        if len(state.visited) >= limits.max_phrases:
            # причина сохранения кандидата фиксируется по факту ветки:
            # «во фронтире что-то осталось» само по себе не значит обрезку
            # по фразам (могли остаться только depth-обрезанные записи)
            truncated_by_phrases = True
            state.frontier.append(entry)
            continue

        state.visited[entry.phrase] = entry.depth
        try:
            neighbors = provider.neighbors(entry.phrase)
        except EdgesNotAvailable:
            # Данных для раскрытия ещё нет (#82): кандидат возвращается во
            # фронтир и не считается ни раскрытым, ни тупиком.
            del state.visited[entry.phrase]
            state.frontier.append(entry)
            continue
        provider_calls += 1
        for neighbor in neighbors:
            if neighbor.phrase == entry.phrase:
                continue  # петля на самого себя — не ребро
            if neighbor.phrase in state.visited:
                continue
            heapq.heappush(
                heap,
                FrontierEntry(phrase=neighbor.phrase, frequency=float(neighbor.frequency), depth=entry.depth + 1),
            )

    return TraversalResult(
        state=state,
        provider_calls=provider_calls,
        truncated_by_phrases=truncated_by_phrases,
        truncated_by_depth=truncated_by_depth,
    )

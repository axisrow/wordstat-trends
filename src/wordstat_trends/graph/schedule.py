"""Планирование обхода в конвейере сбора (issue #82): бюджет и приоритеты.

Здесь живут константы лимитов обхода и выбор фраз дня из фронтира: в
порядке приоритета (частотность ребра ``top_related`` по убыванию, затем
глубина появления, затем лексикография — порядок ``FrontierEntry`` ядра)
в пределах остатка суточного бюджета. Фразы без экспорта остаются во
фронтире сами — ядро сохраняет их при ``EdgesNotAvailable``.
"""

from __future__ import annotations

import heapq

from wordstat_trends.graph.traverse import TraversalState

#: Потолок словаря обхода. Стоимость фразы словаря — один прогон сбора при
#: появлении (≈90 c + ``PHRASE_DELAY_S``; суточный прогон после этого
#: пересобирает только seed-набор и фразы дня, а не весь словарь). Потолок
#: защищает не время суток, а сам масштаб: словарь нужен как основа витрины
#: трендов тематики, а не слепок Вордстата целиком — рёбра ветвятся широко,
#: и без потолка «похожие на похожие» разбегаются далеко за тематику
#: (глубину с той же целью ограничивает GRAPH_MAX_DEPTH). Темп роста к тому
#: же пейсируется суточным бюджетом, потолок — лишь предохранитель от
#: неосторожного его поднятия.
GRAPH_MAX_PHRASES = 1000

#: Глубина обхода от seed. На рёбрах «похожих запросов» семантика уплывает
#: быстро: уже на глубине 2 от «ремонт квартир» оказываются «дизайн
#: интерьера» и «строительные компании». Глубже — слив смыслов, а не
#: расширение словаря тематики.
GRAPH_MAX_DEPTH = 2


def pick_pending(state: TraversalState, limit: int, exclude: set[str]) -> list[str]:
    """Фразы дня из ожидающих сбора, в порядке приоритета фронтира.

    ``limit`` — остаток суточного бюджета (<= 0 — пустой список), ``exclude``
    — фразы, уже собранные этим прогоном по расписанию (seed-набор).
    """
    if limit <= 0:
        return []
    heap = [
        entry
        for entry in state.frontier
        if entry.phrase not in state.visited and entry.phrase not in exclude
    ]
    heapq.heapify(heap)
    picked: list[str] = []
    seen: set[str] = set()
    while heap and len(picked) < limit:
        entry = heapq.heappop(heap)
        if entry.phrase in seen:
            continue
        seen.add(entry.phrase)
        picked.append(entry.phrase)
    return picked

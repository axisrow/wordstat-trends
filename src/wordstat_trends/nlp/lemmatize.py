"""Лемматизация фраз запросов (issue #68, Фаза 4).

`pymorphy3` — форк-преемник заброшенного pymorphy2 (0.9.1 от 2020-09-26,
ломается на новых Python), API совместим. Нормализация словоформ нужна обоим
конвейерам Фазы 4: TF-IDF по леммам вместо словоформ и более чистый вход
эмбеддингам.

Воркер заполняет `Morph` один раз на процесс: словарь (~секунда загрузки)
общий для всех фраз, а `parse` потокобезопасен на read-only инстансе.
"""

from __future__ import annotations

import re
from functools import lru_cache

from pymorphy3 import MorphAnalyzer

# Токен — слово из букв (включая кириллицу) и дефиса внутри слова.
# Цифры и пунктуация фразы запроса не несут смысла для лемматизации.
_TOKEN_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*")


@lru_cache(maxsize=1)
def _morph() -> MorphAnalyzer:
    """Единственный анализатор на процесс (загрузка словаря — дорогое)."""
    return MorphAnalyzer()


def tokenize(phrase: str) -> list[str]:
    """Слова фразы в нижнем регистре; пунктуация и цифры отбрасываются."""
    return _TOKEN_RE.findall(phrase.lower())


def lemmatize(phrase: str) -> list[str]:
    """Леммы слов фразы в исходном порядке: «курсы английского» → [курс, английский]."""
    return [_morph().parse(token)[0].normal_form for token in tokenize(phrase)]


def lemmatized_phrase(phrase: str) -> str:
    """Фраза из лемм через пробел — вход для TF-IDF и эмбеддингов."""
    return " ".join(lemmatize(phrase))

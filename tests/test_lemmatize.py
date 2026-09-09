"""Тесты лемматизации фраз (issue #68, Фаза 4).

Фразы — реальные запросы из фикстур Фазы 1 (docs/DATA.md): каждая dynamics-
фикстура в tests/fixtures/ собрана для одной фразы.
"""

import pytest

from wordstat_trends.nlp.lemmatize import lemmatize, lemmatized_phrase, tokenize

# Три профильные фразы замера #2 (docs/DATA.md → «Прогон»).
FIXTURE_PHRASES = [
    ("купить телефон", ["купить", "телефон"]),
    ("курсы английского языка", ["курс", "английский", "язык"]),
    ("новогодние подарки", ["новогодний", "подарок"]),
]


@pytest.mark.parametrize(("phrase", "expected"), FIXTURE_PHRASES)
def test_fixture_phrases_lemmatized(phrase: str, expected: list[str]):
    assert lemmatize(phrase) == expected
    assert lemmatized_phrase(phrase) == " ".join(expected)


def test_tokenize_lowercase_and_punctuation():
    assert tokenize("Купить ТЕЛЕФОН!!! (Москва)") == ["купить", "телефон", "москва"]


def test_digits_are_dropped():
    # Вордстат отдаёт фразы вроде «iphone 15 pro» — цифра не токен для лемм.
    assert lemmatize("iphone 15 pro") == ["iphone", "pro"]


def test_word_order_preserved():
    assert lemmatize("языка английского") == ["язык", "английский"]


def test_empty_phrase_gives_empty_result():
    assert lemmatize("") == []
    assert lemmatized_phrase("") == ""

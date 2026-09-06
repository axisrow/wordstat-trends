"""Тесты CI-гейта «без LLM» (scripts/check_no_llm.py, issue #4)."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_no_llm import (
    ALLOWED_SUBTREES,
    FORBIDDEN_PACKAGES,
    find_violations,
    is_forbidden,
    normalize,
)


def make_lock(packages: dict[str, list[str]]) -> dict:
    """Синтетический uv.lock: {имя: [зависимости]}; корень — wordstat-trends."""
    entries = [
        {
            "name": name,
            "version": "0.0.0",
            "dependencies": [{"name": dep} for dep in deps],
        }
        for name, deps in packages.items()
    ]
    return {"package": entries}


def test_clean_lock_has_no_violations():
    lock = make_lock(
        {
            "wordstat-trends": ["sktime", "wordstat"],
            "wordstat": ["browser-use"],
            "sktime": ["numpy"],
            "numpy": [],
        }
    )
    assert find_violations(lock) == []


def test_direct_forbidden_dependency_is_violation():
    lock = make_lock({"wordstat-trends": ["openai"], "openai": []})
    assert find_violations(lock) == ["openai"]


def test_transitive_forbidden_dependency_outside_allowlist_is_violation():
    lock = make_lock(
        {
            "wordstat-trends": ["wordstat"],
            "wordstat": ["langchain-community"],
            "langchain-community": [],
        }
    )
    assert find_violations(lock) == ["langchain-community"]


def test_forbidden_only_under_allowlisted_subtree_is_allowed():
    # Реальная конфигурация: openai приходит только из browser-use.
    lock = make_lock(
        {
            "wordstat-trends": ["wordstat"],
            "wordstat": ["browser-use"],
            "browser-use": ["openai", "anthropic", "groq"],
            "openai": [],
            "anthropic": [],
            "groq": [],
        }
    )
    assert find_violations(lock) == []


def test_forbidden_via_allowlist_and_other_path_is_violation():
    # Второй путь к тому же пакету — уже не «чужое поддерево».
    lock = make_lock(
        {
            "wordstat-trends": ["wordstat", "new-shiny"],
            "wordstat": ["browser-use"],
            "browser-use": ["openai"],
            "new-shiny": ["openai"],
            "openai": [],
        }
    )
    assert find_violations(lock) == ["openai"]


def test_direct_dependency_shadowing_allowlisted_name_is_violation():
    # Прямая зависимость openai в проекте падает, даже если имя где-то
    # уже разрешено внутри browser-use.
    lock = make_lock(
        {
            "wordstat-trends": ["wordstat", "openai"],
            "wordstat": ["browser-use"],
            "browser-use": ["openai"],
            "openai": [],
        }
    )
    assert find_violations(lock) == ["openai"]


def test_prefix_pattern_matches_family():
    lock = make_lock({"wordstat-trends": ["langchain-some-new-addon"], "langchain-some-new-addon": []})
    assert find_violations(lock) == ["langchain-some-new-addon"]


def test_root_dev_and_extra_dependencies_are_violations():
    # uv.lock хранит dev/extra-зависимости корня не в dependencies, а в
    # [package.optional-dependencies] и [package.metadata].requires-dist:
    # LLM-клиент из extra попадает в окружение через uv sync --extra,
    # поэтому гейт ловит его наравне с основными зависимостями.
    lock = {
        "package": [
            {
                "name": "wordstat-trends",
                "version": "0.0.0",
                "dependencies": [{"name": "sktime"}],
                "optional-dependencies": {"dev": [{"name": "gigachat"}]},
                "metadata": {
                    "requires-dist": [
                        {"name": "sktime"},
                        {"name": "openai", "marker": "extra == 'dev'"},
                    ]
                },
            },
            {"name": "sktime", "version": "1.1.0"},
            {"name": "gigachat", "version": "0.0.0"},
            {"name": "openai", "version": "0.0.0"},
        ]
    }
    assert find_violations(lock) == ["gigachat", "openai"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("OpenAI", "openai"),
        ("LangChain_Community", "langchain-community"),
        ("Zhipu.AI", "zhipu-ai"),
        ("zhipu..ai", "zhipu-ai"),
        ("Open_AI.tools", "open-ai-tools"),
    ],
)
def test_normalize_pep503(name: str, expected: str):
    assert normalize(name) == expected


def test_is_forbidden_no_substring_false_positives():
    # Подстрока 'openai' в чужом имени — не повод; только имена и патчи.
    assert not is_forbidden("openai-tools-extra")
    assert not is_forbidden("notopenai")
    assert is_forbidden("openai")
    assert is_forbidden("LangChain-Community")


def test_every_forbidden_entry_is_lowercase_normalized():
    for pattern in FORBIDDEN_PACKAGES:
        stem = pattern[:-2] if pattern.endswith("-*") else pattern
        assert normalize(stem) == stem


def test_allowlist_entries_have_rationale():
    for key, reason in ALLOWED_SUBTREES.items():
        assert len(reason) > 20, f"обоснование для {key} слишком короткое"


def test_real_uv_lock_is_clean():
    """Позитивный сценарий на настоящем uv.lock репозитория."""
    repo_root = Path(__file__).resolve().parent.parent
    lock_path = repo_root / "uv.lock"
    if not lock_path.is_file():
        pytest.skip("uv.lock отсутствует")
    import tomllib

    with lock_path.open("rb") as handle:
        assert find_violations(tomllib.load(handle)) == []


def test_main_exit_codes(tmp_path: Path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "check_no_llm.py"

    clean = tmp_path / "clean.lock"
    clean.write_text(
        '[[package]]\nname = "wordstat-trends"\nversion = "0.0.0"\n\n'
        '[[package]]\nname = "sktime"\nversion = "1.1.0"\n',
        encoding="utf-8",
    )
    dirty = tmp_path / "dirty.lock"
    dirty.write_text(
        '[[package]]\nname = "wordstat-trends"\nversion = "0.0.0"\n'
        'dependencies = [{ name = "gigachat" }]\n\n'
        '[[package]]\nname = "gigachat"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )

    ok = subprocess.run(
        [sys.executable, str(script), str(clean)], capture_output=True, text=True, check=False
    )
    assert ok.returncode == 0

    bad = subprocess.run(
        [sys.executable, str(script), str(dirty)], capture_output=True, text=True, check=False
    )
    assert bad.returncode == 1
    assert "gigachat" in bad.stderr
    assert "БЕЗ LLM" in bad.stderr

    missing = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "nope.lock")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2


def test_lock_without_root_package_fails_loudly(tmp_path: Path):
    # Регрессия fail-open: без корневого пакета reachable пуст и гейт молча
    # печатал бы OK при любом содержимом lock. Теперь — exit 2 и сообщение,
    # а не зелёный прогон.
    script = Path(__file__).resolve().parent.parent / "scripts" / "check_no_llm.py"
    rootless = tmp_path / "rootless.lock"
    rootless.write_text('[[package]]\nname = "sktime"\nversion = "1.1.0"\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), str(rootless)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "wordstat-trends" in result.stderr

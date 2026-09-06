"""CI-гейт «без LLM»: падает, если в дереве зависимостей появился LLM-клиент.

Принцип проекта (README, «Принцип: без LLM, только классический ML»):
ни одно число, ни один кластер, ни один тренд не получаются вызовом LLM.
Установка LLM-клиента в окружение проекта размывает этот принцип, даже если
код его пока не использует, — поэтому проверяется **разрешённое дерево**
(uv.lock), а не только прямые зависимости pyproject.toml: транзитивная
зависимость протаскивает LLM-клиент так же надёжно. Dev- и extra-зависимости
корня проверяются наравне с основными: `uv sync --extra` ставит их в то же
окружение.

Запуск: `python scripts/check_no_llm.py [путь/к/uv.lock]` (из корня репо путь
можно не указывать). Нужен только stdlib (`tomllib`) — шаг гейта в CI идёт
до `uv sync`.

Сравнение — по **нормализованным именам пакетов** (PEP 503: lower + дефисы),
никогда по подстроке: пакет вида `notopenai-utils` нарушением не считается.

Исключения: LLM-клиенты, приходящие **исключительно** из allowlist-поддерева
(см. ALLOWED_SUBTREES), нарушением не считаются — это чужие транзитивные
зависимости инструмента сбора, сам проект их не импортирует и ключей им не
даёт. Если запрещённый пакет становится достижимым любым другим путём
(прямая зависимость, новое поддерево), гейт падает. Расширять allowlist
можно только с обоснованием в комментарии рядом с ALLOWED_SUBTREES.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# LLM API-клиенты и оркестраторы. Суффикс `-*` — префиксный патч на семейство
# (langchain-*, langchain-community, ...). Сознательно НЕ в списке:
# transformers/sentence-transformers/torch — локальные модели, не вызовы LLM;
# tiktoken — токенайзер. Список один, расширение — просто добавить строку.
FORBIDDEN_PACKAGES: tuple[str, ...] = (
    "openai",
    "anthropic",
    "langchain",
    "langchain-*",
    "gigachat",
    "gigachain",
    "yandexgpt",
    "cohere",
    "mistralai",
    "ollama",
    "litellm",
    "google-generativeai",
    "google-genai",
    "dashscope",
    "zhipuai",
    "together",
    "groq",
    "deepseek",
    "openrouter",
    "portkey",
)

# Поддеревья, внутри которых запрещённые пакеты разрешены. Ключ — имя пакета,
# значение — обоснование, попадающее в сообщение об ошибке и ревью.
ALLOWED_SUBTREES: dict[str, str] = {
    # browser-use (через wordstat -> wordstat-cli) тащит openai/anthropic/
    # google-genai/groq/ollama как необязательные клиники своих LLM-баэкендов;
    # wordstat-cli их не использует — сбор идёт через CDP к живому Chrome.
    "browser-use": "транзитивная зависимость wordstat-cli (сбор через CDP), сам проект LLM-клиенты не импортирует",
}

# Корень графа зависимостей — сам проект в uv.lock (editable-запись).
ROOT_PACKAGE = "wordstat-trends"

PRINCIPLE_URL = (
    "https://github.com/axisrow/wordstat-trends"
    "#принцип-без-llm-только-классический-ml"
)
ISSUE_URL = "https://github.com/axisrow/wordstat-trends/issues/4"


def normalize(name: str) -> str:
    """PEP 503: нижний регистр, повторяющиеся `-`/`_`/`.` -> один дефис."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def is_forbidden(package: str) -> bool:
    """Точное совпадение имени или префиксный патч вида `langchain-*`."""
    normalized = normalize(package)
    for pattern in FORBIDDEN_PACKAGES:
        if pattern.endswith("-*"):
            if normalized.startswith(pattern[:-1]):  # без `*`: "langchain-"
                return True
        elif normalized == pattern:
            return True
    return False


def root_declared_deps(entry: dict) -> list[str]:
    """Имена dev/extra-зависимостей корневого пакета из его записи в lock.

    В uv.lock они не попадают в dependencies корня: extra-зависимости лежат
    в [package.optional-dependencies], полный список объявленных требований
    со всеми маркерами — в [package.metadata] requires-dist. Без них гейт
    пропустил бы LLM-клиент, добавленный в extra, хотя `uv sync --extra`
    ставит его прямо в CI-окружение.
    """
    names = [
        normalize(req["name"])
        for req in entry.get("metadata", {}).get("requires-dist", [])
        if "name" in req
    ]
    for group in entry.get("optional-dependencies", {}).values():
        names.extend(normalize(dep["name"]) for dep in group)
    return names


def build_graph(lock: dict) -> dict[str, list[str]]:
    """uv.lock -> {имя пакета: [имена прямых зависимостей]}."""
    graph: dict[str, list[str]] = {}
    for entry in lock.get("package", []):
        name = normalize(entry["name"])
        deps = [normalize(dep["name"]) for dep in entry.get("dependencies", [])]
        # Только у корня: в lock у сторонних пакетов тоже бывают
        # optional-dependencies, но их extras по умолчанию не ставятся —
        # идти по ним значило бы проверять недостижимое.
        if name == ROOT_PACKAGE:
            deps.extend(root_declared_deps(entry))
        # Одно имя может встретиться несколько раз (маркеры платформ) — объединяем.
        graph[name] = sorted(set(graph.get(name, [])) | set(deps))
    return graph


def reachable_without_allowlist(graph: dict[str, list[str]]) -> set[str]:
    """Пакеты, достижимые от корня проекта, не заходя в allowlist-поддеревья.

    Поддерево разрешённого корня отсекается целиком: его содержимое недостижимо
    «напрямую», и именно поэтому там LLM-клиенты не нарушают принцип.
    """
    if ROOT_PACKAGE not in graph:
        # Fail-open здесь навсегда спрятал бы гейт: пустой reachable даёт OK
        # при любом содержимом lock. Громко падаем — main() разведёт это в
        # exit 2, отличая от нарушения (exit 1).
        raise ValueError(
            f"корневой пакет {ROOT_PACKAGE} не найден в lock — "
            "переименован в pyproject.toml или передан чужой файл"
        )
    seen: set[str] = set()
    stack = [ROOT_PACKAGE]
    while stack:
        node = stack.pop()
        if node in seen or node in ALLOWED_SUBTREES:
            continue
        seen.add(node)
        stack.extend(dep for dep in graph.get(node, []) if dep not in seen)
    return seen


def find_violations(lock: dict) -> list[str]:
    """Запрещённые пакеты, достижимые вне allowlist-поддеревьев, по алфавиту."""
    graph = build_graph(lock)
    reachable = reachable_without_allowlist(graph)
    return sorted(name for name in reachable if is_forbidden(name))


def load_lock(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверяет uv.lock на появление LLM-зависимостей (принцип «без LLM»)."
    )
    default_lock = Path(__file__).resolve().parent.parent / "uv.lock"
    parser.add_argument("lock", nargs="?", type=Path, default=default_lock)
    args = parser.parse_args(argv)

    if not args.lock.is_file():
        print(f"no-llm: {args.lock} не найден — гейт не смог проверить зависимости", file=sys.stderr)
        return 2

    try:
        violations = find_violations(load_lock(args.lock))
    except ValueError as error:
        print(f"no-llm: {error}", file=sys.stderr)
        return 2

    if not violations:
        print(f"no-llm: OK — запрещённых LLM-пакетов в {args.lock.name} нет")
        return 0

    print("no-llm: НАРУШЕНИЕ ПРИНЦИПА «БЕЗ LLM»", file=sys.stderr)
    print(file=sys.stderr)
    print("В дереве зависимостей (uv.lock) появились LLM-пакеты:", file=sys.stderr)
    for name in violations:
        reason = ALLOWED_SUBTREES.get(name)
        # Allowlist применяется к поддеревьям, а не к отдельным пакетам:
        # запрещённое имя, найденное вне разрешённого поддерева, нарушению
        # подлежит даже если совпадает с ключом ALLOWED_SUBTREES.
        note = f" (допустимо только внутри поддерева: {reason})" if reason else ""
        print(f"  - {name}{note}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"Принцип проекта: {PRINCIPLE_URL}", file=sys.stderr)
    print("Это осознанный гейт, а не поломка сборки; исключения оформляются", file=sys.stderr)
    print(f"в ALLOWED_SUBTREES scripts/check_no_llm.py с обоснованием ({ISSUE_URL}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

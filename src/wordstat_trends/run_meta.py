"""Метаданные прогона эксперимента: единый хелпер вместо копипасты (issue #95).

Каждый экспериментальный скрипт пишет рядом с результатами слот ``run`` из
:data:`run_metadata`: git-коммит, хэш/путь входных датасетов, seed,
версии ключевых пакетов и Python. Этого достаточно, чтобы от коммита до
числа дошёл тот же код на тех же данных (docs/REPRODUCE.md).

Контракт детерминизма: в метаданных НЕТ времени запуска и абсолютных
путей — повторный запуск того же эксперимента на том же коммите обязан
дать побайтово тот же артефакт (на этом построен CI-шаг проверки
пересчёта). Пути входных датасетов записываются относительно корня репо.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

#: Пакеты, версии которых меняют числа экспериментов (стат-бэкенды и
#: численная база). ``scikit-learn`` — под дистрибутивным именем.
TRACKED_PACKAGES = ("sktime", "statsmodels", "statsforecast", "scikit-learn", "numpy", "pandas")

#: Значение git-коммита, когда репозиторий недоступен (установка не из
#: git-checkout) — честная пометка вместо падения или пустого поля.
COMMIT_UNKNOWN = "unknown"

#: Значение версии пакета, когда он не установлен в текущем окружении
#: (как COMMIT_UNKNOWN для git): метаданные — диагностика, а не гейт,
#: честная пометка лучше traceback посреди эксперимента.
VERSION_UNKNOWN = "not-installed"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return VERSION_UNKNOWN


def _git_commit(repo_root: Path) -> tuple[str, bool]:
    """(коммит HEAD, есть ли незакоммиченные изменения). Обе проверки —
    ``git`` в корне репо; вне git-репозитория — (COMMIT_UNKNOWN, False)."""
    try:
        commit = subprocess.run(  # noqa: S603 — фиксированные аргументы, без shell
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return COMMIT_UNKNOWN, False
    status = subprocess.run(  # noqa: S603 — фиксированные аргументы, без shell
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def _dataset_entry(path: Path, repo_root: Path) -> dict:
    """Путь относительно корня репо (вне репо — абсолютный, напр. run-каталог
    с живым сбором) + sha256 + размер байтов."""
    try:
        shown = str(path.resolve().relative_to(repo_root))
    except ValueError:
        shown = str(path.resolve())
    return {
        "path": shown,
        "sha256": dataset_sha256(path),
        "bytes": path.stat().st_size,
    }


def dataset_sha256(path: Path) -> str:
    """sha256 файла входного датасета (побайтово, без нормализации формата:
    у выгрузок Вордстата CR-only переводы строк сами по себе часть формата)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_metadata(inputs: tuple[Path, ...] = (), seed: int | tuple[int, ...] | None = None) -> dict:
    """Собрать слот ``run`` для артефакта эксперимента.

    - ``inputs`` — входные датасеты прогона (фикстуры, run-каталоги);
      для скриптов с датасетом, зашитым в код (issue #13), — пусто.
    - ``seed`` — seed стохастики прогона из :mod:`wordstat_trends.seeds`;
      ``None`` значит «в этом прогоне стохастики нет» (см. аудит в
      докстринге seeds.py) — это фиксируется явно, а не пропуском поля.
    """
    repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_commit(repo_root)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
        "seed": seed,
        "inputs": [_dataset_entry(path, repo_root) for path in inputs],
    }


__all__ = ["COMMIT_UNKNOWN", "TRACKED_PACKAGES", "VERSION_UNKNOWN", "dataset_sha256", "run_metadata"]

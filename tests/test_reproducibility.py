"""Воспроизводимость экспериментов (issue #95): seed-политика, метаданные
прогона, побайтовый пересчёт самого дешёвого детерминированного эксперимента.

Сlow-тест rerun (запускается отдельным шагом CI, см. ci.yml) гоняет
``issue13_tfidf_vs_berta.py --tfidf`` дважды: это самый дешёвый
детерминированный прогон репозитория (TF-IDF по леммам + k-means на
фиксированной сетке seed'ов, без ETS-фитов и без extra ``nlp``).
Артефакт включает слот ``run`` из run_meta — время запуска и абсолютные
пути в нём отсутствуют по контракту, поэтому побайтовое совпадение двух
запусков на одном коммите достижимо и обязательно.
"""

from __future__ import annotations

import json
import subprocess  # noqa: S404 — фиксированные аргументы, без shell
import sys
from pathlib import Path

import pytest

from wordstat_trends import run_meta, seeds
from wordstat_trends.forecasting.models import RANDOM_STATE

REPO = Path(__file__).resolve().parent.parent


def test_seeds_policy_constants() -> None:
    """Единая точка правды о seed'ах: модели и сетка k-means берут значения
    отсюда, а не разрозненными литералами по скриптам."""
    assert seeds.DEFAULT_SEED == RANDOM_STATE == 42
    assert seeds.KMEANS_SEEDS == (0, 1, 2, 42)


def test_run_metadata_records_commit_input_hash_and_versions(tmp_path: Path) -> None:
    dataset = tmp_path / "dynamics.csv"
    dataset.write_bytes(b"per;t;\r")

    meta = run_meta.run_metadata(inputs=(dataset,), seed=7)

    assert meta["seed"] == 7
    assert meta["packages"]["scikit-learn"]  # дистрибутивное имя, не import-имя
    assert set(meta["packages"]) == set(run_meta.TRACKED_PACKAGES)
    # Неустановленный пакет — честная пометка, не PackageNotFoundError
    assert run_meta._package_version("wordstat-trends-definitely-not-installed") == run_meta.VERSION_UNKNOWN
    assert meta["python"]
    # В git-checkout коммит известен; «unknown» допустим только вне репо
    if meta["git_commit"] != run_meta.COMMIT_UNKNOWN:
        assert len(meta["git_commit"]) == 40
    (entry,) = meta["inputs"]
    assert entry["sha256"] == run_meta.dataset_sha256(dataset)
    assert entry["bytes"] == dataset.stat().st_size
    # Путь вне репо — абсолютный (tmp), внутри репо был бы относительным
    assert entry["path"] == str(dataset)


def test_run_metadata_is_deterministic_between_calls() -> None:
    """Контракт побайтового пересчёта: два вызова на одном дереве дают
    одинаковый слот (нет времени запуска и иных меняющихся полей)."""
    fixture = REPO / "tests/fixtures/dynamics_daily_seasonal_mvp.csv"
    assert run_meta.run_metadata(inputs=(fixture,), seed=None) == run_meta.run_metadata(
        inputs=(fixture,), seed=None
    )


@pytest.mark.slow
def test_cheapest_experiment_rerun_is_byte_identical() -> None:
    """Повторный запуск самого дешёвого детерминированного эксперимента
    даёт побайтово тот же JSON (CI-шаг проверки пересчёта, issue #95)."""
    outputs = []
    for _ in range(2):
        proc = subprocess.run(  # noqa: S603 — фиксированные аргументы, без shell
            [sys.executable, str(REPO / "scripts/issue13_tfidf_vs_berta.py"), "--tfidf"],
            capture_output=True,
            check=True,
        )
        outputs.append(proc.stdout)

    first, second = outputs
    if first != second:
        # Понятное сообщение при любом виде расхождения: сначала убедиться,
        # что stdout — валидный JSON (диагностический print в скрипте не
        # должен превращать эту диагностику в нечитаемый traceback от loads).
        reports = []
        for output in (first, second):
            try:
                reports.append(json.loads(output))
            except json.JSONDecodeError as error:
                pytest.fail(f"stdout скрипта не является JSON ({error}): {output[:200]!r}...")
        first_json, second_json = reports
        # Что именно разошлось: слот run (метаданные) против чисел.
        if first_json.get("run") != second_json.get("run"):
            pytest.fail(
                "Расхождение в метаданных прогона (слот run) между двумя "
                f"запусками: {first_json.get('run')!r} != {second_json.get('run')!r}"
            )
        pytest.fail(
            "Повторный запуск issue13 (--tfidf) не воспроизводится побайтово — "
            "числа эксперимента изменились между двумя прогонами на одном "
            f"коммите. Размеры: {len(first)} vs {len(second)} байт."
        )

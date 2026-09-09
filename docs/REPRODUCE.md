# Воспроизведение экспериментов: от коммита до числа

Тезис проекта (эпик #1): каждое число на витрине обязано пересчитываться
в то же самое. Этот документ — порядок запуска, единый для всех
экспериментов (issue #95). Три опоры воспроизводимости:

1. **Окружение** — версии зафиксированы `uv.lock` и `.python-version`
   (Python 3.12, #3): `uv sync --extra dev` ставит ровно то, что в lock.
2. **Seed-политика** — единая точка правды `src/wordstat_trends/seeds.py`
   (`DEFAULT_SEED`, `KMEANS_SEEDS`): константа вместо разрозненных
   литералов, плюс аудит всех мест стохастики (включая места, где
   стохастики нет — и исторические seed'ы закрытых прогонов, которые
   сознательно не меняются).
3. **Метаданные прогона** — каждый экспериментальный скрипт пишет в
   артефакт слот `run` из `wordstat_trends.run_meta.run_metadata`:
   git-коммит, хэш/путь входных датасетов, seed, версии ключевых пакетов
   (sktime/statsmodels/statsforecast/scikit-learn/numpy/pandas), Python.
   В метаданных нет времени запуска и абсолютных путей внутри репо —
   повторный запуск на том же коммите даёт побайтово тот же артефакт.

Детерминизм вещественной арифметики: скрипты ограничивают BLAS/OMP-потоки
(`OMP_NUM_THREADS` и др.) ДО импорта numpy — не менять порядок этих
строк при правке скриптов.

## Порядок запуска

```bash
# 1. Окружение ровно из lock-файла
uv sync --extra dev

# 2. Эксперименты на фикстурах (входы лежат в tests/fixtures, в репо):
python scripts/experiment_vector_d.py            # MVP #23, итерация 1 (sp=12)
python scripts/experiment_vector_d.py --daily    # MVP #23, итерация 2 (sp=7)
python scripts/issue34_window_threshold.py       # #34: порог окна sp=7
python scripts/issue13_tfidf_vs_berta.py         # #13: TF-IDF vs BERTA
python scripts/issue13_tfidf_vs_berta.py --tfidf # то же без extra `nlp` (BERTA-конвейер SKIPPED)
python scripts/issue23_combined_vector.py        # #23/#52: объединённый вектор sp=7+sp=12
python scripts/issue32_seasonal_structure.py     # #32: смена has_seasonal между окнами

# 3. Бэктест по живым run-каталогам (прогон с --keep-raw):
python scripts/backtest_forecasters.py runs/run_ph1 [runs/run_ph2 ...]

# 4. Проверки
uv run ruff check .
uv run python -m pytest -q                        # быстрые тесты
uv run python -m pytest -q tests/test_reproducibility.py -m slow  # побайтовый пересчёт
```

Команды сверены с закрытыми прогонами: прецедент воспроизводимого запуска
#52 — `python scripts/issue23_combined_vector.py` (JSON: stdout +
`/tmp/issue23_combined_vector_results.json`); эксперимент #34 —
`python scripts/issue34_window_threshold.py` (JSON в stdout, таблица —
`docs/ISSUE_34_WINDOW_THRESHOLD.md`). Остальные скрипты — те же команды,
которыми записаны их отчёты в `docs/`.

## Где какой артефакт

| Скрипт | Артефакт | Формат |
|---|---|---|
| `experiment_vector_d.py` | stdout | JSON, слот `run` внутри |
| `issue34_window_threshold.py` | stdout | JSON, слот `run` внутри |
| `issue13_tfidf_vs_berta.py` | stdout | JSON, слот `run` внутри |
| `issue23_combined_vector.py` | stdout + `/tmp/issue23_combined_vector_results.json` | JSON, слот `run` внутри |
| `issue32_seasonal_structure.py` | `/tmp/issue32_seasonal_structure_results.json` | JSON, слот `run` внутри |
| `backtest_forecasters.py` | `outputs/backtest_mase*.csv` + `outputs/backtest_run_meta.json` | CSV + JSON-метаданные |

## Как проверить пересчёт самому

CI делает это на каждый пуш (`ci.yml`, шаг «Проверка пересчёта»):
`issue13_tfidf_vs_berta.py --tfidf` запускается дважды, stdout
сравнивается побайтово; при расхождении тест падает с указанием,
разошлись метаданные (`run`) или сами числа. Локально — та же команда
из блока выше (`-m slow`). Это самый дешёвый детерминированный прогон
репозитория; полный бэктест для этих целей не используется.

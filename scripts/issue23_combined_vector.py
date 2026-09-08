"""Issue #23: объединённый вектор (sp=7 + sp=12) как представление спроса.

Третья итерация MVP #23 поверх смёрженного фундамента:
- docs/EXPERIMENT_VECTOR_D.md — итерация 1 (sp=12, месячные фикстуры:
  рефит неисполним на 24 точках; сезонный вектор существует только при
  information_criterion="aic") и итерация 2 (sp=7, дневные фикстуры:
  форма профиля устойчива, corr >= 0.994, но параметризация mul/add плывёт
  между рефитами; у high_freq сезонной компоненты нет вовсе).
- docs/ISSUE_32_SEASONAL_STRUCTURE.md — has_seasonal есть исход решающего
  правила loglik-vs-штраф, а не свойство ряда.
- docs/ISSUE_34_WINDOW_THRESHOLD.md — порог окна для sp=7.

Вопрос этой итерации — не «устойчивы ли коэффициенты» (ответ известен по
частям), а «работает ли объединение двух блоков как ПРЕДСТАВЛЕНИЕ»: меняет
ли объединённый вектор отношения близости между фразами по сравнению с
одинарными sp, и сохраняется ли это отношение при рефитах.

Дизайн (зафиксирован до прогона на реальных данных):
1. Блоки вектора на фразу:
   - sp=7: дневной train-окон 56 дней (TRAIN_WINDOW_DAYS из
     experiment_vector_d), AutoETS(aic) — та же конвенция, что векторы
     раздела 1 обеих итераций EXPERIMENT_VECTOR_D;
   - sp=12: месячный ряд (24 точки), AutoETS(aic) — единственный критерий,
     при котором сезонный блок существует (итерация 1: при aicc оба бэкенда
     отвергают сезонность на всех трёх фикстурах).
2. Объединение: конкатенация двух блоков, каждый приведён к z-оценке
   (среднее 0, std 1). Обоснование: EXPERIMENT_VECTOR_D установлен факт,
   что корреляция — единственная метрика, валидная через смену
   параметризации mul/add (scale/shift-инвариантна); z-скоринг делает
   блоки безразмерными и равновесными (по 7 и по 12 координат — веса
   блоков в корреляции конкатенации не выравнивать нельзя, длинный блок
   иначе доминирует только через число координат... НЕ выравнивается
   намеренно: корреляция конкатенации честно учитывает, что годовой блок
   несёт больше координат; дополнительно репортится поэлементная
   корреляция каждого блока отдельно, так что эффект длины блока виден
   явно). Инвариант NaN-семантики: similarity с участием блока
   has_seasonal=False (или вырожденного константного) — NaN, ДЛЯ ВСЕХ
   представлений, включая combined: combined-сравнение пары возвращает
   NaN, если хотя бы один из ЧЕТЫРЁХ участвующих блоков не определён.
   Подмешивать нулевой блок-маркер в живую конкатенацию нельзя: она дала
   бы определённое число, в котором «блок-ничто» одной фразы коррелирует
   с нулевым блоком партнёра и разбавляет живые блоки — сравнение
   «combined против одинарных sp» было бы искажено. Цена инварианта
   честно видна в отчёте: пары с high_freq (нет недельного блока,
   итерация 2) в combined не определены так же, как в sp7.
3. Структура представления: три попарные корреляции между тремя фразами
   для каждого из трёх представлений {sp7, sp12, combined} + ближайший
   сосед каждой фразы в каждом представлении.
4. Устойчивость структуры представления: рефит только дневного блока
   (−7 и −14 дней, конвенция итерации 2), месячный блок фиксирован —
   его рефит неисполним (итерация 1), поэтому combined-устойчивость
   здесь = устойчивость sp=7-части представления, что репортится явно.
   Вердикт: сохранились ли упорядочение трёх попарных близостей и
   ближайшие соседи на обоих рефитах.
5. Диагностика месячного блока (бесплатно, один фит на ряд): сравнение
   циклов 1↔2 внутри одного фита (compare_seasonal_cycles_within_full_series)
   — та же оговорка, что в итерации 1: не заменяет внешний рефит.

Пункт 4 issue #23 (кластеризация и сравнение с эмбеддингами) на трёх
фразах статистически неисполним и НЕ выполняется — как в итерациях 1–2.

Запуск: python scripts/issue23_combined_vector.py
(полный JSON — /tmp/issue23_combined_vector_results.json, таблицы — stdout).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ограничить BLAS/OMP-потоки ДО импорта numpy/statsmodels — та же причина,
# что в experiment_vector_d.py (параллельные прогоны воркеров).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd

# Запуск файлом (python scripts/issue23_combined_vector.py) кладёт в sys.path
# каталог scripts/, а не корень репозитория — без этого бутстрапа импорт
# scripts.experiment_vector_d падает ModuleNotFoundError (issue #52).
# При импорте модуля из тестов (корень уже в sys.path) вставка безвредна:
# дубликат пути не меняет разрешение импорта.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.experiment_vector_d import (
    DAILY_FIXTURES,
    FIXTURES,
    TRAIN_WINDOW_DAYS,
    _fit_autoets,
    compare_seasonal_cycles_within_full_series,
    fit_autoets_vector,
    load_daily_dynamics_csv,
    load_dynamics_csv,
    truncate_series,
    vector_from_fitted,
)

PHRASES = [
    "seasonal (новогодние подарки)",
    "high_freq (купить телефон)",
    "mid_freq (курсы английского)",
]

DAILY_REFIT_DROPS = (7, 14)  # −1 и −2 недели, конвенция итерации 2


def zscore_block(seasonal: np.ndarray, has_seasonal: bool) -> np.ndarray:
    """Привести сезонный блок к z-оценке (среднее 0, std 1).

    Блок без сезонной компоненты (has_seasonal=False) — нулевой вектор по
    построению (не «измеренно плоский профиль»): у нулевого вектора нет ни
    среднего, ни std, z-скоринг к нему неприменим — возвращаем нули как
    маркер «блока нет», а similarity с его участием вычисляющий код обязан
    трактовать как NaN (см. pearson_or_nan).
    """
    if not has_seasonal:
        return np.zeros_like(seasonal, dtype=float)
    std = float(seasonal.std())
    if std == 0:
        # Вырожденный, но has_seasonal=True профиль (константа): после
        # центрирования — нули. Сходство с ним не определено так же, как
        # с отсутствующим блоком — NaN в pearson_or_nan.
        return np.zeros_like(seasonal, dtype=float)
    return (seasonal - seasonal.mean()) / std


@dataclass
class CombinedVector:
    """Объединённый вектор фразы: z-оскоренные блоки sp=7 и sp=12.

    block7/block12 хранятся до z-скоринга для отчёта (model_spec,
    has_seasonal); z7/z12 — блоки, вошедшие в combined.
    monthly_fitted — уже посчитанный _fit_autoets месячного ряда (не
    сериализуется в to_dict): переиспользуется диагностикой циклов, чтобы
    месячный ряд фитился один раз (issue #33, находка 3).
    """

    label: str
    spec7: str
    spec12: str
    has_seasonal7: bool
    has_seasonal12: bool
    z7: np.ndarray
    z12: np.ndarray
    monthly_fitted: object = None

    @property
    def combined(self) -> np.ndarray:
        return np.concatenate([self.z7, self.z12])

    @property
    def has_combined_signal(self) -> bool:
        """True, если хотя бы один блок несёт ненулевой сигнал.

        Если оба блока отсутствуют/константны, combined — нулевой вектор:
        similarity с его участием NaN, фраза в этом представлении не
        представлена вовсе (это данные, не ошибка кода).
        """
        return bool(np.any(self.z7) or np.any(self.z12))

    def to_dict(self) -> dict:
        return {
            "spec_sp7": self.spec7,
            "spec_sp12": self.spec12,
            "has_seasonal_sp7": self.has_seasonal7,
            "has_seasonal_sp12": self.has_seasonal12,
            "has_combined_signal": self.has_combined_signal,
        }


def pearson_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    """Корреляция Пирсона; NaN, если любой из векторов константный.

    np.corrcoef на константном векторе возвращает NaN и варнинг — здесь это
    легитимный исход («сравнивать нечего»), репортим его явно и без варнинга.
    """
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_combined_vector(
    label: str,
    daily_series: pd.Series,
    monthly_series: pd.Series,
    daily_drop: int = 0,
) -> CombinedVector:
    """Снять оба блока и собрать CombinedVector.

    daily_drop — укорочение дневного train-окна с конца (0 = полное окно
    56 дней). Месячный ряд не укорачивается никогда (рефит неисполним,
    итерация 1) — параметра нет принципиально, не забыт.

    Месячный ряд фитится здесь ОДИН раз (_fit_autoets), вектор извлекается
    vector_from_fitted, а fitted сохраняется в CombinedVector.monthly_fitted
    для диагностики циклов — без дублирующего рефита (issue #33, находка 3).
    """
    daily_window = truncate_series(daily_series, daily_drop) if daily_drop else daily_series

    vec7 = fit_autoets_vector(daily_window, sp=7, calendar_anchor="weekday", information_criterion="aic")
    monthly_fitted = _fit_autoets(monthly_series, sp=12, information_criterion="aic")
    vec12 = vector_from_fitted(monthly_fitted, monthly_series, sp=12, calendar_anchor="month")

    return CombinedVector(
        label=label,
        spec7=vec7.model_spec,
        spec12=vec12.model_spec,
        has_seasonal7=vec7.has_seasonal,
        has_seasonal12=vec12.has_seasonal,
        z7=zscore_block(vec7.seasonal, vec7.has_seasonal),
        z12=zscore_block(vec12.seasonal, vec12.has_seasonal),
        monthly_fitted=monthly_fitted,
    )


REPRESENTATIONS = ("sp7", "sp12", "combined")


def block_defined(vec: CombinedVector, which: str) -> bool:
    """Определён ли блок вектора для сравнения: заявленная сезонность есть
    И z-скоренный блок не вырожден в нулевой маркер.

    z-скоренный живой блок имеет std ровно 1; has_seasonal=False и
    константный сезонный профиль оба дают нулевой вектор — для similarity
    это один и тот же случай «блока нет» (см. zscore_block).
    """
    z = vec.z7 if which == "sp7" else vec.z12
    has = vec.has_seasonal7 if which == "sp7" else vec.has_seasonal12
    return bool(has) and float(z.std()) != 0.0


def similarity(va: CombinedVector, vb: CombinedVector, repr_name: str) -> float:
    """Попарная близость двух фраз в представлении repr_name.

    Единая точка вычисления sim для pairwise_structure и nearest_neighbors
    (две независимые реализации одной метрики разошлись бы молча).

    Инвариант NaN-семантики (единый для всех представлений): NaN, если
    хотя бы один УЧАСТУЮЩИЙ блок не определён:
    - sp7/sp12: не определён соответствующий блок хотя бы у одной фразы
      (pearson_or_nan на нулевом маркере возвращает NaN сам);
    - combined: не определён любой из ЧЕТЫРЁХ участвующих блоков.
      Корреляция полной конкатенации при живом втором блоке вернула бы
      ОПРЕДЕЛЁННОЕ число, в котором нулевой блок-маркер одной фразы
      коррелирует с нулевым блоком партнёра и разбавляет живые блоки, —
      такое сравнение объявлено недопустимым (см. дизайн, пункт 2).
    """
    if repr_name == "sp7":
        return pearson_or_nan(va.z7, vb.z7)
    if repr_name == "sp12":
        return pearson_or_nan(va.z12, vb.z12)
    if repr_name != "combined":
        raise ValueError(f"Неизвестное представление: {repr_name!r}")
    for vec in (va, vb):
        if not (block_defined(vec, "sp7") and block_defined(vec, "sp12")):
            return float("nan")
    return pearson_or_nan(va.combined, vb.combined)


def pairwise_structure(vectors: dict[str, CombinedVector]) -> dict:
    """Попарные корреляции фраз в трёх представлениях.

    sim(X,Y) для combined — корреляция конкатенации z-блоков (только при
    всех четырёх определённых блоках, см. similarity); для sp7/sp12 —
    корреляция соответствующего одиночного блока (то же значение, что и
    блок-корреляция combined, приведено отдельно, чтобы сравнение
    «объединённый против одинарных» было прямым).

    NaN в sim — легитимный исход (участвующий блок отсутствует/константный,
    см. similarity), не ошибка; весь следующий код относится к NaN
    как к «сравнение не определено», никогда как к 0 или 1.
    """
    structure: dict = {}
    for repr_name in REPRESENTATIONS:
        sims = {}
        keys = list(vectors)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                sim = similarity(vectors[a], vectors[b], repr_name)
                sims[f"{a} <-> {b}"] = round(sim, 6) if not math.isnan(sim) else None
        structure[repr_name] = sims
    return structure


def nearest_neighbors(vectors: dict[str, CombinedVector]) -> dict:
    """Ближайший сосед каждой фразы в каждом представлении.

    Перебирает ключи vectors (не глобальный PHRASES — функция обязана
    работать на любом наборе меток, в т.ч. синтетическом в тестах).
    NaN-sim не участвует в выборе (argmax по NaN — ошибка выбора); если
    ВСЕ sims фразы NaN, сосед — None с флагом undefined: фраза в этом
    представлении не представлена, а не «сама себе ближайшая».
    """
    result: dict = {}
    for repr_name in REPRESENTATIONS:
        per_phrase = {}
        for phrase in vectors:
            sims = {
                other: similarity(vectors[phrase], vectors[other], repr_name) for other in vectors if other != phrase
            }
            defined = {k: v for k, v in sims.items() if not math.isnan(v)}
            if defined:
                best = max(defined, key=defined.get)  # type: ignore[arg-type]
                per_phrase[phrase] = {"neighbor": best, "sim": round(defined[best], 6), "undefined": False}
            else:
                per_phrase[phrase] = {"neighbor": None, "sim": None, "undefined": True}
        result[repr_name] = per_phrase
    return result


def neighbor_preserved(before: dict, after: dict) -> dict:
    """Сохранился ли ближайший сосед фразы между двумя прогонами.

    undefined→undefined «сохранившимся» НЕ считается: у фразы нет соседа
    в обоих прогонах — сравнивать нечего, и засчитывать это в стабильность
    значило бы завышать флаг («соседи сохранились» ≠ «соседи не
    определены дважды»). preserved вычисляется только по парам, где оба
    соседа определены; число пар, оставшихся undefined в обоих прогонах,
    репортится отдельно.
    """
    both_defined = before["neighbor"] is not None and after["neighbor"] is not None
    both_undefined = before["undefined"] and after["undefined"]
    return {
        "preserved": bool(both_defined and before["neighbor"] == after["neighbor"]),
        "undefined_in_both": bool(both_undefined),
        "undefined_in_either": bool(before["undefined"] or after["undefined"]),
    }


def structure_fingerprint(structure: dict[str, float | None]) -> list[str]:
    """Упорядочение пар по убыванию близости — «отпечаток» структуры
    представления. Пары с sim=None (NaN) ставятся в конец в исходном
    порядке; их позиция в отпечатке нестабильна по построению, поэтому
    сравнение отпечатков между представлениями/рефитами честно учитывает
    только определённые пары — сравнивает отпечаток целиком, но вызывающий
    код репортит и число определённых пар (см. main()).
    """
    defined = [(k, v) for k, v in structure.items() if v is not None]
    undefined = [k for k, v in structure.items() if v is None]
    ordered = [k for k, _ in sorted(defined, key=lambda kv: kv[1], reverse=True)]
    return ordered + undefined


def main() -> int:
    report: dict = {"phrases": {}, "representations": {}, "refits": {}}

    # --- Блоки и векторы на полном train-окне ---
    daily_series: dict[str, pd.Series] = {}
    monthly_series: dict[str, pd.Series] = {}
    full_vectors: dict[str, CombinedVector] = {}

    for label in PHRASES:
        d = load_daily_dynamics_csv(DAILY_FIXTURES[label])
        if len(d) < TRAIN_WINDOW_DAYS:
            raise ValueError(f"Дневной ряд '{label}' короче train-окна: {len(d)} < {TRAIN_WINDOW_DAYS}")
        daily_series[label] = truncate_series(d, len(d) - TRAIN_WINDOW_DAYS)
        monthly_series[label] = load_dynamics_csv(FIXTURES[label])
        vec = build_combined_vector(label, daily_series[label], monthly_series[label])
        full_vectors[label] = vec
        entry = vec.to_dict()
        # Диагностика месячного блока — тот же одноразовый фит-статистика,
        # что в итерации 1 (сравнение циклов внутри одного фита), здесь
        # включена в отчёт, потому что месячный блок теперь ЧАСТЬ вектора.
        # Фит НЕ дублируется: передаётся уже посчитанный
        # vec.monthly_fitted (issue #33, находка 3).
        entry["monthly_cycle_comparison"] = compare_seasonal_cycles_within_full_series(
            monthly_series[label], fitted_forecaster=vec.monthly_fitted
        )
        report["phrases"][label] = entry

    # --- Структура представления: одинарные sp против объединённого ---
    structure = pairwise_structure(full_vectors)
    neighbors = nearest_neighbors(full_vectors)
    fingerprints = {name: structure_fingerprint(structure[name]) for name in REPRESENTATIONS}
    report["representations"] = {
        "pairwise_similarity": structure,
        "nearest_neighbor": neighbors,
        "fingerprint_order_most_to_least_similar": fingerprints,
    }

    # --- Устойчивость структуры представления при рефитах дневного блока ---
    # Месячный блок фиксирован: его рефит неисполним на 24 точках
    # (docs/EXPERIMENT_VECTOR_D.md, итерация 1), поэтому это проверка
    # устойчивости sp=7-части представления; месячная часть гипотезы
    # остаётся непроверяемой на имеющихся данных — репортится явно.
    refit_results: dict = {}
    for drop in DAILY_REFIT_DROPS:
        refit_vectors = {
            label: build_combined_vector(label, daily_series[label], monthly_series[label], daily_drop=drop)
            for label in PHRASES
        }
        refit_structure = pairwise_structure(refit_vectors)
        refit_neighbors = nearest_neighbors(refit_vectors)
        refit_fingerprints = {name: structure_fingerprint(refit_structure[name]) for name in REPRESENTATIONS}
        refit_results[f"daily_minus_{drop}"] = {
            "blocks": {label: refit_vectors[label].to_dict() for label in PHRASES},
            "pairwise_similarity": refit_structure,
            "fingerprint_preserved": {name: refit_fingerprints[name] == fingerprints[name] for name in REPRESENTATIONS},
            # preserved считается только по парам с ОБОИМИ определёнными
            # соседями; undefined→undefined репортится отдельно (см.
            # neighbor_preserved) и в «сохранившиеся» не засчитывается.
            "nearest_neighbors": {
                name: {p: neighbor_preserved(neighbors[name][p], refit_neighbors[name][p]) for p in PHRASES}
                for name in REPRESENTATIONS
            },
        }
    report["refits"] = refit_results

    # Сводный вердикт устойчивости структуры combined: отпечаток
    # сохранился; соседи — stable только там, где ОПРЕДЕЛЕНЫ в обоих
    # прогонах. undefined→undefined ни в «стабильные», ни в «нестабильные»
    # не попадает: сравнивать нечего — репортится отдельным счётчиком,
    # чтобы ни завышать, ни занижать флаг (ревью PR #47).
    combined_neighbor_details = [r["nearest_neighbors"]["combined"] for r in refit_results.values()]
    combined_fingerprint_stable = all(r["fingerprint_preserved"]["combined"] for r in refit_results.values())
    defined_pairs = [
        (detail, p) for detail in combined_neighbor_details for p in PHRASES if not detail[p]["undefined_in_either"]
    ]
    combined_undefined_in_both = {
        p: all(detail[p]["undefined_in_both"] for detail in combined_neighbor_details) for p in PHRASES
    }
    report["representation_stability_combined"] = {
        "fingerprint_stable_on_all_refits": combined_fingerprint_stable,
        "defined_neighbors_all_preserved_on_all_refits": all(detail[p]["preserved"] for detail, p in defined_pairs),
        "n_defined_neighbor_pairs_compared": len(defined_pairs),
        "neighbors_undefined_in_both_refits": combined_undefined_in_both,
        "caveat": (
            "проверена только sp=7-часть представления (дневные рефиты); "
            "месячный блок рефит неисполним на 24 точках — устойчивость "
            "combined-вектора в целом НЕ подтверждается этим прогоном. "
            "undefined→undefined пары не входят ни в stable, ни в счёт "
            "сравненных (см. neighbors_undefined_in_both_refits и "
            "n_defined_neighbor_pairs_compared)."
        ),
    }

    out_path = Path("/tmp/issue23_combined_vector_results.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nПолный отчёт: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

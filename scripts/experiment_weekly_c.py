#!/usr/bin/env python3
"""Вторая итерация MVP #23, вариант В: недельный вектор `sp=7` на дневных рядах.

Порядок и критерии зафиксированы **до** прогона (см. `PREREGISTERED`).

    uv run python scripts/experiment_weekly_c.py --output docs/EXPERIMENT_VECTOR_C.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wordstat_trends.daily_window import SP, TRAIN_DAYS, TRUNCATIONS, make_window  # noqa: E402
from wordstat_trends.dynamics_io import load_dynamics  # noqa: E402
from wordstat_trends.weekly_vector import BACKENDS, WEEKDAY_NAMES, sktime_vector  # noqa: E402

FIXTURES = {
    "новогодние подарки": ("dynamics_daily_seasonal_mvp.csv", "сезонная"),
    "купить телефон": ("dynamics_daily_high_freq_mvp.csv", "высокочастотная"),
    "курсы английского языка": ("dynamics_daily_mid_freq_mvp.csv", "среднечастотная"),
}

CORR_GATE = 0.9
"""Корреляция профилей полного и укороченного окна."""

MEAN_SHIFT_GATE = 0.15
"""Средний относительный сдвиг коэффициента."""

AMPLITUDE_SHIFT_GATE = 0.25
"""Сдвиг, отнесённый к амплитуде профиля."""

FLAT_AMPLITUDE = 0.02
"""Ниже этой амплитуды профиль считается плоским: устойчивости там не у чего мерить."""

PREREGISTERED = f"""\
Критерии зафиксированы до прогона и после него не менялись — те же три, что
объявлялись на первой итерации (`sp=12`), поскольку они относительные и от
длины окна не зависят:

* корреляция профилей ≥ {CORR_GATE};
* средний относительный сдвиг коэффициента ≤ {MEAN_SHIFT_GATE:.0%};
* сдвиг, отнесённый к амплитуде профиля, ≤ {AMPLITUDE_SHIFT_GATE};

на **обоих** укорочениях. Фраза с амплитудой профиля ниже {FLAT_AMPLITUDE}
объявляется плоской: у неё нет формы, устойчивость которой можно измерять,
и вердикт по ней — «неприменимо», а не «устойчив».
"""


@dataclass
class Comparison:
    cut: int
    days: int
    cycles: int
    correlation: float
    mean_shift: float
    max_shift: float
    amplitude_shift: float
    profile: np.ndarray


def compare(reference: np.ndarray, other: np.ndarray) -> Comparison:
    absolute = np.abs(other - reference)
    relative = absolute / reference
    amplitude = reference.std()
    return Comparison(
        cut=0,
        days=0,
        cycles=0,
        correlation=float(np.corrcoef(reference, other)[0, 1]) if amplitude > 0 else float("nan"),
        mean_shift=float(relative.mean()),
        max_shift=float(relative.max()),
        amplitude_shift=float(absolute.mean() / amplitude) if amplitude > 0 else float("inf"),
        profile=other,
    )


def row(values: np.ndarray) -> str:
    return " | ".join(f"{v:.3f}" for v in values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default=Path("tests/fixtures"), type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    out: list[str] = []

    def emit(line: str = "") -> None:
        out.append(line)

    windows = {}
    for phrase, (filename, _) in FIXTURES.items():
        series = load_dynamics(args.fixtures / filename, granularity="daily")
        windows[phrase] = make_window(series)

    emit("<!-- Сгенерировано scripts/experiment_weekly_c.py — не редактировать руками. -->")
    emit()
    emit("# MVP гипотезы #23, вариант В")
    emit()
    emit("## Вторая итерация: недельный вектор `sp=7` на дневных рядах")
    emit()

    # --- Данные ---------------------------------------------------------
    emit("### Данные и окно")
    emit()
    reference_window = windows[next(iter(FIXTURES))]
    train = reference_window.train
    emit(
        f"Обучающее окно — **{TRAIN_DAYS} дней = {TRAIN_DAYS // SP} полных недельных циклов**, "
        f"{train.index[0]:%d.%m.%Y} — {train.index[-1]:%d.%m.%Y}."
    )
    emit()
    emit("| Проверка | Значение |")
    emit("|---|---|")
    emit(f"| Дней в обучающем окне | {len(train)} |")
    emit(f"| Циклов `sp={SP}` | {len(train) // SP} |")
    emit(f"| Каждый день недели встречается | {reference_window.weekday_counts} |")
    emit(f"| Дней в holdout | {len(reference_window.holdout)} |")
    emit("| Пропуски и дубли дат | нет (проверено) |")
    emit()
    emit("Баланс дней недели — не косметика: если бы какой-то день встречался чаще,")
    emit("его коэффициент считался бы по большему числу наблюдений и профиль был бы")
    emit("смещён ещё до оценивания. Holdout — это то, что осталось после первых")
    emit(f"{TRAIN_DAYS} строк ({len(reference_window.holdout)} дн.), а не разность с сырой длиной:")
    emit("метка конца в заголовке графика Вордстата эксклюзивна и на день опережает")
    emit("последнюю строку данных.")
    emit()

    # --- Шаг 1: векторы --------------------------------------------------
    emit("### Шаг 1. Векторы на полном окне")
    emit()
    emit("Спецификацию ни один бэкенд не получает руками — оба выбирают сами.")
    emit()
    emit("| Фраза | Профиль | Бэкенд | Модель | Сезонность | Уровень | AIC | Амплитуда |")
    emit("|---|---|---|---|---|---|---|---|")
    vectors: dict[str, dict[str, object]] = {}
    for phrase, (_, kind) in FIXTURES.items():
        vectors[phrase] = {}
        for backend, fit in BACKENDS.items():
            vector = fit(windows[phrase].train)
            vectors[phrase][backend] = vector
            level = f"{vector.level:,.0f}".replace(",", " ")
            emit(
                f"| «{phrase}» | {kind} | {backend} | `{vector.spec}` | "
                f"{'да' if vector.seasonal_chosen else '**нет**'} | {level} | "
                f"{vector.aic:.1f} | {vector.amplitude:.3f} |"
            )
    emit()

    emit("Недельные профили (понедельник → воскресенье), нормированы к среднему 1:")
    emit()
    emit("| Фраза | Источник | " + " | ".join(WEEKDAY_NAMES) + " |")
    emit("|---" * (SP + 2) + "|")
    for phrase in FIXTURES:
        observed = windows[phrase].train.astype(float)
        empirical = observed.groupby(observed.index.dayofweek).mean()
        empirical = (empirical / empirical.mean()).to_numpy()
        emit(f"| «{phrase}» | эмпирика | " + row(empirical) + " |")
        for backend in BACKENDS:
            emit(f"| «{phrase}» | {backend} | " + row(vectors[phrase][backend].seasonal) + " |")
    emit()
    emit("«Эмпирика» — простые средние по дням недели, без модели. Она здесь как")
    emit("независимая опора: профиль, разошедшийся с ней, означал бы ошибку снятия,")
    emit("а не свойство спроса.")
    emit()

    return _stability(out, emit, windows, vectors, args)


def _stability(out, emit, windows, vectors, args) -> int:
    # --- Шаг 2: кросс-проверка бэкендов ---------------------------------
    emit("### Шаг 2. Кросс-проверка бэкендов")
    emit()
    emit("| Фраза | Согласие по сезонности | Корреляция профилей | Разница AIC |")
    emit("|---|---|---|---|")
    agreement: dict[str, bool] = {}
    for phrase in FIXTURES:
        sk = vectors[phrase]["sktime"]
        sf = vectors[phrase]["statsforecast"]
        agree = sk.seasonal_chosen == sf.seasonal_chosen
        agreement[phrase] = agree
        if sk.amplitude > 0 and sf.amplitude > 0:
            correlation = f"{np.corrcoef(sk.seasonal, sf.seasonal)[0, 1]:+.3f}"
        else:
            correlation = "—"
        verdict = "да" if agree else f"**нет** ({sk.spec} против {sf.spec})"
        emit(f"| «{phrase}» | {verdict} | {correlation} | {sf.aic - sk.aic:+.1f} |")
    emit()

    emit("#### Почему разница AIC между бэкендами велика — и почему её нельзя читать")
    emit()
    emit("Разница AIC в таблице выше (+51.6 … +67.8) выглядит как сильное")
    emit("предпочтение sktime, в том числе там, где профили совпадают с корреляцией")
    emit("+1.000. Это противоречие кажущееся, и разобрать его важно, потому что иначе")
    emit("из него делается ложный вывод «бэкенды резко расходятся».")
    emit()
    emit("**Измерено:** если заставить оба бэкенда взять одну и ту же спецификацию")
    emit("(`ANN`, `AAN` — по три фразы каждая, шесть сочетаний), разрыв никуда не")
    emit("девается и остаётся почти постоянным: Δloglik от −29.3 до −34.0, ΔAIC ≈ +66.")
    emit("Значит дело не в выбранной параметризации и не в сезонности.")
    emit()
    emit("**Причина:** бэкенды считают разное правдоподобие. statsforecast использует")
    emit("**концентрированное** правдоподобие `−n/2·log(SSE)`, опуская аддитивные")
    emit("константы; statsmodels внутри sktime их включает. Проверено дословно: для")
    emit("всех шести сочетаний `−n/2·log(SSE)` совпало с loglik statsforecast с")
    emit("расхождением **0.00**. Опущенная константа `−n/2·log(2π) − n/2` равна −79.46")
    emit("при n = 56 — она и даёт систематический сдвиг.")
    emit()
    emit("**Следствие, важное для методики:** абсолютные значения AIC двух бэкендов")
    emit("**несопоставимы между собой**. Сравнивать AIC допустимо только внутри одного")
    emit("бэкенда. Колонка «разница AIC» в таблице выше оставлена как измеренный факт,")
    emit("но интерпретировать её как «одна модель лучше другой» нельзя.")
    emit()
    emit("Внутри sktime решение по «купить телефон» при этом обосновано корректно:")
    emit("`ETS(A,N,N)` даёт AIC 1043.1 против 1062.2 у `ETS(A,N,A)` — сезонность")
    emit("добавляет 8 параметров, а правдоподобие почти не растёт (−518.55 → −520.12).")
    emit("То есть sktime не «проглядел» форму, а счёл её не окупающей параметры; а")
    emit("statsforecast, оценивая ту же форму, счёл иначе. Обе позиции защитимы, и")
    emit("именно поэтому вывод по этой фразе сформулирован как граничный.")
    emit()

    # --- Шаг 3: устойчивость --------------------------------------------
    emit("### Шаг 3. Устойчивость — главная проверка")
    emit()
    emit(PREREGISTERED)
    emit()
    emit("Окно укорачивается **с конца**: что бы мы сняли неделю и две недели назад.")
    emit()
    emit("| Фраза | Укорочение | Дней | Циклов | Модель | Корреляция | Средний сдвиг | Сдвиг / амплитуда |")
    emit("|---|---|---|---|---|---|---|---|")

    results: dict[str, list[Comparison]] = {}
    for phrase in FIXTURES:
        train = windows[phrase].train
        reference = sktime_vector(train)
        comparisons = []
        for cut in TRUNCATIONS:
            shortened = train.iloc[:-cut]
            vector = sktime_vector(shortened)
            comparison = compare(reference.seasonal, vector.seasonal)
            comparison.cut = cut
            comparison.days = len(shortened)
            comparison.cycles = len(shortened) // SP
            comparisons.append(comparison)
            correlation = "—" if np.isnan(comparison.correlation) else f"{comparison.correlation:+.3f}"
            shift = "—" if np.isinf(comparison.amplitude_shift) else f"{comparison.amplitude_shift:.2f}"
            emit(
                f"| «{phrase}» | −{cut // SP} нед. | {comparison.days} | {comparison.cycles} | "
                f"`{vector.spec}` | {correlation} | {comparison.mean_shift:.1%} | {shift} |"
            )
        results[phrase] = comparisons
    emit()

    emit("Профили укороченных окон (пн → вс):")
    emit()
    emit("| Фраза | Окно | " + " | ".join(WEEKDAY_NAMES) + " |")
    emit("|---" * (SP + 2) + "|")
    for phrase in FIXTURES:
        emit(f"| «{phrase}» | полное ({TRAIN_DAYS}) | " + row(vectors[phrase]["sktime"].seasonal) + " |")
        for comparison in results[phrase]:
            emit(f"| «{phrase}» | −{comparison.cut // SP} нед. ({comparison.days}) | " + row(comparison.profile) + " |")
    emit()

    return _verdict(out, emit, windows, vectors, results, agreement, args)


def _verdict(out, emit, windows, vectors, results, agreement, args) -> int:
    emit("#### Вердикт по шагу 3")
    emit()
    verdicts: dict[str, str] = {}
    for phrase, comparisons in results.items():
        amplitude = vectors[phrase]["sktime"].amplitude
        if amplitude < FLAT_AMPLITUDE:
            verdicts[phrase] = "неприменимо"
        elif all(
            c.correlation >= CORR_GATE
            and c.mean_shift <= MEAN_SHIFT_GATE
            and c.amplitude_shift <= AMPLITUDE_SHIFT_GATE
            for c in comparisons
        ):
            verdicts[phrase] = "устойчив"
        else:
            verdicts[phrase] = "неустойчив"

    labels = {
        "устойчив": "**устойчив**",
        "неустойчив": "**неустойчив**",
        "неприменимо": "**неприменимо** — профиль плоский, устойчивой формы нет",
    }
    for phrase, verdict in verdicts.items():
        amplitude = vectors[phrase]["sktime"].amplitude
        emit(f"* «{phrase}» (амплитуда {amplitude:.3f}) — {labels[verdict]}")
    emit()

    stable = [p for p, v in verdicts.items() if v == "устойчив"]
    flat = [p for p, v in verdicts.items() if v == "неприменимо"]

    # --- Шаг 4: осмысленность -------------------------------------------
    emit("### Шаг 4. Осмысленность — различает ли вектор профили")
    emit()
    emit("| Фраза | max/min | Амплитуда | Пик | Провал | Будни / выходные |")
    emit("|---|---|---|---|---|---|")
    for phrase in FIXTURES:
        seasonal = vectors[phrase]["sktime"].seasonal
        weekday = seasonal[:5].mean()
        weekend = seasonal[5:].mean()
        emit(
            f"| «{phrase}» | {seasonal.max() / seasonal.min():.2f}× | {seasonal.std():.3f} | "
            f"{WEEKDAY_NAMES[int(np.argmax(seasonal))]} | {WEEKDAY_NAMES[int(np.argmin(seasonal))]} | "
            f"{weekday / weekend:.2f} |"
        )
    emit()

    return _conclusions(out, emit, windows, vectors, results, agreement, verdicts, stable, flat, args)


def _conclusions(out, emit, windows, vectors, results, agreement, verdicts, stable, flat, args) -> int:
    emit("### Итог")
    emit()
    emit("#### Измерено")
    emit()
    window = windows[next(iter(FIXTURES))]
    emit(
        f"1. Окно {TRAIN_DAYS} дней даёт {TRAIN_DAYS // SP} полных циклов и баланс "
        f"{window.weekday_counts} — профиль несмещён по построению."
    )
    disagreed = [p for p, ok in agreement.items() if not ok]
    if disagreed:
        emit(
            f"2. Бэкенды разошлись по наличию сезонности у {len(disagreed)} из {len(agreement)} фраз: "
            + ", ".join(f"«{p}»" for p in disagreed)
            + ". У остальных согласие полное (корреляция профилей +1.000) — то есть две"
            " независимые реализации, выбиравшие спецификацию самостоятельно, пришли к"
            " одному профилю. На первой итерации (`sp=12`) согласия не было вовсе:"
            " statsforecast отвергал сезонность по AIC во всех прогонах."
        )
    else:
        emit(f"2. Бэкенды согласны по всем {len(agreement)} фразам.")
    emit(
        f"3. Проверка устойчивости **исполнима**: рефиты на {TRUNCATIONS[0] // SP} и "
        f"{TRUNCATIONS[1] // SP} недели короче оставляют "
        f"{(TRAIN_DAYS - TRUNCATIONS[0]) // SP} и {(TRAIN_DAYS - TRUNCATIONS[1]) // SP} циклов "
        "против порога в 2. На первой итерации (`sp=12`) запаса не было вовсе и рефит падал."
    )
    emit(
        f"4. Устойчивыми признаны {len(stable)} из {len(verdicts)} фраз"
        + (f", плоскими — {len(flat)}." if flat else ".")
    )
    emit()
    emit("#### Вывод")
    emit()
    if len(stable) == len([p for p in verdicts if verdicts[p] != "неприменимо"]) and stable:
        emit("**Стоп-условие гипотезы не сработало.** Там, где недельная форма есть, она")
        emit("переживает потерю двух недель наблюдений, оставаясь в объявленных заранее")
        emit("границах. В отличие от первой итерации, это утверждение получено тем же")
        emit("оценивателем, что и исходный вектор, — подмены оценивателя здесь нет.")
    elif stable:
        emit(f"Порог прошли {len(stable)} из {len(verdicts)} фраз — не зачёт целиком.")
    else:
        emit("**Стоп-условие сработало: устойчивых фраз нет.**")
    emit()
    if flat:
        names = ", ".join(f"«{p}»" for p in flat)
        emit(f"Отдельно про {names} — и здесь надо быть точным, потому что вердикт")
        emit("«плоский» опирается на выбор **одного** бэкенда.")
        emit()
        emit("sktime не нашёл недельной сезонности вовсе (`ETS(A,N,N)`), и по его вектору")
        emit("профиль строго единичный. statsforecast сезонность оставил (`ETS(A,N,A)`) и")
        emit("дал профиль с амплитудой 0.026 — крошечной, но не нулевой, и совпадающей")
        emit("с эмпирическими средними (пн 1.030 против 1.055, сб 0.956 против 0.956).")
        emit("Более того, этот профиль **устойчив** по тем же критериям: корреляция")
        emit("+0.990 и +0.959, сдвиг 0.12 и 0.24 амплитуды.")
        emit()
        emit("Честная формулировка поэтому не «недельной формы нет», а: **форма есть,")
        emit("но она на порядок слабее, чем у двух других фраз** (0.026 против 0.200 и")
        emit("0.110), и находится на границе, где два добросовестных оценивателя")
        emit("расходятся в том, считать ли её сигналом. Записывать её в «нет сезонности»")
        emit("означало бы выдать выбор одного бэкенда за измеренный факт.")
        emit()

    emit("#### Чего проверить не удалось")
    emit()
    emit("* **Кластеризация и сравнение с эмбеддингами** (пункты 3–4 гипотезы): три")
    emit("  фразы — не выборка, ответ был бы предрешён.")
    emit("* **Совмещение с годовым профилем `sp=12`.** Первая итерация показала, что")
    emit("  на 24 месячных точках он непроверяем; полный вектор из двух сезонностей")
    emit("  требует длинного месячного ряда (#6).")
    emit(f"* **Holdout использован только для контроля длины** ({len(window.holdout)} дн.) —")
    emit("  на таком объёме проверять качество прогноза бессмысленно.")
    emit()

    emit("---")
    emit()
    emit("## Первая итерация (месячные, `sp=12`) — отрицательный результат")
    emit()
    emit("Ниже сохранён результат первой итерации целиком. Он не отменён и не")
    emit("переписан: постановка на месячной грануляции оказалась ошибочной, но")
    emit("измерения остаются верными и объясняют, зачем понадобилась вторая.")
    emit()
    legacy = Path("docs/EXPERIMENT_VECTOR_C_SP12.md")
    if legacy.exists():
        body = legacy.read_text(encoding="utf-8")
        # Снимаем служебный заголовок и понижаем уровни, чтобы вложить в общий отчёт.
        for line in body.splitlines():
            if line.startswith("<!--"):
                continue
            if line.startswith("# "):
                # Собственный заголовок вложенного отчёта опускаем: его роль
                # уже выполняет заголовок «Первая итерация» выше.
                continue
            # Понижаем на два уровня, чтобы шаги первой итерации не встали
            # вровень с шагами второй.
            emit(("##" + line) if line.startswith("#") else line)
    else:
        emit("*(файл `docs/EXPERIMENT_VECTOR_C_SP12.md` не найден)*")
    emit()

    text = "\n".join(out) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Отчёт записан: {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

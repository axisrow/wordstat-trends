"""Тесты для scripts/experiment_vector_d.py (MVP #23, вариант "Г")."""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_vector_d import (
    DAILY_FIXTURES,
    DAILY_STABILITY_THRESHOLDS,
    FIXTURES,
    MONTHLY_STABILITY_THRESHOLDS,
    TRAIN_WINDOW_DAYS,
    ParamVector,
    _fit_autoets,
    _parse_daily_period,
    check_statsforecast_model_choice,
    check_weekday_balance,
    compare_seasonal_cycles_within_full_series,
    compare_vectors,
    empirical_weekday_profile,
    fit_autoets_vector,
    fit_statsforecast_seasonal_profile,
    is_comparison_unstable,
    load_daily_dynamics_csv,
    load_dynamics_csv,
    main,
    run_stability_check,
    truncate_series,
)


@pytest.mark.parametrize("path", FIXTURES.values(), ids=FIXTURES.keys())
def test_load_dynamics_csv_parses_all_fixtures(path: Path):
    series = load_dynamics_csv(path)

    assert len(series) == 24
    # Ряд начинается с августа 2024 и идёт помесячно без пропусков (docs/DATA.md).
    assert str(series.index[0]) == "2024-08"
    assert str(series.index[-1]) == "2026-07"
    assert series.index.is_monotonic_increasing
    assert (series > 0).all()


def test_load_dynamics_csv_rejects_duplicate_and_missing_month(tmp_path: Path):
    # Регрессия (Codex, цикл 3): парсер сортировал периоды, но не проверял
    # уникальность/непрерывность — CSV с пропущенным месяцем и дубликатом
    # другого месяца тихо давал 4 "валидные" строки на неверной временной
    # оси (октябрь пропущен, сентябрь задублирован), которую AutoETS затем
    # интерпретирует как регулярный ряд по позиции, а не по календарю.
    content = (
        "﻿Период;Число запросов;Доля от всех запросов, %;заголовок\r"
        "август 2024;1000;0,01;\r"
        "сентябрь 2024;1001;0,01;\r"
        "сентябрь 2024;1002;0,01;\r"
        "ноябрь 2024;1003;0,01;\r"
    )
    path = tmp_path / "broken.csv"
    path.write_bytes(content.encode("utf-8"))

    with pytest.raises(ValueError, match="[Дд]убликат"):
        load_dynamics_csv(path)


def test_load_dynamics_csv_seasonal_peak_in_december():
    # «Новогодние подарки»: пик спроса — декабрь 2024, известен из docs/DATA.md.
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    assert series.loc["2024-12"] == 1234632
    assert series.loc["2024-12"] == series.max()


def test_truncate_series_drops_from_the_end():
    series = load_dynamics_csv(FIXTURES["high_freq (купить телефон)"])
    truncated = truncate_series(series, 3)
    assert len(truncated) == 21
    assert truncated.index[-1] == series.index[-4]


# До issue #37 тест был в xfail(strict=False) на linux: AIC-выбор AutoETS на
# 24 точках зависел от BLAS — на OpenBLAS выигрывал mul/add/mul/damped с
# вырожденным сезонным профилем (11 из 12 mul-множителей ≤ 0). Починка —
# _seasonal_state_is_valid в _fit_autoets: фиты с невалидным сезонным
# состоянием выбывают из AIC-перебора на любой платформе.
def test_fit_autoets_vector_on_full_series_returns_calendar_aligned_seasonal():
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    vector = fit_autoets_vector(series, sp=12)

    assert len(vector.seasonal) == 12
    # Декабрьский сезонный коэффициент (индекс 11) — максимум профиля.
    assert int(np.argmax(vector.seasonal)) == 11


def test_seasonal_state_is_valid_rejects_degenerate_mul_profile():
    """Регрессия issue #37: на linux/OpenBLAS SLSQP-«сошедшийся» фит
    mul/add/mul/damped имел 11 из 12 сезонных множителей ≈ -1e7 при строго
    положительном ряде. Такой профиль обязан признаваться невалидным — иначе
    он выигрывает AIC-перебор и вырожденный вектор молча зависит от BLAS.
    """
    from scripts.experiment_vector_d import _seasonal_state_is_valid

    class _FakeFitted:
        def __init__(self, seasonal_values):
            self.states = pd.DataFrame({"seasonal": seasonal_values})

    # Профиль из упавшего CI-прогона (run 33941883186): мусорные величины,
    # одна из них ~1.0, остальные отрицательные/гигантские.
    ci_profile = np.array(
        [1.83211567e06, -3.43620438e06, -3.81864295e05, -1.00109400e06,
         -2.14414754e06, -6.10979393e06, -2.17114232e04, 9.99841103e-01,
         4.33479165e05, -3.88639645e07, -7.09792721e06, -1.02949987e07]
    )
    assert _seasonal_state_is_valid(_FakeFitted(ci_profile), "mul") is False
    assert _seasonal_state_is_valid(_FakeFitted(np.array([1.1, 0.9, 1.0])), "mul") is True
    assert _seasonal_state_is_valid(_FakeFitted(np.array([1.1, np.nan, 1.0])), "mul") is False
    # add-сезонность может быть отрицательной — валидна, пока конечна.
    assert _seasonal_state_is_valid(_FakeFitted(np.array([-5.0, 3.0])), "add") is True
    assert _seasonal_state_is_valid(_FakeFitted(np.array([np.inf])), "add") is False
    # Модель без сезонности — валидна по определению.
    assert _seasonal_state_is_valid(None, None) is True


def test_fit_autoets_vector_fails_on_series_shorter_than_two_seasonal_cycles():
    """Главный результат MVP: AutoETS(sp=12) требует >= 2 полных цикла (24
    точки). Ряд из 21 точки (24 - 3) физически не фитится — это не дрейф
    коэффициентов, а отказ на этапе инициализации сезонности.
    """
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    truncated = truncate_series(series, 3)

    with pytest.raises(ValueError, match="two full seasonal cycles"):
        fit_autoets_vector(truncated, sp=12)


def test_within_series_cycle_comparison_returns_high_correlation_on_full_series():
    """Побочное измерение (раздел 2b отчёта): цикл 1 и цикл 2 внутри одного
    фита на полном ряде должны быть сильно скоррелированы — это не проверка
    внешней устойчивости (см. caveat в результате), а факт про один фит.
    """
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    result = compare_seasonal_cycles_within_full_series(series, sp=12)

    assert "error" not in result
    assert len(result["cycle_1"]) == 12
    assert len(result["cycle_2"]) == 12
    assert result["corr"] > 0.9
    assert "caveat" in result


def test_within_series_cycle_comparison_reports_error_on_short_series():
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    truncated = truncate_series(series, 3)  # 21 точка, < 2*sp

    result = compare_seasonal_cycles_within_full_series(truncated, sp=12)
    assert "error" in result


def test_statsforecast_cross_check_reports_model_structure():
    """Раздел 3 отчёта: statsforecast должен фититься без ошибок (в отличие
    от основного бэкенда) и вернуть структуру выбранной модели.
    """
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    result = check_statsforecast_model_choice(series, sp=12)

    assert result["method"].startswith("ETS(")
    assert len(result["components"]) == 4
    assert isinstance(result["has_seasonal"], bool)


# --- Вторая итерация (дневные данные, sp=7): парсер и проверка баланса дней недели.
# Дневных фикстур ещё нет в репозитории (собираются отдельно) — эти тесты покрывают
# только код на контролируемых мини-примерах, не подменяют реальный эксперимент.


def test_parse_daily_period_reads_dotted_date():
    # Формат дневной выгрузки — "22.06.2026" (день.месяц.год с точками),
    # НЕ русское название месяца, как в месячных фикстурах.
    period = _parse_daily_period("22.06.2026")
    assert period == pd.Period("2026-06-22", freq="D")


def test_parse_daily_period_rejects_ru_month_format():
    # Месячный парсер и дневной несовместимы — проверяем, что дневной именно
    # падает на "август 2024", а не молча даёт неверный результат.
    with pytest.raises(ValueError):
        _parse_daily_period("август 2024")


def test_check_weekday_balance_detects_balanced_window():
    # 56-дневное окно, кратное семи, начинающееся с понедельника — каждый
    # день недели встречается ровно 8 раз.
    index = pd.period_range(start="2026-06-22", periods=56, freq="D")  # понедельник
    series = pd.Series(range(56), index=index)

    result = check_weekday_balance(series)

    assert result["balanced"] is True
    assert set(result["counts_by_weekday"].values()) == {8}
    assert sum(result["counts_by_weekday"].values()) == 56


def test_check_weekday_balance_detects_unbalanced_window():
    # Окно длиной не кратной семи (49 + 3 дня) — баланс должен быть нарушен.
    index = pd.period_range(start="2026-06-22", periods=52, freq="D")
    series = pd.Series(range(52), index=index)

    result = check_weekday_balance(series)

    assert result["balanced"] is False


@pytest.mark.parametrize("path", DAILY_FIXTURES.values(), ids=DAILY_FIXTURES.keys())
def test_load_daily_dynamics_csv_parses_all_fixtures(path: Path):
    # Собранные фикстуры: 58 строк, 23.06.2026 — 19.08.2026 (docs/EXPERIMENT_VECTOR_D.md,
    # вторая итерация). Заголовок графика в CSV называет конечную дату 20.08 —
    # эксклюзивную границу, не последнюю строку данных.
    series = load_daily_dynamics_csv(path)

    assert len(series) == 58
    assert str(series.index[0]) == "2026-06-23"
    assert str(series.index[-1]) == "2026-08-19"
    assert series.index.is_monotonic_increasing
    assert series.index.is_unique
    assert (series > 0).all()


def test_check_weekday_balance_rejects_duplicate_and_missing_date():
    # Предохранитель проверяет только распределение по дням недели — окно с
    # дубликатом одной даты вместо пропущенной другой (тот же weekday) может
    # тихо сохранить баланс 8/8/8/... при фактически некорректной временной
    # оси (пропуск + дубликат). balanced обязан учитывать непрерывность и
    # уникальность дат, а не только частоты weekday.
    index = list(pd.period_range(start="2026-06-22", periods=56, freq="D"))  # понедельник
    index[10] = index[3]  # дубликат четверга вместо пропущенного четверга на позиции 10
    series = pd.Series(range(56), index=pd.PeriodIndex(index, freq="D"))

    result = check_weekday_balance(series)

    assert result["balanced"] is False


def test_daily_fixtures_train_window_is_balanced_by_weekday():
    # Предрегистрация: train = первые TRAIN_WINDOW_DAYS (56) строк, должны
    # покрывать ровно 8 полных недель, каждый день недели встречается 8 раз.
    for path in DAILY_FIXTURES.values():
        series = load_daily_dynamics_csv(path)
        train = truncate_series(series, len(series) - TRAIN_WINDOW_DAYS)

        assert len(train) == TRAIN_WINDOW_DAYS
        assert str(train.index[0]) == "2026-06-23"
        assert str(train.index[-1]) == "2026-08-17"

        balance = check_weekday_balance(train)
        assert balance["balanced"] is True
        assert set(balance["counts_by_weekday"].values()) == {8}


def test_daily_fixtures_holdout_is_two_days_not_three():
    # docs/EXPERIMENT_VECTOR_D.md подчёркивает: holdout — то, что реально
    # осталось после train (58 - 56 = 2), а не заранее посчитанное число (3),
    # которого в данных нет.
    for path in DAILY_FIXTURES.values():
        series = load_daily_dynamics_csv(path)
        holdout = series.iloc[TRAIN_WINDOW_DAYS:]

        assert len(holdout) == 2
        assert str(holdout.index[0]) == "2026-08-18"
        assert str(holdout.index[-1]) == "2026-08-19"


def test_fit_autoets_vector_daily_high_freq_has_no_seasonal_component():
    """Ключевая находка второй итерации: у «купить телефон» AutoETS(sp=7) не
    включает сезонную компоненту даже на полном train-окне — has_seasonal
    должен быть False, а не "измеренный плоский профиль" (см. docs/EXPERIMENT_VECTOR_D.md,
    вторая итерация, раздел 1/3).
    """
    series = load_daily_dynamics_csv(DAILY_FIXTURES["high_freq (купить телефон)"])
    train = truncate_series(series, len(series) - TRAIN_WINDOW_DAYS)

    vector = fit_autoets_vector(train, sp=7, calendar_anchor="weekday")

    assert vector.has_seasonal is False
    assert vector.to_dict()["seasonal"] is None


def test_fit_autoets_vector_daily_seasonal_has_seasonal_component():
    # Контраст с high_freq: «новогодние подарки» получает сезонный вектор.
    series = load_daily_dynamics_csv(DAILY_FIXTURES["seasonal (новогодние подарки)"])
    train = truncate_series(series, len(series) - TRAIN_WINDOW_DAYS)

    vector = fit_autoets_vector(train, sp=7, calendar_anchor="weekday")

    assert vector.has_seasonal is True
    assert len(vector.to_dict()["seasonal"]) == 7


def test_compare_vectors_marks_both_seasonal_false_when_either_vector_lacks_seasonality():
    """Регрессия на NaN-баг: если хотя бы один из двух векторов не имеет
    сезонной компоненты, seasonal_corr/seasonal_mae_rel_to_full_range должны
    быть NaN и both_seasonal=False — а не тихо "проходить" пороги устойчивости
    как false positive.
    """
    with_seasonal = ParamVector(
        level=100.0,
        trend=0.0,
        has_trend=False,
        has_seasonal=True,
        seasonal=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.7]),
        model_spec="mul/None/mul",
    )
    without_seasonal = ParamVector(
        level=100.0,
        trend=0.0,
        has_trend=False,
        has_seasonal=False,
        seasonal=np.zeros(7),
        model_spec="add/None/None",
    )

    comparison = compare_vectors(with_seasonal, without_seasonal)

    assert comparison["both_seasonal"] is False
    assert np.isnan(comparison["seasonal_corr"])
    assert np.isnan(comparison["seasonal_mae_rel_to_full_range"])


def test_is_comparison_unstable_requires_both_seasonal_before_using_seasonal_corr():
    """Регрессия: main() применяла seasonal_corr < порог напрямую, без проверки
    both_seasonal. Когда сравниваемый вектор теряет сезонность, seasonal_corr
    приходит NaN, а `nan < 0.5` в Python — False, поэтому сравнение тихо
    считалось "стабильным", хотя сезонность фактически не сравнивалась вовсе.
    is_comparison_unstable обязана относиться к both_seasonal=False как к
    сигналу нестабильности (наравне с реальным дрейфом), а не к пропуску.
    """
    comparison = {
        "level_diff_rel": 0.01,  # уровень стабилен
        "both_seasonal": False,
        "seasonal_spec_compatible": True,
        "seasonal_corr": float("nan"),
        "seasonal_mae_rel_to_full_range": float("nan"),
    }

    assert is_comparison_unstable(comparison, MONTHLY_STABILITY_THRESHOLDS) is True


def test_is_comparison_unstable_flat_profile_mae_rel_does_not_silently_pass():
    """Регрессия: когда полный сезонный профиль плоский (max == min), MAE
    относительно размаха — NaN, а `nan > порог` — False, из-за чего
    вырожденный плоский профиль тихо проходил порог стабильности MAE даже
    при both_seasonal=True. is_comparison_unstable обязана считать
    NaN-метрику сравнения нестабильной, а не пропускать её.
    """
    comparison = {
        "level_diff_rel": 0.01,
        "both_seasonal": True,
        "seasonal_spec_compatible": True,
        "seasonal_corr": 0.99,  # корреляция высокая
        "seasonal_mae_rel_to_full_range": float("nan"),  # но relative MAE не определён
    }

    assert is_comparison_unstable(comparison, DAILY_STABILITY_THRESHOLDS) is True


def test_is_comparison_unstable_nan_level_diff_rel_is_not_silently_stable():
    """Регрессия (Codex + /review, цикл 2): при full.level == 0 level_diff_rel
    приходит NaN, а `abs(nan) > порог` в Python — False. Комбинация с
    both_seasonal=True и валидным seasonal_corr раньше давала False целиком
    (уровень не измерен, но gate этого не замечал).
    """
    comparison = {
        "level_diff_rel": float("nan"),
        "both_seasonal": True,
        "seasonal_spec_compatible": True,
        "seasonal_corr": 0.99,
        "seasonal_mae_rel_to_full_range": 0.05,
    }

    assert is_comparison_unstable(comparison, DAILY_STABILITY_THRESHOLDS) is True


def test_is_comparison_unstable_nan_seasonal_corr_with_both_seasonal_true_is_not_silently_stable():
    """Регрессия (Codex + /review, цикл 2): both_seasonal=True не гарантирует
    seasonal_corr не-NaN — вырожденный (константный) сезонный вектор даёт
    np.corrcoef == NaN даже когда обе стороны формально "имеют сезонность".
    `nan < порог` в Python — False, что тихо считало бы такое сравнение
    стабильным.
    """
    comparison = {
        "level_diff_rel": 0.01,
        "both_seasonal": True,
        "seasonal_spec_compatible": True,
        "seasonal_corr": float("nan"),
        "seasonal_mae_rel_to_full_range": 0.05,
    }

    assert is_comparison_unstable(comparison, DAILY_STABILITY_THRESHOLDS) is True


def test_is_comparison_unstable_passes_genuinely_stable_comparison():
    comparison = {
        "level_diff_rel": 0.01,
        "both_seasonal": True,
        "seasonal_spec_compatible": True,
        "seasonal_corr": 0.99,
        "seasonal_mae_rel_to_full_range": 0.05,
    }

    assert is_comparison_unstable(comparison, DAILY_STABILITY_THRESHOLDS) is False


# --- Отложенные находки ревью PR #29 (issue #33) ---


def test_is_comparison_unstable_spec_mismatch_is_not_silently_stable():
    """Регрессия (issue #33, находка 4): при mul-сезонности на полном ряде и
    add-сезонности на укороченном seasonal_mae сравнивает коэффициенты около
    1.0 со сдвигами в единицах ряда — формально исчислимое, но бессмысленное
    число (на фикстурах второй итерации mid_freq давало mae_rel ~400 против
    порога 0.3). Подтверждено на реальном прогоне: несоответствие
    параметризации обязано трактоваться как неприменимое сравнение
    (нестабильность), даже если все остальные метрики в норме.
    """
    mul_seasonal = ParamVector(
        level=100.0,
        trend=0.0,
        has_trend=False,
        has_seasonal=True,
        seasonal=np.array([1.0, 1.1, 1.2, 1.0, 0.9, 0.8, 0.7]),
        model_spec="mul/add/mul/damped",
    )
    add_seasonal = ParamVector(
        level=100.0,
        trend=0.0,
        has_trend=False,
        has_seasonal=True,
        seasonal=np.array([10.0, 11.0, 12.0, 10.0, 9.0, 8.0, 7.0]),  # та же форма, другая шкала
        model_spec="add/add/add/damped",
    )

    comparison = compare_vectors(mul_seasonal, add_seasonal)

    # Корреляция инвариантна к шкале и остаётся идеальной...
    assert comparison["both_seasonal"] is True
    assert comparison["seasonal_corr"] == 1.0
    # ...но сравнение несопоставимо по параметризации — и обязано быть
    # нестабильным, а не проходить пороги на артефактных MAE.
    assert comparison["seasonal_spec_compatible"] is False
    assert is_comparison_unstable(comparison, DAILY_STABILITY_THRESHOLDS) is True


def test_compare_vectors_matching_seasonal_spec_is_compatible():
    mul_a = ParamVector(
        level=100.0,
        trend=0.0,
        has_trend=False,
        has_seasonal=True,
        seasonal=np.ones(7),
        model_spec="mul/add/mul/damped",
    )
    # Расхождение в error/trend/damped не влияет на сезонные метрики.
    mul_b = ParamVector(
        level=100.0,
        trend=0.0,
        has_trend=False,
        has_seasonal=True,
        seasonal=np.ones(7),
        model_spec="add/None/mul",
    )

    comparison = compare_vectors(mul_a, mul_b)

    assert comparison["seasonal_spec_compatible"] is True


def test_run_stability_check_valueerror_is_not_measurable(monkeypatch):
    """Регрессия (issue #33, находка 2, часть 1): ожидаемый отказ AutoETS
    (ValueError на слишком коротком ряде) по-прежнему репортится как
    not_measurable — экспериментальная данность, а не баг.
    """
    series = load_dynamics_csv(FIXTURES["high_freq (купить телефон)"])
    full_vector = fit_autoets_vector(series)

    def raise_value_error(*args, **kwargs):
        raise ValueError("two full seasonal cycles")

    monkeypatch.setattr("scripts.experiment_vector_d.fit_autoets_vector", raise_value_error)

    result = run_stability_check(series, "label", full_vector, drops=(3,))

    entry = result["truncated_minus_3"]
    assert entry["outcome"] == "not_measurable"
    assert "ValueError" in entry["error"]


def test_run_stability_check_unexpected_exception_propagates(monkeypatch):
    """Регрессия (issue #33, находка 2, часть 2): широкий except Exception
    маскировал баг в compare_vectors/to_dict (KeyError, несовместимость
    shape) под «модель не фитится» (not_measurable) — дефект кода
    репортился как экспериментальная данность. Неожиданные исключения
    обязаны пробрасываться наружу.
    """
    series = load_dynamics_csv(FIXTURES["high_freq (купить телефон)"])
    full_vector = fit_autoets_vector(series)

    def raise_key_error(*args, **kwargs):
        raise KeyError("баг в коде сравнения, а не отказ модели")

    monkeypatch.setattr("scripts.experiment_vector_d.fit_autoets_vector", raise_key_error)

    with pytest.raises(KeyError):
        run_stability_check(series, "label", full_vector, drops=(3,))


def test_compare_seasonal_cycles_reuses_provided_forecaster(monkeypatch):
    """Регрессия (issue #33, находка 3): compare_seasonal_cycles заново
    фитила тот же ряд, который main() уже зафитил для full_vector, — дублируя
    стоимость фита на каждую фикстуру. При переданном fitted_forecaster
    рефита происходить не должно.
    """
    series = load_dynamics_csv(FIXTURES["seasonal (новогодние подарки)"])
    fitted = _fit_autoets(series, sp=12)

    def no_refit(*args, **kwargs):
        raise AssertionError("дублирующий рефит: fitted_forecaster не переиспользован")

    monkeypatch.setattr("scripts.experiment_vector_d._fit_autoets", no_refit)

    result = compare_seasonal_cycles_within_full_series(series, sp=12, fitted_forecaster=fitted)

    assert "error" not in result
    assert len(result["cycle_2"]) == 12


def test_main_returns_nonzero_exit_code_when_stop_condition_triggered(capsys):
    """Регрессия (issue #33, находка 1): main() всегда возвращала 0, даже при
    stability_stop_condition_triggered=True — единственный сигнал о срыве
    стоп-условия был спрятан в JSON на stdout и требовал парсинга. На
    месячных фикстурах первой итерации стоп-условие срабатывает
    детерминированно (укороченные ряды 21/18 точек < 2 циклов →
    not_measurable), поэтому exit code обязан быть ненулевым.
    """
    exit_code = main()

    report = json.loads(capsys.readouterr().out)
    assert report["stability_stop_condition_triggered"] is True
    assert exit_code == 1


def test_load_daily_dynamics_csv_rejects_duplicate_and_missing_date(tmp_path: Path):
    """Регрессия (issue #33, находка 5): дневной парсер сортировал даты, но
    не проверял уникальность/непрерывность сам — защита была размазана по
    вызывающему коду (check_weekday_balance на train-подвыборке), а holdout-
    хвост и любые другие вызывающие оставались без проверки. Теперь парсер
    валидирует ось сам, целиком по файлу, как месячный.
    """
    content = (
        "﻿Дата;Число запросов;Доля от всех запросов, %;заголовок\r"
        "22.06.2026;1000;0,01;\r"
        "23.06.2026;1001;0,01;\r"
        "23.06.2026;1002;0,01;\r"  # дубликат вместо уникальной даты
        "25.06.2026;1003;0,01;\r"  # и пропуск 24.06
    )
    path = tmp_path / "broken_daily.csv"
    path.write_bytes(content.encode("utf-8"))

    with pytest.raises(ValueError, match="[Дд]убликат"):
        load_daily_dynamics_csv(path)


def test_statsforecast_is_a_diagnostic_backend_not_wired_into_the_stability_gate():
    """Регрессионный тест против дыры, найденной в независимом варианте MVP #23
    (issue #23, ветка mvp-23-b, коммит 0ff4532 «keep diagnostic backend outside
    stability gate»): там statsforecast по ошибке участвовал в one `all(...)` со
    sktime, из-за чего pass/fail стоп-условия мог решаться вспомогательным
    бэкендом, а не только предрегистрированным первичным (sktime). Дыра тихая —
    числа отчёта не меняются, но семантика gate меняется незаметно.

    Проверяем структурно: run_stability_check() — единственная функция, чей
    результат участвует в вычислении any_unstable/any_not_measurable в main()/
    main_daily() (см. код) — не принимает statsforecast-результаты и не имеет
    возможности на них ссылаться (statsforecast_cross_check добавляется в
    отчёт ПОСЛЕ вызова run_stability_check, отдельным полем, см. main_daily()).
    """
    signature = inspect.signature(run_stability_check)
    assert "statsforecast" not in str(signature).lower()

    source = inspect.getsource(run_stability_check)
    assert "statsforecast" not in source.lower()


def test_empirical_weekday_profile_matches_high_freq_saturday_dip():
    """Независимое подтверждение находки другого варианта MVP #23 (не
    копирование чисел, а собственный пересчёт): у «купить телефон»
    эмпирические средние по дням недели дают провал в субботу,
    качественно похожий на будни/выходные у сезонных фраз — слабый
    сигнал, но не плоский.
    """
    series = load_daily_dynamics_csv(DAILY_FIXTURES["high_freq (купить телефон)"])
    train = truncate_series(series, len(series) - TRAIN_WINDOW_DAYS)

    profile = empirical_weekday_profile(train)

    assert len(profile) == 7
    assert abs(profile.mean() - 1.0) < 1e-9
    # Суббота (индекс 5) — минимум профиля, как и у сезонных фраз этого отчёта.
    assert int(np.argmin(profile)) == 5


def test_fit_statsforecast_seasonal_profile_high_freq_has_seasonal_and_is_stable():
    """Независимая проверка граничного случая «купить телефон»
    (docs/EXPERIMENT_VECTOR_D.md, вторая итерация, раздел 3): statsforecast
    находит слабую, но устойчивую недельную форму там, где основной бэкенд
    её не видит вовсе.
    """
    series = load_daily_dynamics_csv(DAILY_FIXTURES["high_freq (купить телефон)"])
    train = truncate_series(series, len(series) - TRAIN_WINDOW_DAYS)

    full_profile, components = fit_statsforecast_seasonal_profile(train, sp=7)
    assert full_profile is not None, f"ожидалась сезонная модель, получено {components}"
    assert len(full_profile) == 7

    for drop in (7, 14):
        window = truncate_series(train, drop)
        profile, _ = fit_statsforecast_seasonal_profile(window, sp=7)
        assert profile is not None
        corr = float(np.corrcoef(full_profile, profile)[0, 1])
        # Слабая, но устойчивая форма: корреляция заметно выше нуля на обоих рефитах.
        assert corr > 0.9, f"drop={drop}: corr={corr}"

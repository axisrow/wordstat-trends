"""Тесты краткосрочных трендов на дневных данных (issue #22).

Дневные фикстуры — реальные выгрузки Вордстата (60-дневное окно, собраны
в MVP #23): tests/fixtures/dynamics_daily_*_mvp.csv. Формат — docs/DATA.md.
Синтетические ряды ниже построены от понедельника, чтобы границы недель
были предсказуемы.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wordstat_trends.short_term import (
    SPIKE_RATIO_THRESHOLD,
    DailySnapshot,
    detect_spikes,
    load_daily_snapshot,
    short_term_report,
    weekday_profile,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

DAILY_SEASONAL = FIXTURES_DIR / "dynamics_daily_seasonal_mvp.csv"
DAILY_MID_FREQ = FIXTURES_DIR / "dynamics_daily_mid_freq_mvp.csv"


def synthetic_daily_series(n_weeks: int = 8, weekday_value: float = 120.0, weekend_value: float = 50.0) -> pd.Series:
    """Ровный недельный профиль без тренда, начиная с понедельника."""
    start = pd.Period("2026-06-01", freq="D")  # понедельник
    days = pd.period_range(start=start, periods=n_weeks * 7, freq="D")
    values = [weekday_value if p.to_timestamp().weekday() < 5 else weekend_value for p in days]
    return pd.Series(values, index=days, name="count")


class TestLoadDailySnapshot:
    def test_real_fixture_parses_to_58_days_with_phrase(self):
        snapshot = load_daily_snapshot(DAILY_SEASONAL)
        assert snapshot.phrase == "новогодние подарки"
        assert len(snapshot.series) == 58
        assert snapshot.series.index[0] == pd.Period("2026-06-23", freq="D")
        assert snapshot.series.index[-1] == pd.Period("2026-08-19", freq="D")

    def test_measured_through_is_last_data_day_not_header_bound(self):
        # Конечная дата в заголовке графика — 20.08.2026 (эксклюзивная
        # граница), последняя строка данных — 19.08.2026. Дата замера —
        # последняя СТРОКА ДАННЫХ, не заголовок.
        snapshot = load_daily_snapshot(DAILY_SEASONAL)
        assert snapshot.measured_through == pd.Period("2026-08-19", freq="D")

    def test_header_only_file_raises_clear_error(self, tmp_path: Path):
        # Только заголовок, тело пустое: понятное сообщение по образцу
        # остальных ошибок парсинга, а не невнятный unpack от zip(*[]).
        path = tmp_path / "daily_header_only.csv"
        path.write_text("Дата;Число запросов;Доля;Заголовок", encoding="utf-8-sig")
        with pytest.raises(ValueError, match="Нет строк данных"):
            load_daily_snapshot(path)

    def test_monthly_fixture_rejected_by_daily_parser(self):
        # Месячная выгрузка начинается с «Период», дневная — с «Дата»;
        # дневной парсер не должен молча принимать месячный файл.
        with pytest.raises(ValueError, match="Дата"):
            load_daily_snapshot(FIXTURES_DIR / "dynamics_seasonal.csv")

    def test_gap_in_dates_raises(self, tmp_path: Path):
        series = synthetic_daily_series(n_weeks=2)
        path = tmp_path / "daily_gap.csv"
        lines = ["Дата;Число запросов;Доля;Заголовок"]
        # выкидываем СРЕДНИЙ день — окно цельное с краёв, но с дыркой внутри
        for period in list(series.index)[:5] + list(series.index)[6:]:
            lines.append(f"{period.to_timestamp().strftime('%d.%m.%Y')};100;0,1;".rstrip(";"))
        path.write_text("\r".join(lines), encoding="utf-8-sig")
        with pytest.raises(ValueError, match="Пропуски дней"):
            load_daily_snapshot(path)


class TestWeekdayProfile:
    def test_weekend_dip_survives_strong_growth_trend(self):
        # Фикстура «новогодние подарки» растёт в ~3 раза по уровню за окно,
        # но недельный профиль нормируется внутри каждой недели: провал
        # выходных должен проявиться и на растущем ряду.
        snapshot = load_daily_snapshot(DAILY_SEASONAL)
        profile = weekday_profile(snapshot.series)
        assert profile[5] < 0.8  # суббота
        assert profile[6] < 0.9  # воскресенье (в этой фикстуре — 0.82)
        assert profile[:4].min() > 1.0  # пн–чт выше среднего
        assert profile[4] > 0.9  # пятница почти на уровне будней (0.99)

    def test_profile_of_flat_weekly_series_is_flat(self):
        profile = weekday_profile(synthetic_daily_series(weekday_value=100.0, weekend_value=100.0))
        assert np.allclose(profile, 1.0, atol=0.05)

    def test_too_short_window_raises(self):
        # 2 недели: после отбрасывания неполных может остаться меньше данных,
        # чем нужно для всех 7 дней недели; 8 дней (одна неполная неделя) —
        # заведомо мало, профиль не оценивается, а не выдумывается.
        with pytest.raises(ValueError, match="не оценивается"):
            weekday_profile(synthetic_daily_series(n_weeks=1)[:-1])


class TestDetectSpikes:
    def test_injected_spike_is_detected_with_ratio_near_multiplier(self):
        series = synthetic_daily_series()
        # Всплеск x3 в среду четвёртой недели: 2026-06-01 — понедельник,
        # среда каждой недели — индексы 2, 9, 16, 23, ...
        spike_day = series.index[23]
        assert spike_day.to_timestamp().weekday() == 2  # среда
        spiked = series.copy()
        spiked.iloc[23] = spiked.iloc[23] * 3
        spikes = detect_spikes(spiked)
        assert [s.date for s in spikes] == [spike_day]
        assert spikes[0].ratio == pytest.approx(3.0, rel=0.05)
        assert spikes[0].weekday == "среда"

    def test_clean_weekly_series_has_no_spikes(self):
        assert detect_spikes(synthetic_daily_series()) == []

    def test_loo_prevents_spike_from_masking_itself(self):
        # Ожидание считается без оцениваемого дня (leave-one-week-out по
        # собственной неделе): даже всплеск x3 не должен поднять своё же
        # «ожидание» настолько, чтобы уйти под порог.
        series = synthetic_daily_series()
        spiked = series.copy()
        spiked.iloc[17] = spiked.iloc[17] * 2  # четверг третьей недели
        dates = [s.date for s in detect_spikes(spiked)]
        assert series.index[17] in dates

    def test_days_of_incomplete_edge_weeks_are_not_scored(self):
        # Однодневный всплеск в обрезанной крайней неделе окна не оценивается
        # и не создаёт ложный сигнал: неполная неделя исключена целиком.
        series = synthetic_daily_series(n_weeks=4)
        truncated = series.iloc[3:-3]  # неполные края, 2 полные недели в середине
        spiked = truncated.copy()
        spiked.iloc[-1] = spiked.iloc[-1] * 10  # последний день — неполная неделя
        assert detect_spikes(spiked) == []

    def test_fewer_than_two_full_weeks_raises(self):
        # Одной полной недели мало: оцениваемой неделе не с чем сравниваться —
        # честный отказ, а не профиль, подогнанный по единственной неделе.
        series = synthetic_daily_series(n_weeks=1)
        with pytest.raises(ValueError, match="2 полные недели"):
            detect_spikes(series)

    def test_threshold_is_preregistered_default(self):
        # Порог предрегистрирован (docs/SHORT_TERM.md) и не молчаливо
        # переопределяется между модулем и детектором.
        assert SPIKE_RATIO_THRESHOLD == 1.5


class TestShortTermReport:
    def test_report_carries_measurement_date_and_threshold(self):
        snapshot = load_daily_snapshot(DAILY_MID_FREQ)
        report = short_term_report(snapshot)
        assert report["measured_through"] == "2026-08-19"
        assert report["threshold"] == SPIKE_RATIO_THRESHOLD
        assert report["phrase"] == "курсы английского языка"
        assert report["n_days_window"] == 58

    def test_recent_spike_reported_old_spike_not(self):
        series = synthetic_daily_series(n_weeks=9)
        spiked = series.copy()
        spiked.iloc[10] *= 3  # давно: вторая неделя окна
        spiked.iloc[-3] *= 3  # недавно: последняя полная неделя
        snapshot = DailySnapshot(
            phrase="синтетика", series=spiked, measured_through=spiked.index[-1]
        )
        report = short_term_report(snapshot, last_days=7)
        dates = [s["date"] for s in report["spikes_recent"]]
        assert str(spiked.index[-3]) in dates
        assert str(spiked.index[10]) not in dates
        # Свежий всплеск — первым, сортировка по силе, а не по дате.
        assert dates[0] == str(spiked.index[-3])

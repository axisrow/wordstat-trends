#!/bin/sh
# Точка входа контейнера сбора (issue #8).
#
# Задачи: поднять Chrome с CDP (только на 127.0.0.1), дождаться готовности
# порта отладки, затем раз в сутки прогонять сбор по списку фраз с паузами
# между фразами и коммитить результаты в репозиторий машинным ключом.
#
# Cookies и SSH-ключи живут ТОЛЬКО на томе /app/data и никогда:
#   - не попадают в образ,
#   - не читаются из переменных окружения,
#   - не печатаются в логи (маскируем пути-аргументы при выводе).
set -eu

CDP_PORT="${CDP_PORT:-9222}"
USER_DATA_DIR="${USER_DATA_DIR:-/app/data/profile}"
COOKIES_FILE="${COOKIES_FILE:-/app/data/cookies.txt}"
PHRASES_FILE="${PHRASES_FILE:-/app/data/phrases.txt}"
RESULTS_DIR="${RESULTS_DIR:-/app/data/results}"
GIT_DIR="${GIT_DIR:-/app/data/repo}"
# Пауза между фразами, секунды. Непрерывный массовый парсинг не нужен и вреден.
PHRASE_DELAY_S="${PHRASE_DELAY_S:-30}"
# Пауза между суточными прогонами.
COLLECT_INTERVAL_S="${COLLECT_INTERVAL_S:-86400}"

mkdir -p "$USER_DATA_DIR" "$RESULTS_DIR"

# --- Chrome -----------------------------------------------------------------
# --remote-debugging-address=127.0.0.1: CDP — полное управление браузером,
# наружу его не отдаём ни при каком раскладе (эндпоинт запуска — отдельная
# задача #24, и он не проксирует CDP).
chromium \
    --headless=new \
    --no-sandbox \
    --disable-gpu \
    --disable-dev-shm-usage \
    --user-data-dir="$USER_DATA_DIR" \
    --remote-debugging-address=127.0.0.1 \
    --remote-debugging-port="$CDP_PORT" \
    about:blank &

CHROME_PID=$!

# Ждём готовности CDP (до 60с) — сборщик не должен стартовать раньше браузера.
i=0
until curl -fsS "http://127.0.0.1:${CDP_PORT}/json/version" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "entrypoint: Chrome CDP не поднялся на 127.0.0.1:${CDP_PORT} за 60с" >&2
        kill "$CHROME_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
echo "entrypoint: CDP готов на 127.0.0.1:${CDP_PORT}"

cleanup() {
    kill "$CHROME_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Сервис сбора: расписание + HTTP-эндпоинт запуска (issue #24) -------------
# Суточный цикл (как в #8: первый прогон сразу, далее раз в сутки, паузы между
# фразами, коммит результатов) и узкий HTTP-эндпоинт живут в одном
# Python-процессе (src/wordstat_trends/collect_service.py) и делят одну
# блокировку: запрос по HTTP при активном сборе получает 409, а не второй Chrome.
# Известное ограничение (axisrow/wordstat-cli#2): прогон может упасть на середине.
# Расписание запускать можно, полноту одного прогона не гарантируем —
# докачка недостающих фраз отдельной задачей.
#
# TRIGGER_TOKEN задаётся через dokku config (не в образ и не в репозиторий);
# без него HTTP-эндпоинт не поднимается, расписание работает.
CHROME_PID="$CHROME_PID" python -m wordstat_trends.collect_service &
SERVICE_PID=$!

cleanup() {
    kill "$SERVICE_PID" 2>/dev/null || true
    kill "$CHROME_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Умер сервис или Chrome (сервис сам завершается кодом 1 при смерти Chrome) —
# выходим, рестарт-политика Dokku поднимет контейнер заново.
while kill -0 "$CHROME_PID" 2>/dev/null && kill -0 "$SERVICE_PID" 2>/dev/null; do
    sleep 5
done
exit 1

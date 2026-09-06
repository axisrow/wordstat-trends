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

# --- Сбор по расписанию ------------------------------------------------------
# Известное ограничение (axisrow/wordstat-cli#2): прогон может упасть на середине.
# Расписание запускать можно, полноту одного прогона не гарантируем —
# докачка недостающих фраз отдельной задачей.
collect_once() {
    # Cookies подкладываются в профиль до старта сбора (scripts/upload_cookies.sh
    # на хосте). В логи содержимое и даже полный путь не печатаем.
    if [ ! -f "$PHRASES_FILE" ]; then
        echo "entrypoint: список фраз не найден (${PHRASES_FILE} не существует), прогон пропущен"
        return 0
    fi

    while IFS= read -r phrase; do
        case "$phrase" in ""|\#*) continue ;; esac
        echo "entrypoint: сбор фразы <${phrase}>"
        wordstat collect "$phrase" --output-dir "$RESULTS_DIR" || {
            echo "entrypoint: фраза <${phrase}> не собрана, продолжаем (см. wordstat-cli#2)" >&2
        }
        sleep "$PHRASE_DELAY_S"
    done < "$PHRASES_FILE"

    # Результаты коммитятся в репозиторий машинным ключом с правом записи.
    # Ключ — на томе; здесь он только используется, но не копируется и не логируется.
    if [ -d "$GIT_DIR/.git" ]; then
        git -C "$GIT_DIR" add . >/dev/null \
            && git -C "$GIT_DIR" -c user.name=collector -c user.email=collector@localhost \
                commit -m "chore(data): автосбор $(date -u +%Y-%m-%d)" --quiet \
            && git -C "$GIT_DIR" push --quiet origin HEAD:main \
            && echo "entrypoint: результаты закоммичены и отправлены" \
            || echo "entrypoint: коммит/пуш результатов не удался (не фатально)" >&2
    else
        echo "entrypoint: git-репозиторий результатов не настроен (${GIT_DIR}), коммит пропущен"
    fi
}

# Первый прогон сразу (при деплое удобно видеть, что всё живо), далее — раз в сутки.
while kill -0 "$CHROME_PID" 2>/dev/null; do
    collect_once
    sleep "$COLLECT_INTERVAL_S"
done

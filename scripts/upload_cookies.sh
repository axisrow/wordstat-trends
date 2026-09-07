#!/bin/sh
# Заливка cookies на хост Dokku в том контейнера сбора (issue #8).
#
# Cookies идут по SSH с laptops оператора — единственный путь. Они НЕ кладутся:
#   - в образ,
#   - в переменные окружения Dokku (dokku config),
#   - в логи (файл передаётся scp-потоком, содержимое не печатается).
#
# Использование:
#   scripts/upload_cookies.sh <dokku-host> <app-name> <путь-к-cookies.txt>
#
# После заливки контейнер перезапускается — Chrome подхватывает профиль на томе.
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <dokku-host> <app-name> <cookies.txt>" >&2
    exit 64
fi

HOST="$1"
APP="$2"
COOKIES_LOCAL="$3"

[ -f "$COOKIES_LOCAL" ] || {
    echo "ошибка: файл cookies не найден: $COOKIES_LOCAL" >&2
    exit 66
}

# Том контейнера монтируется как /app/data (см. deploy/Dockerfile и docs/DEPLOY.md).
# Права 600 и владелец uid 1000 (пользователь collector в контейнере):
# на хосте файл не должен быть читаем никем лишним.
VOLUME_DIR="/var/lib/dokku/data/storage/${APP}-data"

echo ">> передача cookies на ${HOST}:${VOLUME_DIR}/cookies.txt (содержимое не логируется)"
ssh "$HOST" "sudo mkdir -p '${VOLUME_DIR}' && sudo chown 1000:1000 '${VOLUME_DIR}'"
scp -p "$COOKIES_LOCAL" "${HOST}:/tmp/${APP}-cookies.txt"
ssh "$HOST" "sudo install -o 1000 -g 1000 -m 600 /tmp/${APP}-cookies.txt '${VOLUME_DIR}/cookies.txt' && rm -f /tmp/${APP}-cookies.txt"

echo ">> перезапуск ${APP} на ${HOST}, чтобы Chrome перечитал профиль"
ssh "$HOST" "dokku ps:restart ${APP}"

echo ">> готово. Проверка: ssh ${HOST} dokku logs ${APP} --tail 50"

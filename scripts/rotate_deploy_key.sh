#!/bin/sh
# Ротация машинного ключа wordstat-data (issue #58).
#
# Запускается С РАБОЧЕЙ СТАНЦИИ ОПЕРАТОРА (нужны ssh до хоста Dokku и gh
# с правами admin на axisrow/wordstat-data). Выполняет полный цикл:
#
#   1. Проверка истории data-репозитория на хосте: попадал ли .ssh_key
#      в коммиты (до фикса #50 ключ лежал в рабочем дереве repo/).
#   2. Генерация нового ed25519-ключа (без passphrase, только deploy).
#   3. Регистрация нового ключа как deploy key с правом записи на GitHub.
#   4. Отзыв ВСЕХ старых deploy keys репозитория (ротация = старые недействительны).
#   5. Установка ключа на том ВНЕ рабочего дерева repo/ (схема #50/PR #55):
#      /var/lib/dokku/data/storage/<app>-data/.ssh_key, core.sshCommand.
#   6. Проверка аутентификации и push новым ключом.
#
# Само решение (чистить ли историю filter-repo/BFG) скрипт не принимает:
# для приватного data-репозитория ротация ключа обнуляет ущерб — старый ключ
# бесполезен, даже если он остаётся в старых коммитах. Фиксация решения —
# в issue #58.
#
# Использование:
#   scripts/rotate_deploy_key.sh <dokku-host> <app-name>
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <dokku-host> <app-name>" >&2
    exit 64
fi

HOST="$1"
APP="$2"
DATA_REPO="axisrow/wordstat-data"
VOLUME_DIR="/var/lib/dokku/data/storage/${APP}-data"
SSH_COMMAND="ssh -i /app/data/.ssh_key -o StrictHostKeyChecking=accept-new"

command -v gh >/dev/null 2>&1 || {
    echo "ошибка: нужен gh с правами admin на ${DATA_REPO}" >&2
    exit 69
}
command -v ssh-keygen >/dev/null 2>&1 || {
    echo "ошибка: нужен ssh-keygen" >&2
    exit 69
}

# --- 1. История: попадал ли ключ в коммиты ------------------------------
echo ">> [1/6] проверка истории ${VOLUME_DIR}/repo на хосте ${HOST}"
FOUND=$(ssh "$HOST" "sudo -u 1000 git -C '${VOLUME_DIR}/repo' \
    log --all --full-history --oneline -- .ssh_key '**/.ssh_key'" || true)
if [ -n "$FOUND" ]; then
    echo "!! ВНИМАНИЕ: .ssh_key найден в истории data-репозитория:"
    echo "$FOUND"
    echo "   Ротация ниже обнуляет ущерб (старый ключ отзывается);"
    echo "   чистка истории filter-repo/BFG — отдельное решение, см. issue #58."
else
    echo "   .ssh_key в истории не найден — ротация профилактическая."
fi

# --- 2. Новый ключ -------------------------------------------------------
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
NEW_KEY="${WORKDIR}/machine_key_ed25519"
ssh-keygen -t ed25519 -N '' -C "wordstat-collector $(date -u +%Y-%m-%dT%H:%M:%SZ)" -f "$NEW_KEY" >/dev/null
echo ">> [2/6] новый ключ сгенерирован: ${NEW_KEY}"

# --- 3. Deploy key на GitHub --------------------------------------------
echo ">> [3/6] регистрация нового deploy key (write) на ${DATA_REPO}"
gh repo deploy-key add "${NEW_KEY}.pub" -w --title "wordstat-collector rotated $(date -u +%Y-%m-%d)" -R "$DATA_REPO"

# --- 4. Отзыв старых ключей ---------------------------------------------
echo ">> [4/6] отзыв старых deploy keys (все, кроме только что добавленного)"
NEW_FINGERPRINT=$(ssh-keygen -lf "${NEW_KEY}.pub" | awk '{print $2}')
for ID in $(gh api "repos/${DATA_REPO}/keys" -q '.[].id'); do
    KEY_FP=$(gh api "repos/${DATA_REPO}/keys/${ID}" -q '.key' \
        | ssh-keygen -lf /dev/stdin | awk '{print $2}')
    if [ "$KEY_FP" = "$NEW_FINGERPRINT" ]; then
        continue
    fi
    echo "   отзыв key id=${ID} (${KEY_FP})"
    gh api -X DELETE "repos/${DATA_REPO}/keys/${ID}"
done

# --- 5. Установка на том -------------------------------------------------
echo ">> [5/6] установка ключа на ${HOST}:${VOLUME_DIR}/.ssh_key (вне repo/)"
scp -q "${NEW_KEY}" "${NEW_KEY}.pub" "${HOST}:/tmp/"
ssh "$HOST" "sudo install -o 1000 -g 1000 -m 600 /tmp/machine_key_ed25519 '${VOLUME_DIR}/.ssh_key' \
    && rm -f /tmp/machine_key_ed25519 /tmp/machine_key_ed25519.pub \
    && sudo -u 1000 git -C '${VOLUME_DIR}/repo' config core.sshCommand '${SSH_COMMAND}' \
    && sudo rm -f '${VOLUME_DIR}/repo/.ssh_key'"

# --- 6. Проверка ----------------------------------------------------------
echo ">> [6/6] проверка: аутентификация GitHub и push новым ключом"
ssh "$HOST" "sudo -u 1000 ssh -i '${VOLUME_DIR}/.ssh_key' \
    -o StrictHostKeyChecking=accept-new -T git@github.com" 2>&1 \
    | grep -q 'successfully authenticated' \
    || { echo "ошибка: GitHub не принял новый ключ" >&2; exit 1; }
ssh "$HOST" "sudo -u 1000 git -C '${VOLUME_DIR}/repo' push --dry-run" \
    || { echo "ошибка: push --dry-run не прошёл" >&2; exit 1; }

echo ">> готово. Окончательная проверка боенным прогоном:"
echo "   ssh ${HOST} dokku ps:restart ${APP}"
echo "   ssh ${HOST} dokku logs ${APP} --tail 50   # убедиться, что push результатов прошёл"

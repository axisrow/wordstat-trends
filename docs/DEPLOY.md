# Деплой контейнера сбора на Dokku

Инфраструктура Фазы 0 ([#8](https://github.com/axisrow/wordstat-trends/issues/8)):
постоянно живущий Chrome с CDP, авторизованный в Яндексе, и `wordstat-cli`,
который раз в сутки собирает сотни фраз с паузами между ними и коммитит
результаты в репозиторий.

Сбор **не может жить в GitHub Actions**: раннеры одноразовые, авторизованного
Chrome там нет, а учётные данные в секреты не кладём — это путь к капче и
нарушение принципа проекта. Actions только публикуют статику, реагируя на
коммиты этого контейнера.

## Состав

| Файл | Назначение |
|---|---|
| `deploy/Dockerfile` | Chrome (chromium из debian), Python 3.12, зависимости из `uv.lock` |
| `deploy/entrypoint.sh` | старт Chrome с CDP на `127.0.0.1`, суточный цикл сбора, коммит результатов |
| `scripts/upload_cookies.sh` | заливка cookies на хост по SSH, прямо в том |

## Развёртывание

Приложение — не веб-сервис, поэтому не даём Dokku создавать внешний маршрут:
CDP должен быть доступен **только изнутри контейнера**.

```bash
# на хосте Dokku
dokku apps:create wordstat-collector

# том: профиль Chrome, cookies, фразы, результаты, git-репозиторий и ssh-ключ
sudo mkdir -p /var/lib/dokku/data/storage/wordstat-collector-data
sudo chown 1000:1000 /var/lib/dokku/data/storage/wordstat-collector-data
dokku storage:mount wordstat-collector /var/lib/dokku/data/storage/wordstat-collector-data:/app/data

# Никаких портов наружу: не настраиваем domains/ports для этого приложения.

git remote add dokku dokku@<host>:wordstat-collector
git push dokku main
```

Деплой — это `git push dokku`; редеплой не трогает том, профиль переживает
перезапуск.

### Подготовка тома

```bash
# список фраз (по одной на строку, # — комментарий)
scp phrases.txt dokku@<host>:/tmp/phrases.txt
ssh dokku@<host> "sudo install -o 1000 -g 1000 -m 644 /tmp/phrases.txt \
    /var/lib/dokku/data/storage/wordstat-collector-data/phrases.txt"

# git-репозиторий результатов + машинный ключ с правом записи (deploy key)
ssh dokku@<host>
sudo -u 1000 git clone git@github.com:axisrow/wordstat-data.git \
    /var/lib/dokku/data/storage/wordstat-collector-data/repo
sudo install -o 1000 -g 1000 -m 600 machine_key_ed25519 \
    /var/lib/dokku/data/storage/wordstat-collector-data/repo/.ssh_key
# и указать ключ в repo/.git/config через core.sshCommand:
#   ssh -i /app/data/repo/.ssh_key -o StrictHostKeyChecking=accept-new
```

Ключ живёт на томе, а не в образе и не в `dokku config`.

### Cookies

Единственный путь — по SSH с машины оператора:

```bash
scripts/upload_cookies.sh dokku@<host> wordstat-collector cookies.txt
```

Скрипт кладёт файл в том с правами `600`/uid 1000 и перезапускает контейнер.
Cookies **не кладутся** в образ, в переменные окружения и не печатаются в логи.

## Доступ к CDP — только изнутри

`entrypoint.sh` стартует Chrome с `--remote-debugging-address=127.0.0.1
--remote-debugging-port=9222`. Порт не публикуется (`EXPOSE` — только
декларация; маршрута и проброса портов у приложения нет). Открытый наружу
отладочный порт авторизованного Chrome — это вся сессия Яндекса любому, кто до
него дотянется, поэтому доступ возможен только через `dokku run`:

```bash
dokku run wordstat-collector curl -s http://127.0.0.1:9222/json/version
```

Отдельная задача [#24](https://github.com/axisrow/wordstat-trends/issues/24)
(после мержа #8) добавит узкий HTTP-эндпоинт запуска сбора — он не проксирует
CDP и принимает один POST со списком фраз.

## Диагностика

```bash
dokku logs wordstat-collector --tail 100   # ход сбора; фразы печатаются, cookies — никогда
dokku run wordstat-collector curl -s http://127.0.0.1:9222/json/version   # жив ли CDP
dokku ps:report wordstat-collector
dokku run wordstat-collector ls /app/data/results   # что накопилось
```

- **Chrome не поднялся / CDP не готов за 60с** — entrypoint завершает контейнер;
  смотрите `dokku logs`, затем запускайте `chromium` руками через `dokku run`.
- **Прогон упал на середине фразы** — ожидаемо: wordstat-cli#2 (устойчивость
  прогона) открыт. Расписание продолжает работать, полноту одного прогона до
  закрытия #2 не гарантируем.
- **Wordstat показывает страницу входа** — `wordstat collect` сам прекращает
  работу; это сигнал, что сессия протухла — обновите cookies (раздел выше).

## Открытый вопрос №4 эпика: живучесть сессии

Переживёт ли сессия Яндекса перезапуск контейнера и как часто обновлять cookies
— **выясняется на практике, наблюдение фиксируется здесь**:

> _(пока пусто: после первых недель эксплуатации записать — пережил ли
> редеплой, через сколько дней/недель впервые потребовалась заливка свежих
> cookies, был ли при этом капч-ответ)_

## Настройки окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `PHRASE_DELAY_S` | `30` | пауза между фразами, сек (главный «предохранитель» от капчи) |
| `COLLECT_INTERVAL_S` | `86400` | пауза между суточными прогонами |
| `CDP_PORT` | `9222` | порт CDP внутри контейнера (слушает `127.0.0.1`) |
| `PHRASES_FILE` | `/app/data/phrases.txt` | список фраз |
| `RESULTS_DIR` | `/app/data/results` | выгрузки `wordstat collect` |
| `GIT_DIR` | `/app/data/repo` | git-репозиторий для коммита результатов |

Меняются через `dokku config:set wordstat-collector PHRASE_DELAY_S=60` и
перезапуск. Переменных с учётными данными нет и быть не должно.

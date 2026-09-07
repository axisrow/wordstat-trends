# deploy/

Инфраструктура контейнера сбора (issue
[#8](https://github.com/axisrow/wordstat-trends/issues/8)): Dockerfile и
entrypoint для Dokku — Chrome с CDP, том под профиль, суточный цикл сбора.

HTTP-эндпоинт запуска сбора по требованию (issue
[#24](https://github.com/axisrow/wordstat-trends/issues/24)) — модуль
`src/wordstat_trends/collect_service.py`: entrypoint стартует его как
`python -m wordstat_trends.collect_service`; один процесс держит и расписание,
и узкий HTTP-интерфейс (`POST /collect`, `GET /status`) с общей блокировкой
от параллельного запуска.

Развёртывание, заливка cookies и диагностика — в
[docs/DEPLOY.md](../docs/DEPLOY.md).

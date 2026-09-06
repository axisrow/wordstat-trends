# Контейнер сбора (issue #8): Chrome с CDP + wordstat-cli на Python 3.12.
#
# Инварианты (проверяются тестом tests/test_deploy_container.py):
#   - cookies НЕ попадают в образ: слой профиля — только том /app/data;
#   - CDP слушает только 127.0.0.1 — наружу порт не публикуется;
#   - базовый образ один и тот же для сборки и рантайма (нет секретов в слоях сборки).

FROM python:3.12-slim

# Chromium из debian-stable: стабильная версия внутри образа важнее
# новизны — профиль на томе должен переживать обновления пакета без сюрпризов.
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium git openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости проекта отдельным слоем: uv.lock меняется реже исходников.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

# wordstat-cli и остальные зависимости — из uv.lock (включая git-пин `wordstat`).
ENV PATH="/app/.venv/bin:$PATH"

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Профиль Chrome и результаты живут на томе, а не в слоях образа.
# Dokku: dokku storage:mount wordstat-collector /var/lib/dokku/services/...:/app/data
VOLUME ["/app/data"]
ENV USER_DATA_DIR=/app/data/profile \
    PHRASES_FILE=/app/data/phrases.txt \
    RESULTS_DIR=/app/data/results

# Не root: Chrome в контейнере от root — лишняя поверхность атаки.
RUN useradd --create-home --uid 1000 collector \
    && mkdir -p /app/data \
    && chown -R collector:collector /app
USER collector

EXPOSE 9222

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

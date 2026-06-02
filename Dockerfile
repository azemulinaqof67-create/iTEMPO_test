FROM python:3.12-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Установка менеджера пакетов uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Копирование описания зависимостей для эффективного кэширования слоев Docker
COPY pyproject.toml uv.lock ./

# Установка зависимостей в системное окружение Python
RUN uv pip install --system --no-cache -r pyproject.toml

# Копирование исходного кода проекта
COPY . .

# Экспонируем порты для FastAPI сервера (8000) и панели администратора (8080)
EXPOSE 8000 8080

# Запуск всех ботов (Telegram + MAX) и API сервера
CMD ["python", "run_all_bots.py"]

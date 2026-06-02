"""
Точка входа для запуска только MAX бота.

Использование:
    uv run run_max_bot.py
"""
import asyncio
import logging
import os
import warnings

from dotenv import load_dotenv

# Игнорируем SyntaxWarning от сторонних библиотек (например, pydub в Python 3.12)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Настройка прокси из .env (выполняется до импорта AI-клиентов)
load_dotenv()
proxy_url = os.getenv("BOT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY")
force_proxy = os.getenv("BOT_FORCE_PROXY") or os.getenv("FORCE_PROXY")

if proxy_url and force_proxy == "1":
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    print(f"Прокси применен: {proxy_url}")

# Настройка логирования — ОБЯЗАТЕЛЬНО до импортов src.*
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ВАЖНО: импорты ниже должны оставаться строго после настройки прокси и логирования
from src.core.config import Config
from src.interfaces.max_bot import run_max_bot

if __name__ == "__main__":
    try:
        config = Config.from_env()
    except Exception as e:
        print(f"ОШИБКА загрузки конфигурации: {e}")
        print("Скопируйте env.example в .env и заполните токены.")
        raise SystemExit(1)

    token_preview = (config.max_token[:8] + "...") if config.max_token else "НЕ ЗАДАН"
    print(f"MAX_TOKEN: {token_preview}")

    if not config.max_token:
        print("ОШИБКА: MAX_TOKEN не задан в .env")
        print("Получите токен в разделе: Chat-боты -> Интеграция -> Получить токен")
        raise SystemExit(1)

    print("Запуск MAX бота...")
    asyncio.run(run_max_bot(config))

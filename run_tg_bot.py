"""
Точка входа для запуска только Telegram бота.

Использование:
    uv run run_tg_bot.py

Аналог run_bot.py — оба файла запускают только Telegram.
"""
import os
import warnings

from dotenv import load_dotenv

# Игнорируем SyntaxWarning от сторонних библиотек (например, pydub в Python 3.12)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Автоматическая настройка прокси из .env (выполняется до импорта AI-клиентов)
load_dotenv()
proxy_url = os.getenv("BOT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY")
force_proxy = os.getenv("BOT_FORCE_PROXY") or os.getenv("FORCE_PROXY")

if proxy_url and force_proxy == "1":
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    print(f"✅ Прокси применен локально для процесса бота: {proxy_url}")

# ВАЖНО: Импорты ниже должны оставаться строго после настройки прокси
from src.core.config import Config
from src.interfaces.telegram_bot import run_bot

if __name__ == "__main__":
    try:
        config = Config.from_env()
    except Exception as e:
        print(f"ОШИБКА: {e}")
        print("Скопируйте env.example в .env и заполните токены.")
        raise SystemExit(1)

    print("Запуск Telegram бота...")
    run_bot(config, update_db=False)

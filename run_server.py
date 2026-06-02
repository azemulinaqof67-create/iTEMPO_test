"""
Точка входа для запуска FastAPI сервера.

Опционально может запустить Telegram бота вместе с сервером.
"""
import argparse
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
    print(f"✅ Прокси применен локально: {proxy_url}")

from src.core.config import Config
from src.interfaces.api_server import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Assistant Server")
    parser.add_argument(
        "--with-bot",
        action="store_true",
        help="Запустить Telegram бота вместе с сервером"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Хост для сервера (по умолчанию: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для сервера (по умолчанию: 8000)"
    )
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Обновить базу знаний перед запуском"
    )
    
    args = parser.parse_args()
    
    try:
        config = Config.from_env()
    except Exception as e:
        print(f"ОШИБКА: {e}")
        print("Скопируйте env.example в .env и заполните токены.")
        raise SystemExit(1)
    
    print("\n" + "="*40)
    print(" КОРПОРАТИВНЫЙ ИИ АССИСТЕНТ (SERVER)")
    print("="*40)
    if args.with_bot:
        print("Режим: Сервер + Telegram бот")
    else:
        print("Режим: Только сервер")
    print("="*40 + "\n")
    
    run_server(
        config, 
        with_bot=args.with_bot, 
        host=args.host, 
        port=args.port
    )

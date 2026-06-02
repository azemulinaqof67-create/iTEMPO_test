"""
Точка входа для запуска Telegram + MAX ботов одновременно.

Архитектура:
    - Оба бота (Telegram и MAX) запускаются параллельно в едином event loop в главном потоке.
    - Оба бота используют один и тот же экземпляр AssistantService.
    - При наличии вебхуков для MAX бота FastAPI-сервер запускается в отдельном фоновом потоке, чтобы не блокировать основной event loop.

Использование:
    uv run run_all_bots.py
"""
import asyncio
import logging
import os
import threading
import warnings

# Игнорируем SyntaxWarning от сторонних библиотек (например, pydub в Python 3.12)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from dotenv import load_dotenv

# Настройка прокси из .env (выполняется до импорта AI-клиентов)
load_dotenv()
proxy_url = os.getenv("BOT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY")
force_proxy = os.getenv("BOT_FORCE_PROXY") or os.getenv("FORCE_PROXY")

if proxy_url and force_proxy == "1":
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    # Исключаем локальные адреса из прокси
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    print(f"✅ Прокси применен: {proxy_url} (NO_PROXY: {os.environ['NO_PROXY']})")

# ВАЖНО: импорты ниже должны оставаться строго после настройки прокси
from src.assistant.assistant import AssistantService
from src.core.clients import ClientManager
from src.core.config import Config
from src.interfaces.max_bot import run_max_bot
from src.interfaces.telegram_bot import create_telegram_app
from admin.app import create_admin_app, set_bot_instances

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# Приглушаем шумные библиотеки
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _run_all(config: Config):
    """Основная корутина: запускает оба бота параллельно в одном event loop."""

    # ── Общий AssistantService для обоих ботов ──────────────────────────
    logger.info("⚙️  Инициализация AssistantService...")
    assistant = AssistantService(config)
    await assistant.initialize()

    # Предзагрузка моделей (устраняет задержку на первых запросах)
    client_manager = ClientManager.get_instance(config)
    client_manager.preload_models()
    logger.info("✅ Модели предзагружены.")

    tg_enabled = bool(config.telegram_token)
    max_enabled = bool(config.max_token)

    if not tg_enabled and not max_enabled:
        logger.error("❌ Не задан ни TELEGRAM_TOKEN, ни MAX_TOKEN. Нечего запускать.")
        return

    tasks = []

    # ── Подготовка Telegram ─────────────────────────────────────────────
    if tg_enabled:
        logger.info("🔵 Подготовка Telegram бота...")
        tg_app = create_telegram_app(config, assistant)
        
        async def run_tg():
            logger.info("🔵 Telegram бот: запуск polling...")
            await tg_app.initialize()
            await tg_app.start()
            await tg_app.updater.start_polling()
            
            # Ждем, пока бот работает или пока нас не отменят
            try:
                while tg_app.updater.running:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("🔵 Telegram бот: получен сигнал отмены.")
            finally:
                logger.info("🔵 Telegram бот: завершение работы...")
                if tg_app.updater.running:
                    await tg_app.updater.stop()
                if tg_app.running:
                    await tg_app.stop()
                await tg_app.shutdown()
        
        tasks.append(run_tg())
    else:
        logger.warning("⚠️  TELEGRAM_TOKEN не задан — Telegram бот пропущен.")

    # ── Подготовка MAX ──────────────────────────────────────────────────
    if max_enabled:
        logger.info("🟢 Подготовка MAX бота...")
        tasks.append(run_max_bot(config, assistant=assistant))

        # Если включен вебхук, нужно обязательно запустить FastAPI сервер
        if config.max_webhook_url:
            logger.info("🌐 Подготовка API Сервера для приема вебхуков...")
            import uvicorn
            from src.interfaces.api_server import app
            
            # Передаем конфигурацию в FastAPI
            app.state.config = config
            
            # Запускаем uvicorn в отдельном потоке, чтобы он не конфликтовал с loop
            def run_uvicorn():
                import uvicorn
                # uvicorn.run сам создаст правильный event loop для своего потока
                api_host = os.getenv("API_HOST", "0.0.0.0")
                uvicorn.run(app, host=api_host, port=8000, log_level="info")

            api_thread = threading.Thread(target=run_uvicorn, daemon=True)
            api_thread.start()
            logger.info("✅ API Сервер запущен в фоновом потоке на 0.0.0.0:8000")

    else:
        logger.warning("⚠️  MAX_TOKEN не задан — MAX бот пропущен.")

    # ── Панель администратора ────────────────────────────────────────────
    if config.admin_enabled:
        logger.info(f"🖥️  Запуск панели администратора на порту {config.admin_port}...")
        admin_app = create_admin_app(config, assistant)
        
        # Обновляем ссылки на ботов для отображения статуса
        tg_running = tg_enabled and 'tg_app' in dir()
        set_bot_instances(
            tg_app=tg_app if tg_enabled and 'tg_app' in locals() else None,
            max_running=max_enabled
        )
        
        def run_admin_server():
            import uvicorn
            admin_host = os.getenv("ADMIN_HOST", "0.0.0.0")
            uvicorn.run(
                admin_app,
                host=admin_host,
                port=config.admin_port,
                log_level="warning"
            )
        
        admin_thread = threading.Thread(target=run_admin_server, daemon=True)
        admin_thread.start()
        logger.info(f"✅ Панель администратора: http://localhost:{config.admin_port}")
    else:
        logger.info("ℹ️  Панель администратора отключена (ADMIN_ENABLED=0)")

    # ── Совместный запуск ───────────────────────────────────────────────
    if tasks:
        logger.info(f"🚀 Запуск {len(tasks)} ботов в едином event loop...")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
    
    # ── Мягкое закрытие базы данных ─────────────────────────────────────
    try:
        if hasattr(client_manager, '_shared_qdrant') and client_manager._shared_qdrant:
            client_manager._shared_qdrant.close()
            logger.info("✅ Qdrant база отключена штатно.")
    except Exception:
        pass

    logger.info("👋 Все боты остановлены.")


if __name__ == "__main__":
    try:
        config = Config.from_env()
        
        # Диагностика прокси
        logger.info("--- Конфигурация прокси ---")
        logger.info(f"BOT_HTTPS_PROXY: {config.https_proxy}")
        logger.info(f"BOT_FORCE_PROXY: {config.force_proxy}")
        logger.info("---------------------------")
    except Exception as e:
        print(f"ОШИБКА загрузки конфигурации: {e}")
        print("Скопируйте env.example в .env и заполните токены.")
        raise SystemExit(1)

    print("=" * 50)
    print("  Запуск ботов:")
    print(f"  Telegram : {'[ON]' if config.telegram_token else '[OFF: NO TOKEN]'}")
    print(f"  MAX      : {'[ON]' if config.max_token else '[OFF: NO TOKEN]'}")
    print("=" * 50)

    try:
        asyncio.run(_run_all(config))
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")

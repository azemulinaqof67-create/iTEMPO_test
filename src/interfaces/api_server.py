"""
Async FastAPI сервер.

Использует AssistantService для обработки запросов.
Поддерживает опциональный запуск Telegram бота.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.assistant.assistant import AssistantService
from src.core.config import Config
from src.core.exceptions import AssistantError
from src.rag.ingestion.document_processor import DocumentProcessor
from src.rag.ingestion.embeddings import EmbeddingService

import os
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=log_level)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="AI Assistant Server")

# Глобальный экземпляр ассистента
_assistant: Optional[AssistantService] = None


# Модели данных
class QueryRequest(BaseModel):
    query: str
    limit: int = 15
    session_id: Optional[str] = None
    user_name: Optional[str] = None
    user_login: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    resolution_status: str = "unknown"
    ticket_offer_available: bool = False
    ticket_created: bool = False
    ticket_number: Optional[str] = None
    ticket_draft_saved: bool = False
    ticket_creation_reason: Optional[str] = None


class TicketCreateRequest(BaseModel):
    query: str
    assistant_answer: str
    session_id: Optional[str] = None
    user_name: Optional[str] = None
    user_login: Optional[str] = None
    extra_info_1: Optional[str] = None
    extra_info_2: Optional[str] = None
    extra_fields: Optional[Dict[str, Any]] = None


class TicketCreateResponse(BaseModel):
    ticket_created: bool
    ticket_number: Optional[str]
    ticket_draft_saved: bool
    ticket_creation_reason: str
    fallback_all_used: bool = False
    long_context_summary_used: bool = False
    created_via: str = "AI_ASSISTANT"
    trace_id: Optional[str] = None
    auth_scheme_used: str = "none"
    delegation_status: str = "not_attempted"
    error_code: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера"""
    global _assistant
    logger.info("--- ЗАПУСК СЕРВЕРА ---")

    # Создаем ассистента
    config = app.state.config
    _assistant = AssistantService(config)
    await _assistant.initialize()
    logger.info("AssistantService initialized and LangGraph compiled")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка ресурсов при остановке сервера"""
    logger.info("--- ОСТАНОВКА СЕРВЕРА ---")

    # Закрываем все клиенты для предотвращения ошибок в __del__
    try:
        from src.core.clients import ClientManager

        ClientManager.get_instance().close_all()
        logger.info("All clients closed successfully")
    except Exception as e:
        logger.error(f"Error closing clients: {e}")


@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Эндпоинт для чата (используется десктопным клиентом).

    ИСПРАВЛЕНО: Теперь полностью async!
    """
    logger.info(f"Received chat request: {request.query}")

    try:
        if _assistant is None:
            raise HTTPException(status_code=500, detail="Assistant not initialized")

        result = await _assistant.process_text_query(
            request.query,
            limit=request.limit,
            session_id=request.session_id,
            platform="api",
            user_name=request.user_name,
        )

        return QueryResponse(
            answer=result["answer"],
            context=result["context"],
            resolution_status=result.get("resolution_status", "unknown"),
            ticket_offer_available=result.get("ticket_offer_available", False),
            ticket_created=result.get("ticket_created", False),
            ticket_number=result.get("ticket_number"),
            ticket_draft_saved=result.get("ticket_draft_saved", False),
            ticket_creation_reason=result.get("ticket_creation_reason"),
        )

    except AssistantError as e:
        logger.error(f"Assistant error: {e}")
        # Проверяем, это ошибка занятости или другая ошибка
        error_msg = str(e)
        if "Пожалуйста, подождите ответа на предыдущее сообщение" in error_msg:
            # Возвращаем специальный статус для занятости
            raise HTTPException(status_code=429, detail=error_msg) from e
        else:
            raise HTTPException(status_code=500, detail=error_msg) from e
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/ticket/create", response_model=TicketCreateResponse)
async def create_ticket_endpoint(request: TicketCreateRequest):
    """Создание helpdesk заявки после сигнала пользователя 'не помогло'."""
    try:
        if _assistant is None:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        logger.info(
            "Ticket create requested: user=%s login=%s query_len=%s extra1=%s extra2=%s",
            request.user_name,
            request.user_login,
            len(request.query or ""),
            bool(request.extra_info_1),
            bool(request.extra_info_2),
        )

        result = await _assistant.create_helpdesk_ticket(
            query=request.query,
            assistant_answer=request.assistant_answer,
            session_id=request.session_id,
            user_name=request.user_name,
            user_login=request.user_login,
            extra_info_1=request.extra_info_1,
            extra_info_2=request.extra_info_2,
            extra_fields=request.extra_fields,
        )

        return TicketCreateResponse(
            ticket_created=result.ticket_created,
            ticket_number=result.ticket_number,
            ticket_draft_saved=result.ticket_draft_saved,
            ticket_creation_reason=result.ticket_creation_reason,
            fallback_all_used=result.fallback_all_used,
            long_context_summary_used=result.long_context_summary_used,
            created_via=result.created_via,
            trace_id=result.trace_id,
            auth_scheme_used=result.auth_scheme_used,
            delegation_status=result.delegation_status,
            error_code=result.error_code,
        )
    except AssistantError as e:
        logger.error(f"Assistant error in ticket create: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error in create_ticket_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/update_db")
async def update_db():
    """Принудительное обновление базы знаний"""
    try:
        config = app.state.config

        # Обновление базы
        processor = DocumentProcessor(config)
        chunks = processor.prepare_chunks()

        embedding_service = EmbeddingService(config)
        await embedding_service.update_database(chunks)

        return {"status": "success", "message": "Knowledge base updated"}
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class ClearHistoryRequest(BaseModel):
    session_id: str


@app.post("/api/clear_history")
async def clear_history(request: ClearHistoryRequest):
    """Очистка истории сообщений для сессии"""
    try:
        if _assistant is None:
            raise HTTPException(status_code=500, detail="Assistant not initialized")

        if _assistant.chat_history:
            await _assistant.chat_history.clear_history(request.session_id, clear_summary=True)
            return {
                "status": "success",
                "message": f"History cleared for session {request.session_id}",
            }
        else:
            return {"status": "error", "message": "Chat history is disabled"}

    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/reload")
async def reload_config():
    """Горячая перезагрузка конфигурации (API ключи, модели)"""
    try:
        global _assistant

        config = app.state.config

        # 1. Перезагружаем Config (быстро)
        config.reload()
        logger.info("Config reloaded via API")

        # 2. Обновляем ClientManager в executor (может блокировать)
        import asyncio

        from src.core.clients import ClientManager

        client_manager = ClientManager.get_instance()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, client_manager.reload_clients)

        # 3. Обновляем AssistantService
        if _assistant:
            _assistant.reload_services()

        return {
            "status": "success",
            "message": "Configuration reloaded",
            "config": {
                "api_key": config.gemini_api_key[:10] + "...",
                "text_model": config.text_model,
                "audio_model": config.audio_model,
                "embedding_model": config.embedding_model,
                "text_api_version": config.text_api_version or "авто",
                "live_api_version": config.live_api_version,
                "embedding_api_version": config.embedding_api_version,
            },
        }

    except Exception as e:
        logger.error(f"Reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/test")
async def test_cloudflare():
    """Тестовый эндпоинт для проверки прохождения трафика через Cloudflare"""
    logger.info("======================================")
    logger.info("==> CLOUDFLARE ПРОБИЛСЯ К СЕРВЕРУ! <==")
    logger.info("======================================")
    return {"status": "CLOUDFLARE WORKS!"}

from fastapi import Request, Header

@app.post("/webhook/max")
async def max_webhook(
    request: Request,
    x_max_bot_api_secret: Optional[str] = Header(None)
):
    """Эндпоинт для приема событий от MAX мессенджера через Webhook."""
    logger.info("======================================")
    logger.info("==> ВХОДЯЩИЙ ЗАПРОС НА WEBHOOK MAX! <==")
    logger.info("======================================")
    
    config = app.state.config
    
    # Проверка секрета если он настроен
    if config.max_webhook_secret and x_max_bot_api_secret != config.max_webhook_secret:
        logger.warning(f"ВНИМАНИЕ: Секреты не совпали! Ожидался: {config.max_webhook_secret}, Пришел: {x_max_bot_api_secret}")
        logger.warning(f"Все заголовки запроса: {request.headers}")
        # ВРЕМЕННО пропускаем запрос для отладки, не выбрасываем 403
        # raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        data = await request.json()
        logger.info(f"Получены данные вебхука: {data}")
        
        # Получаем клиента
        from src.interfaces.max_bot import _dispatch_event, MaxBotClient
        
        # Используем кэшированный клиент в app.state
        if not hasattr(app.state, "max_client"):
            app.state.max_client = MaxBotClient(token=config.max_token)
            
        _max_client = app.state.max_client

        # Обрабатываем событие
        asyncio.create_task(_dispatch_event(data, _max_client, _assistant))
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Ошибка при обработке webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def run_server(config: Config, with_bot: bool = False, host: str = "0.0.0.0", port: int = 8000):
    """
    Запуск FastAPI сервера.

    Args:
        config: Конфигурация
        with_bot: Запустить Telegram бота в отдельном потоке
        host: Хост для сервера
        port: Порт для сервера
    """
    # Сохраняем конфиг в app.state
    app.state.config = config

    # Запуск бота в отдельном потоке (если нужно)
    if with_bot:
        logger.info("Starting Telegram Bot in background thread...")
        from src.interfaces.telegram_bot import run_bot

        # ВАЖНО: Мы НЕ передаем общий экземпляр _assistant в поток бота.
        # Бот должен создать свой экземпляр внутри своего потока/цикла событий,
        # чтобы избежать конфликтов с httpx клиентами и другими async ресурсами.
        bot_thread = threading.Thread(
            target=run_bot,
            args=(config, False, None),  # assistant=None заставит run_bot создать свой
            daemon=True,
        )
        bot_thread.start()
        logger.info("Telegram Bot started.")

    # Запуск сервера
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


async def _update_database(config: Config):
    """Вспомогательная функция для обновления базы"""
    processor = DocumentProcessor(config)
    chunks = processor.prepare_chunks()

    embedding_service = EmbeddingService(config)
    await embedding_service.update_database(chunks)


if __name__ == "__main__":
    import argparse

    from src.core.config import Config

    parser = argparse.ArgumentParser(description="AI Assistant Server")
    parser.add_argument(
        "--with-bot",
        action="store_true",
        help="Запустить Telegram бота вместе с сервером",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Хост для сервера (по умолчанию: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Порт для сервера (по умолчанию: 8000)")
    parser.add_argument("--update-db", action="store_true", help="Обновить базу знаний перед запуском")

    args = parser.parse_args()

    try:
        config = Config.from_env()
    except Exception as e:
        print(f"ОШИБКА: {e}")
        print("Скопируйте env.example в .env и заполните токены.")
        raise SystemExit(1) from e

    # Обновление базы (если нужно)
    if args.update_db:
        print("\n--- Обновление базы знаний ---")
        asyncio.run(_update_database(config))
        print("--- Обновление завершено ---\n")

    print("\n" + "=" * 40)
    print(" КОРПОРАТИВНЫЙ ИИ АССИСТЕНТ (SERVER)")
    print("=" * 40)
    if args.with_bot:
        print("Режим: Сервер + Telegram бот")
    else:
        print("Режим: Только сервер")
    print("=" * 40 + "\n")

    run_server(config, with_bot=args.with_bot, host=args.host, port=args.port)

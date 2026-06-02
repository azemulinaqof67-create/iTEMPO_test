"""
Async Telegram бот интерфейс.

Использует AssistantService для обработки запросов.
"""

import asyncio
import logging
import re
import time
from typing import Optional

import pytz
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Defaults,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from src.assistant.assistant import AssistantService
from src.core.config import Config
from src.core.exceptions import AssistantError
from src.rag.ingestion.document_processor import DocumentProcessor
from src.rag.ingestion.embeddings import EmbeddingService

import os
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=log_level)
logger = logging.getLogger(__name__)


# Глобальный экземпляр ассистента (инициализируется в run_bot)
_assistant: Optional[AssistantService] = None

from src.core.constants import COMPANIES

def _get_company_keyboard() -> InlineKeyboardMarkup:
    """Возвращает клавиатуру выбора предприятия"""
    keyboard = []
    # Группируем по 2 кнопки в ряд
    items = list(COMPANIES.items())
    for i in range(0, len(items), 2):
        row = [
            InlineKeyboardButton(name, callback_data=f"company_{code}")
            for code, name in items[i:i+2]
        ]
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)


def _get_main_menu_keyboard(voice_enabled: bool) -> InlineKeyboardMarkup:
    """Клавиатура главного меню"""
    voice_label = "🎙 Голос: ВКЛ" if voice_enabled else "💬 Голос: ВЫКЛ"
    keyboard = [
        [InlineKeyboardButton("🏭 Выбрать компанию", callback_data="menu_company")],
        [InlineKeyboardButton(voice_label, callback_data="menu_voice")],
        [InlineKeyboardButton("🗑 Очистить историю", callback_data="menu_clear")]
    ]
    return InlineKeyboardMarkup(keyboard)


def _get_company_menu_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора компании с кнопкой Назад"""
    keyboard = []
    items = list(COMPANIES.items())
    for i in range(0, len(items), 2):
        row = [
            InlineKeyboardButton(name, callback_data=f"company_{code}")
            for code, name in items[i:i+2]
        ]
        keyboard.append(row)
    # Добавляем кнопку назад
    keyboard.append([InlineKeyboardButton("⬅️ Назад в меню", callback_data="menu_main")])
    return InlineKeyboardMarkup(keyboard)


def _get_menu_text(voice_enabled: bool, company_name: Optional[str] = None) -> str:
    """Формирует текст для меню"""
    status = "ВКЛЮЧЕНЫ — отвечаю голосом" if voice_enabled else "ВЫКЛЮЧЕНЫ — отвечаю текстом"
    company = company_name if company_name else "Не выбрано"
    
    text = (
        "<b>🏠 Главное меню ассистента</b>\n\n"
        f"🏭 Предприятие: <b>{company}</b>\n"
        f"🎙 Голосовые ответы: <b>{status}</b>\n\n"
        "Пожалуйста, выберите подходящий пункт меню: 👇"
    )
    return text


def _get_user_friendly_error(error_text: str, is_voice: bool = False) -> str:
    """
    Преобразует технические ошибки API в понятные пользователю сообщения.

    Коды ошибок Gemini API:
    - 503 UNAVAILABLE: модель перегружена
    - 429 RESOURCE_EXHAUSTED: превышен лимит запросов
    - 400 INVALID_ARGUMENT: некорректный запрос
    - 400 FAILED_PRECONDITION: региональные ограничения
    - 403 PERMISSION_DENIED: нет доступа
    - 404 NOT_FOUND: ресурс не найден
    - 500 INTERNAL: внутренняя ошибка сервера
    """
    err = error_text.lower()

    # Все API ключи исчерпаны
    if "все api ключи исчерпали лимиты" in err or "all api keys" in err or "all keys exhausted" in err:
        return "⚠️ Все API ключи исчерпали лимиты. Пожалуйста, попробуйте позже или обратитесь к администратору."

    # 503 - Модель перегружена
    if "503" in error_text or "unavailable" in err or "overloaded" in err:
        return "⚡ Модель сейчас перегружена. Попробуйте через минуту."

    # 429 - Лимит запросов
    if "429" in error_text or "resource_exhausted" in err or "rate limit" in err:
        return "⏳ Превышен лимит запросов. Система автоматически переключается на резервный ключ..."

    # 400 FAILED_PRECONDITION - Региональные ограничения
    if "failed_precondition" in err or "user location is not supported" in err:
        if is_voice:
            return "🌍 Голосовой API недоступен в вашем регионе. Используйте текстовые сообщения."
        return "🌍 API недоступен в текущем регионе."

    # 400 INVALID_ARGUMENT - Некорректный запрос
    if "400" in error_text and "invalid" in err:
        return "❌ Некорректный запрос. Попробуйте переформулировать."

    # 403 - Нет доступа
    if "403" in error_text or "permission_denied" in err:
        return "🔒 Нет доступа к API. Обратитесь к администратору."

    # 404 - Не найдено
    if "404" in error_text or "not_found" in err:
        return "🔍 Запрашиваемый ресурс не найден."

    # 500 - Внутренняя ошибка сервера
    if "500" in error_text or "internal" in err:
        return "🔧 Внутренняя ошибка сервера. Попробуйте позже."

    # Fallback
    if is_voice:
        return "🎤 Не удалось обработать голосовое сообщение. Попробуйте ещё раз."
    return "⚠️ Произошла ошибка. Попробуйте ещё раз."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    if not update.message or not update.effective_user or not context.bot_data or not update.effective_chat:
        return

    config: Config = context.bot_data["config"]
    user = update.effective_user

    if config.telegram_whitelist and user.id not in config.telegram_whitelist:
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        logger.warning(f"Unauthorized access attempt: {user.id} ({user.username})")
        return

    # Проверка доступности голоса
    voice_available = bool(config.audio_model)

    # Формируем приветственное сообщение с описанием функционала
    welcome_message = (
        f"👋 Привет, {user.first_name}!\n\n"
        f"Я <b>AI-ассистент TEMPO</b> — ваш корпоративный помощник.\n\n"
        "🎯 <b>Мои возможности:</b>\n\n"
        "📚 <b>Поиск по базе знаний</b>\n"
        "   • Быстрый поиск информации в корпоративных документах\n"
        "   • Точные ответы с учетом контекста\n"
        "💬 <b>Текстовые запросы</b>\n"
        "   • Задавайте вопросы обычным текстом\n"
        "   • Получайте развернутые ответы\n"
    )

    # Добавляем информацию о голосовых сообщениях
    if voice_available:
        welcome_message += (
            "🎤 <b>Голосовые сообщения</b>\n"
            "   • Отправляйте голосовые запросы\n"
            "   • Получайте голосовые ответы\n\n"
        )
    else:
        welcome_message += "🎤 <b>Голосовые сообщения: недоступны</b>\n   • Используйте текстовые запросы\n\n"

    welcome_message += (
        "🧠 <b>История диалогов</b>\n"
        "   • Бот помнит контекст разговора\n"
        "   • Можно ссылаться на предыдущие вопросы\n"
        "   • Используйте /clear для очистки истории\n\n"
        "⚙️ <b>Доступные команды:</b>\n"
        "   /start - показать это сообщение\n"
        "   /menu  - открыть главное меню настроек\n"
        "   /clear - очистить историю разговора\n"
        "   /voice - включить/отключить голосовые ответы\n\n"
        "💡 <b>Примеры запросов:</b>\n"
        '   • "Найди информацию о корпоративной базе отдыха"\n'
        '   • "Найди контактный номер сотрудника ..."\n'
        "Задавайте вопросы — я готов помочь! 🚀"
    )

    try:
        assistant: AssistantService = context.bot_data["assistant"]
        if assistant.chat_history:
            session_id = str(user.id)
            user_company = await assistant.chat_history.get_user_company(session_id)
            if not user_company:
                await update.message.reply_text(
                    "👋 Добро пожаловать! Пожалуйста, выберите ваше предприятие для получения максимально точных ответов:",
                    reply_markup=_get_company_keyboard()
                )
                return
            else:
                user_company_name = COMPANIES.get(user_company, user_company)
                welcome_message = f"✅ Выбрано предприятие: <b>{user_company_name}</b>\n<i>(Для изменения используйте /change_company)</i>\n\n" + welcome_message
    except Exception as e:
        logger.error(f"Error getting user company: {e}")

    await update.message.reply_text(welcome_message, parse_mode="HTML")

async def change_company_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для смены текущего предприятия"""
    if not update.message or not update.effective_user:
        return
        
    await update.message.reply_text(
        "🏭 Выберите ваше предприятие:",
        reply_markup=_get_company_keyboard()
    )


async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /menu - открывает интерактивное меню"""
    if not update.message or not update.effective_user:
        return

    assistant: AssistantService = context.bot_data["assistant"]
    user_id = str(update.effective_user.id)
    
    voice_enabled = True
    company_name = None
    
    if assistant.chat_history:
        voice_enabled = await assistant.chat_history.get_voice_mode(user_id)
        company_id = await assistant.chat_history.get_user_company(user_id)
        if company_id:
            company_name = COMPANIES.get(company_id, company_id)

    text = _get_menu_text(voice_enabled, company_name)
    reply_markup = _get_main_menu_keyboard(voice_enabled)
    
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")


async def toggle_voice_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переключение режима голосовых ответов: голос <-> текст."""
    if not update.message or not update.effective_user:
        return

    user = update.effective_user
    assistant: AssistantService = context.bot_data.get("assistant")

    if not assistant or not assistant.chat_history:
        await update.message.reply_text("⚠️ База данных недоступна.")
        return

    session_id = str(user.id)
    current = await assistant.chat_history.get_voice_mode(session_id)
    new_mode = not current
    await assistant.chat_history.set_voice_mode(session_id, new_mode)

    if new_mode:
        icon, label = "🎙", "ВКЛЮЧЕНЫ — отвечаю голосом на голосовые сообщения"
    else:
        icon, label = "💬", "ОТКЛЮЧЕНЫ — отвечаю текстом на голосовые сообщения"

    await update.message.reply_text(
        f"{icon} <b>Голосовые ответы: {label}</b>",
        parse_mode="HTML"
    )

async def company_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на инлайн кнопки меню и выбора предприятия"""
    query = update.callback_query
    await query.answer()
    
    if not query.data:
        return
        
    user_id = str(update.effective_user.id)
    assistant: AssistantService = context.bot_data["assistant"]
    
    # ─── 1. Возврат в главное меню ──────────────────────────────────────────
    if query.data == "menu_main":
        voice_enabled = await assistant.chat_history.get_voice_mode(user_id)
        company_id = await assistant.chat_history.get_user_company(user_id)
        company_name = COMPANIES.get(company_id, company_id) if company_id else None
        
        await query.edit_message_text(
            _get_menu_text(voice_enabled, company_name),
            reply_markup=_get_main_menu_keyboard(voice_enabled),
            parse_mode="HTML"
        )
        
    # ─── 2. Переход к выбору компании ──────────────────────────────────────
    elif query.data == "menu_company":
        await query.edit_message_text(
            "🏭 <b>Выберите ваше предприятие:</b>",
            reply_markup=_get_company_menu_keyboard(),
            parse_mode="HTML"
        )
        
    # ─── 3. Переключение голосового режима ──────────────────────────────────
    elif query.data == "menu_voice":
        current = await assistant.chat_history.get_voice_mode(user_id)
        new_mode = not current
        await assistant.chat_history.set_voice_mode(user_id, new_mode)
        
        company_id = await assistant.chat_history.get_user_company(user_id)
        company_name = COMPANIES.get(company_id, company_id) if company_id else None
        
        await query.edit_message_text(
            _get_menu_text(new_mode, company_name),
            reply_markup=_get_main_menu_keyboard(new_mode),
            parse_mode="HTML"
        )
        
    # ─── 4. Очистка истории ──────────────────────────────────────────────────
    elif query.data == "menu_clear":
        await clear_history(update, context)
        
    # ─── 5. Установка выбранной компании ─────────────────────────────────────
    elif query.data.startswith("company_"):
        company_id = query.data.split("company_")[1]
        company_name = COMPANIES.get(company_id, company_id)
        
        try:
            if assistant.chat_history:
                await assistant.chat_history.set_user_company(user_id, company_id)
                # После выбора компании возвращаемся в главное меню с подтверждением
                voice_enabled = await assistant.chat_history.get_voice_mode(user_id)
                
                success_text = (
                    f"✅ Предприятие <b>{company_name}</b> успешно установлено!\n\n"
                    f"{_get_menu_text(voice_enabled, company_name)}"
                )
                
                await query.edit_message_text(
                    success_text,
                    reply_markup=_get_main_menu_keyboard(voice_enabled),
                    parse_mode="HTML"
                )
            else:
                await query.edit_message_text("❌ Ошибка: модуль истории не подключен.")
        except Exception as e:
            logger.error(f"Error setting company: {e}")
            await query.edit_message_text("❌ Произошла ошибка при сохранении настроек.")


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clear - очистка истории разговора"""
    user_id = str(update.effective_user.id)
    assistant: AssistantService = context.bot_data["assistant"]
    
    if assistant.chat_history:
        try:
            await assistant.chat_history.clear_history(user_id, clear_summary=True)
            await assistant.orchestrator.clear_memory(user_id)
            
            text = "✅ <b>История разговора очищена.</b>"
            
            if update.callback_query:
                # Получаем текущие настройки для отображения в меню
                voice_enabled = await assistant.chat_history.get_voice_mode(user_id)
                company_id = await assistant.chat_history.get_user_company(user_id)
                company_name = COMPANIES.get(company_id, company_id) if company_id else None
                
                success_text = f"✅ <b>История разговора очищена!</b>\n\n{_get_menu_text(voice_enabled, company_name)}"
                await update.callback_query.edit_message_text(
                    success_text, 
                    reply_markup=_get_main_menu_keyboard(voice_enabled),
                    parse_mode="HTML"
                )
            else:
                await update.effective_message.reply_text("✅ <b>История разговора очищена.</b>", parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            err_text = "❌ Ошибка при очистке истории."
            if update.callback_query:
                await update.callback_query.edit_message_text(err_text)
            else:
                await update.effective_message.reply_text(err_text)
    else:
        err_text = "ℹ️ История разговоров отключена в настройках."
        if update.callback_query:
            await update.callback_query.edit_message_text(err_text)
        else:
            await update.effective_message.reply_text(err_text)


async def reload_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /reload - горячая перезагрузка конфигурации"""
    if not update.message or not update.effective_user or not context.bot_data or not update.effective_chat:
        return

    config: Config = context.bot_data["config"]
    user = update.effective_user

    if config.telegram_whitelist and user.id not in config.telegram_whitelist:
        await update.message.reply_text("⛔ Доступ запрещен.")
        logger.warning(f"Unauthorized reload attempt: {user.id} ({user.username})")
        return

    await update.message.reply_text("🔄 Перезагрузка конфигурации...")

    try:
        # 1. Перезагружаем Config (быстро, не блокирует)
        config.reload()
        logger.info(f"Config reloaded by user {user.id} ({user.username})")

        # 2. Обновляем ClientManager в executor (может блокировать)
        import asyncio

        from src.core.clients import ClientManager

        client_manager = ClientManager.get_instance()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, client_manager.reload_clients)

        # 3. Обновляем AssistantService (быстро)
        assistant: AssistantService = context.bot_data["assistant"]
        assistant.reload_services()

        # 4. Формируем отчет
        await update.message.reply_text(
            "✅ Конфигурация перезагружена!\n\n"
            f"🔑 API Key: {config.gemini_api_key[:10]}...\n"
            f"📝 Текстовая модель: {config.text_model}\n"
            f"🎤 Голосовая модель: {config.audio_model}\n"
            f"🔢 Embedding модель: {config.embedding_model}\n"
            f"📊 Text API версия: {config.text_api_version or 'авто'}\n"
            f"📊 Live API версия: {config.live_api_version}\n"
            f"📊 Embedding API версия: {config.embedding_api_version}"
        )

        logger.info("Configuration reloaded successfully")

    except Exception as e:
        logger.error(f"Error reloading config: {e}")
        await update.message.reply_text(f"❌ Ошибка при перезагрузке конфигурации:\n{str(e)}")


async def send_document(update: Update, document_rule):
    """
    Отправляет документ пользователю.
    
    Args:
        update: Telegram Update объект
        document_rule: DocumentRule объект с информацией о документе
    """
    import os
    
    # Определяем базовый путь
    base_paths = [
        "e:/Old/bots/Worker/iTEMPO/iTEMPO_test",
        "/home/administrator/iTEMPO_test"
    ]
    
    document_path = document_rule.document_path
    file_found = False
    
    for base_path in base_paths:
        full_path = os.path.join(base_path, document_path.replace('/', os.sep))
        if os.path.exists(full_path):
            file_found = True
            break
    
    if not file_found:
        logger.warning(f"Document not found: {document_path}")
        await update.message.reply_text(f"❌ Документ не найден: {document_rule.description}")
        return
    
    try:
        with open(full_path, 'rb') as document_file:
            file_type = document_rule.file_type
            if file_type == "auto":
                # Определяем тип по расширению
                extension = os.path.splitext(full_path)[1].lower()
                if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    file_type = "image"
                elif extension == '.pdf':
                    file_type = "pdf"
                elif extension in ['.docx', '.doc']:
                    file_type = "docx"
                else:
                    file_type = "other"
            
            filename = os.path.basename(full_path)
            
            if file_type == "image":
                await update.message.reply_photo(
                    photo=document_file,
                    caption=document_rule.description
                )
            else:
                # Для PDF, DOCX и других файлов
                await update.message.reply_document(
                    document=document_file,
                    filename=filename,
                    caption=document_rule.description
                )
                
    except Exception as e:
        logger.error(f"Error sending document {document_path}: {e}")
        await update.message.reply_text(f"❌ Не удалось отправить документ: {document_rule.description}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений (без стриминга для стабильности)"""
    if not update.message or not update.effective_user or not context.bot_data or not update.effective_chat:
        return

    config: Config = context.bot_data["config"]
    user = update.effective_user

    if config.telegram_whitelist and user.id not in config.telegram_whitelist:
        await update.message.reply_text("⛔ Доступ запрещен.")
        return

    query = update.message.text
    if not query:
        return

    assistant: AssistantService = context.bot_data["assistant"]
    session_id = str(user.id)

    # Запрашиваем компанию перед продолжением общения
    if assistant.chat_history:
        # Обновляем время последней активности
        try:
            await assistant.chat_history.update_last_activity(session_id, "telegram")
        except Exception:
            pass

        # Проверяем блокировку
        try:
            if await assistant.chat_history.is_user_blocked(session_id):
                await update.message.reply_text("⛔ Доступ ограничен.")
                return
        except Exception:
            pass

        user_company = await assistant.chat_history.get_user_company(session_id)
        if not user_company:
            await update.message.reply_text(
                "🏭 Пожалуйста, выберите ваше предприятие перед началом работы:",
                reply_markup=_get_company_keyboard()
            )
            return


    # Отправляем "печатает..."
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Отправляем начальное сообщение
    status_message = await update.message.reply_text("🔄 Готовлю ответ...", parse_mode="HTML")
    
    # Запускаем задачу для обновления статуса
    async def update_status_periodically():
        """Обновляет сообщение со статусом каждые 8 секунд"""
        status_texts = [
            "🔄 Готовлю ответ...",
            "⏳ Анализирую запрос...",
            "🔍 Поиск информации...",
            "📝 Формирую ответ..."
        ]
        counter = 0
        while True:
            try:
                await asyncio.sleep(8)
                counter += 1
                new_text = status_texts[counter % len(status_texts)]
                await status_message.edit_text(new_text, parse_mode="HTML")
            except Exception:
                break  # Если сообщение было удалено или ошибка редактирования
    
    status_task = asyncio.create_task(update_status_periodically())

    try:
        assistant: AssistantService = context.bot_data["assistant"]
        session_id = str(user.id)

        # Выполняем ОДИН запрос
        result = await assistant.process_text_query(
            query,
            limit=15,
            session_id=session_id,
            platform="telegram",
            user_name=user.first_name,
            user_company=user_company
        )
        
        answer = result.get("answer", "")
        documents_to_send = result.get("documents_to_send", [])
        
        # Останавливаем обновление статуса
        status_task.cancel()
        
        try:
            await status_message.delete()
        except Exception:
            pass  # Сообщение уже могло быть удалено

        # Otpavlyaem otvet
        if answer:
            try:
                # Extract images from answer and send separately
                import re
                
                # Find all image references in markdown format
                image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                images = re.findall(image_pattern, answer)
                
                # Remove image markdown from text
                clean_answer = re.sub(image_pattern, '', answer).strip()
                
                # Send text answer first
                if clean_answer:
                    await update.message.reply_text(clean_answer, parse_mode="HTML")
                
                # Send images from answer (old method)
                for alt_text, image_path in images:
                    try:
                        # Check if it's a local file
                        if image_path.startswith('data/'):
                            import os
                            # Try to detect the correct base path
                            if os.path.exists('/home/administrator/iTEMPO_test'):
                                base_path = '/home/administrator/iTEMPO_test'
                            else:
                                base_path = "e:/Old/bots/Worker/iTEMPO/iTEMPO_test"
                            full_path = os.path.join(base_path, image_path.replace('/', os.sep))
                            with open(full_path, 'rb') as photo_file:
                                await update.message.reply_photo(photo_file, caption=alt_text)
                    except Exception as img_error:
                        logger.error(f"Failed to send image {image_path}: {img_error}")
                        # If image sending fails, send the markdown as text
                        await update.message.reply_text(f"![{alt_text}]({image_path})", parse_mode="HTML")
                
                # Send documents from DocumentSender (new method)
                for document_rule in documents_to_send:
                    await send_document(update, document_rule)
                    
            except Exception as e:
                logger.warning(f"Error sending message with HTML: {e}")
                # Если HTML все-таки сломан, шлем чистый текст
                await update.message.reply_text(answer)
                
                # Still try to send documents
                for document_rule in documents_to_send:
                    await send_document(update, document_rule)
        else:
            await update.message.reply_text("❌ Извините, не удалось получить ответ. Попробуйте другой вопрос.", parse_mode="HTML")

    except AssistantError as e:
        logger.error(f"Assistant error: {e}")
        # Останавливаем обновление статуса
        status_task.cancel()
        
        try:
            await status_message.edit_text(str(e), parse_mode="HTML")
        except Exception:
            # Если не удалось отредактировать, отправляем новое сообщение
            await update.message.reply_text(str(e), parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        # Останавливаем обновление статуса
        status_task.cancel()
        
        try:
            await status_message.delete()
        except Exception:
            pass
            
        err_str = str(e).lower()
        if "timed out" in err_str or "network error" in err_str or "connection" in err_str:
            err_msg = "⚠️ <b>Ошибка связи с Telegram</b>. Повторите запрос через пару секунд."
        else:
            err_msg = "❌ Произошла внутренняя ошибка при обработке запроса."
        
        try:
            await update.message.reply_text(err_msg, parse_mode="HTML")
        except Exception:
            try:
                await update.message.reply_text(err_msg)
            except Exception:
                pass


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик голосовых сообщений"""
    if not update.message or not update.effective_user or not context.bot_data or not update.effective_chat:
        return

    config: Config = context.bot_data["config"]
    user = update.effective_user

    if config.telegram_whitelist and user.id not in config.telegram_whitelist:
        await update.message.reply_text("⛔ Доступ запрещен.")
        return

    assistant: AssistantService = context.bot_data["assistant"]
    session_id = str(user.id)

    # Проверка доступности голоса
    try:
        assistant_mock = assistant
    except Exception:
        pass

    if assistant.chat_history:
        user_company = await assistant.chat_history.get_user_company(session_id)
        if not user_company:
            await update.message.reply_text(
                "🏭 Пожалуйста, выберите ваше предприятие перед началом работы:",
                reply_markup=_get_company_keyboard()
            )
            return

    # Показываем статус "думаю" и в тексте, и в экшене
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
    
    # Отправляем начальное сообщение
    status_message = await update.message.reply_text("🎤 Обрабатываю голосовое сообщение...", parse_mode="HTML")
    
    # Запускаем задачу для обновления статуса
    async def update_voice_status_periodically():
        """Обновляет сообщение со статусом каждые 5 секунд"""
        status_texts = [
            "🎤 Обрабатываю голосовое сообщение...",
            "🔊 Распознаю речь...",
            "⏳ Анализирую запрос...",
            "📝 Формирую ответ..."
        ]
        counter = 0
        while True:
            try:
                await asyncio.sleep(8)
                counter += 1
                new_text = status_texts[counter % len(status_texts)]
                await status_message.edit_text(new_text, parse_mode="HTML")
            except Exception:
                break  # Если сообщение было удалено или ошибка редактирования
    
    status_task = asyncio.create_task(update_voice_status_periodically())

    try:
        # Скачиваем голосовое сообщение
        voice = update.message.voice
        if not voice:
            status_task.cancel()
            try:
                await status_message.delete()
            except Exception:
                pass
            return
        file = await context.bot.get_file(voice.file_id)
        ogg_bytes = await file.download_as_bytearray()
        raw_audio: bytes = bytes(ogg_bytes)
        session_id = str(user.id)

        # ─── Определяем режим: голос или текст ─────────────────────────────
        use_voice_response = True
        if assistant.chat_history:
            use_voice_response = await assistant.chat_history.get_voice_mode(session_id)

        if use_voice_response:
            # ── Режим ГОЛОС: отвечаем аудио через Live API (Gemini) ──────────
            audio_response, transcript, extracted_links = await assistant.process_voice_query(
                raw_audio,
                use_rag=True,
                limit=10,
                session_id=session_id,
                platform="telegram",
            )
            
            status_task.cancel()
            try:
                await status_message.delete()
            except Exception:
                pass

            await update.message.reply_voice(voice=audio_response)

            # Ссылки и изображения из транскрипта
            if transcript:
                patterns = [
                    r"(https?://\S+)",
                    r"(www\.\S+)",
                ]
                links_in_transcript = []
                for pattern in patterns:
                    found = [lnk.rstrip(".,!?;:)\u2026") for lnk in re.findall(pattern, transcript)]
                    links_in_transcript.extend(found)
                links_in_transcript = [l for l in links_in_transcript if l.startswith(('http://', 'https://'))]
                logger.info(f"Links found in transcript: {links_in_transcript}")
                logger.info(f"Extracted links from RAG: {extracted_links}")
                logger.info(f"Transcript content: {transcript[:200]}...")

                all_links = set(links_in_transcript)
                if isinstance(extracted_links, (set, list)):
                    all_links.update(extracted_links)

                logger.info(f"All links to send: {list(all_links)}")

                image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                images = re.findall(image_pattern, transcript)
                for alt_text, image_path in images:
                    try:
                        if image_path.startswith('data/'):
                            import os
                            base_path = '/home/administrator/iTEMPO_test' if os.path.exists('/home/administrator/iTEMPO_test') else "e:/Old/bots/Worker/iTEMPO/iTEMPO_test"
                            full_path = os.path.join(base_path, image_path.replace('/', os.sep))
                            with open(full_path, 'rb') as photo_file:
                                await update.message.reply_photo(photo_file, caption=alt_text)
                    except Exception as img_error:
                        logger.error(f"Failed to send image {image_path}: {img_error}")

                if all_links:
                    # Ограничиваем количество ссылок до 3, чтобы не перегружать сообщение
                    top_links = all_links[:3]
                    links_list = "\n".join(top_links)
                    await update.message.reply_text(
                        f"-> <b>Link / Route:</b>\n\n{links_list}",
                        parse_mode="HTML",
                        disable_web_page_preview=False,
                    )

        else:
            # ── Режим ТЕКСТ: транскрибируем голос, отдаем текстовому агенту ──
            pcm_data = await assistant.audio_llm._convert_ogg_to_pcm(raw_audio)
            transcript = await assistant.audio_llm.transcribe_audio_from_pcm(pcm_data)

            if not transcript or not transcript.strip():
                status_task.cancel()
                try:
                    await status_message.edit_text("❌ Не удалось распознать речь. Попробуйте ещё раз.")
                except Exception:
                    pass
                return

            logger.info(f"Voice transcribed to text: {transcript!r}")

            # Пишем в статусе что именно распознали
            try:
                await status_message.edit_text(
                    f"📝 Распознано: <i>{transcript}</i>\n⏳ Анализирую...",
                    parse_mode="HTML"
                )
            except Exception:
                pass

            result = await assistant.process_text_query(
                transcript,
                session_id=session_id,
                platform="telegram",
            )

            status_task.cancel()
            try:
                await status_message.delete()
            except Exception:
                pass

            answer = result.get("answer", "")
            documents_to_send = result.get("documents_to_send", [])

            if answer:
                import re as _re
                image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                images = _re.findall(image_pattern, answer)
                clean_answer = _re.sub(image_pattern, '', answer).strip()
                try:
                    if clean_answer:
                        await update.message.reply_text(clean_answer, parse_mode="HTML")
                    for alt_text, image_path in images:
                        try:
                            if image_path.startswith('data/'):
                                import os
                                base_path = '/home/administrator/iTEMPO_test' if os.path.exists('/home/administrator/iTEMPO_test') else "e:/Old/bots/Worker/iTEMPO/iTEMPO_test"
                                full_path = os.path.join(base_path, image_path.replace('/', os.sep))
                                with open(full_path, 'rb') as photo_file:
                                    await update.message.reply_photo(photo_file, caption=alt_text)
                        except Exception as img_error:
                            logger.error(f"Failed to send image: {img_error}")
                    for document_rule in documents_to_send:
                        await send_document(update, document_rule)
                except Exception:
                    await update.message.reply_text(answer)
            else:
                await update.message.reply_text("❌ Не удалось получить ответ. Попробуйте другой вопрос.")

        logger.info(f"Voice response sent to {user.username}")

    except AssistantError as e:
        logger.error(f"Error handling voice: {e}")
        # Останавливаем обновление статуса
        status_task.cancel()
        
        try:
            await status_message.edit_text(str(e), parse_mode="HTML")
        except Exception:
            # Если не удалось отредактировать, отправляем новое сообщение
            await update.message.reply_text(str(e), parse_mode="HTML")
    except Exception as e:
        logger.error(f"Unexpected error handling voice: {e}")
        # Останавливаем обновление статуса
        status_task.cancel()
        
        try:
            await status_message.delete()
        except Exception:
            pass
            
        err_str = str(e).lower()
        if "timed out" in err_str or "network error" in err_str or "connection" in err_str:
            err_msg = "⚠️ <b>Ошибка связи с Telegram (голос)</b>. Повторите запрос через пару секунд."
        else:
            err_msg = "❌ Извините, не удалось обработать голосовое сообщение."
            
        try:
            await update.message.reply_text(err_msg, parse_mode="HTML")
        except Exception:
            try:
                await update.message.reply_text(err_msg)
            except Exception:
                pass


def create_bot_application(config: Config) -> Application:
    """Создание приложения бота с увеличенными таймаутами для серверной среды"""
    defaults = Defaults(tzinfo=pytz.UTC)
    
    return (
        ApplicationBuilder()
        .token(config.telegram_token)
        .defaults(defaults)
        .concurrent_updates(True)
        .connect_timeout(30.0)  # Таймаут подключения к Telegram API
        .read_timeout(60.0)     # Таймаут чтения ответа
        .write_timeout(30.0)     # Таймаут отправки запроса
        .pool_timeout(60.0)       # Таймаут пула соединений
        .connection_pool_size(100) # Увеличенный пул для параллельных запросов
        .get_updates_connection_pool_size(10) # Отдельный пул для получения обновлений
        .build()
    )


def create_telegram_app(config: Config, assistant: AssistantService):
    """
    Создает и настраивает экземпляр Application для Telegram.
    Используется для совместного запуска с другими ботами в одном event loop.
    """
    # Предзагрузка моделей (устраняет задержку на первом запросе)
    from src.core.clients import ClientManager
    client_manager = ClientManager.get_instance(config)
    client_manager.preload_models()

    # Создаем приложение
    application = create_bot_application(config)
    
    # Сохраняем конфиг и ассистента в bot_data
    application.bot_data["config"] = config
    application.bot_data["assistant"] = assistant

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", show_menu))
    application.add_handler(CommandHandler("change_company", change_company_command))
    application.add_handler(CommandHandler("reload", reload_config))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("voice", toggle_voice_mode))
    application.add_handler(CallbackQueryHandler(company_callback, pattern='^(company_|menu_)'))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    return application


def run_bot(
    config: Config,
    update_db: bool = False,
    assistant: Optional[AssistantService] = None,
):
    """
    Запускает бота (блокирующий вызов).
    """
    print("Запуск Telegram бота...")

    # Обновление базы (если нужно)
    if update_db:
        print("--- Обновление базы знаний ---")
        asyncio.run(_update_database(config))
        print("--- База знаний готова ---\n")

    # Используем переданного ассистента или создаем нового
    if assistant is None:
        assistant = AssistantService(config)

    # Инициализация индексов перед запуском поллинга
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Если петля уже запущена (редкий случай для run_bot)
        loop.create_task(assistant.initialize())
    else:
        loop.run_until_complete(assistant.initialize())

    application = create_telegram_app(config, assistant)

    # Если бот запущен в отдельном потоке (как в run_server.py),
    # нужно отключить обработку сигналов, иначе asyncio упадет.
    import threading

    is_main_thread = threading.current_thread() is threading.main_thread()
    if not is_main_thread:
        # Увеличиваем количество повторных попыток для серверной среды
        application.run_polling(stop_signals=None, bootstrap_retries=10)
    else:
        application.run_polling(bootstrap_retries=10)


async def _update_database(config: Config):
    """Вспомогательная функция для обновления базы"""
    processor = DocumentProcessor(config)
    chunks = processor.prepare_chunks()

    embedding_service = EmbeddingService(config)
    await embedding_service.update_database(chunks)


if __name__ == "__main__":
    from src.core.config import Config

    try:
        config = Config.from_env()
    except Exception as e:
        print(f"ОШИБКА: {e}")
        print("Скопируйте env.example в .env и заполните токены.")
        raise SystemExit(1) from e

    # Автоматическое обновление базы
    run_bot(config, update_db=True)

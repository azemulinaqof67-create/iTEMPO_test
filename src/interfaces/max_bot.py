"""
MAX мессенджер — интерфейс бота.

Работает параллельно с Telegram через тот же AssistantService.
Использует Long Polling (без webhook) через прямые HTTP-запросы к MAX API.

MAX API документация: https://dev.max.ru/docs-api
Base URL: https://platform-api.max.ru
Auth: заголовок Authorization: <token>
"""

import asyncio
import logging
import re
import json
from typing import Any, Dict, List, Optional

import aiohttp

from src.assistant.assistant import AssistantService
from src.core.config import Config
from src.core.exceptions import AssistantError

logger = logging.getLogger(__name__)

# ── Константы ──────────────────────────────────────────────────────────────
MAX_API_BASE = "https://platform-api.max.ru"
MAX_POLLING_TIMEOUT = 30       # секунды ожидания long polling
MAX_MESSAGE_MAX_LEN = 4000     # максимум символов в одном сообщении MAX
MAX_RETRY_DELAY = 5            # секунды между повторными попытками при ошибке
MAX_MAX_RETRIES = 5            # максимум повторов при сбоях сети

from src.core.constants import COMPANIES


# ── HTTP-клиент MAX API ────────────────────────────────────────────────────

class MaxBotClient:
    """
    Низкоуровневый асинхронный HTTP-клиент для MAX Bot API.

    Все методы автоматически добавляют заголовок Authorization.
    """

    def __init__(self, token: str):
        self.token = token
        self.base_url = MAX_API_BASE
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": self.token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    async def close(self):
        """Закрывает HTTP-сессию."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Выполняет GET-запрос к MAX API."""
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    text = await resp.text()
                    logger.error(f"MAX API GET {path} → {resp.status}: {text[:200]}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"MAX API GET {path} сетевая ошибка: {e}")
            return None

    async def _post(self, path: str, payload: Dict) -> Optional[Dict]:
        """Выполняет POST-запрос к MAX API."""
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                else:
                    text = await resp.text()
                    logger.error(f"MAX API POST {path} → {resp.status}: {text[:200]}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"MAX API POST {path} сетевая ошибка: {e}")
            return None

    async def _delete(self, path: str) -> bool:
        """Выполняет DELETE-запрос к MAX API."""
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.delete(url) as resp:
                if resp.status in (200, 204):
                    return True
                else:
                    text = await resp.text()
                    logger.error(f"MAX API DELETE {path} → {resp.status}: {text[:200]}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"MAX API DELETE {path} сетевая ошибка: {e}")
            return False

    # ── Публичные методы API ───────────────────────────────────────────────

    async def get_me(self) -> Optional[Dict]:
        """Получить информацию о боте (проверка токена)."""
        return await self._get("/me")

    async def set_webhook(self, url: str, update_types: List[str] = None, secret: str = None) -> bool:
        """
        Метод настраивает доставку событий бота через Webhook.
        При активной подписке Long Polling не работает.
        """
        payload = {
            "url": url,
            "update_types": update_types or ["message_created", "bot_started", "message_callback"],
        }
        if secret:
            payload["secret"] = secret
        
        logger.info(f"MAX API: Setting webhook to {url}...")
        res = await self._post("/subscriptions", payload)
        return res is not None and res.get("success", False)

    async def delete_webhook(self, url: Optional[str] = None) -> bool:
        """Удаляет подписку на вебхук. Если url не указан, удаляет ВСЕ активные подписки."""
        logger.info("MAX API: Deleting webhook subscription...")
        if url:
            from urllib.parse import quote
            return await self._delete(f"/subscriptions?url={quote(url)}")
        
        # Если URL не передан, получаем список всех подписок и удаляем каждую
        try:
            logger.info("MAX API: Fetching active subscriptions for cleanup...")
            subs_data = await self._get("/subscriptions")
            if subs_data and "subscriptions" in subs_data:
                success = True
                for sub in subs_data["subscriptions"]:
                    sub_url = sub.get("url")
                    if sub_url:
                        from urllib.parse import quote
                        logger.info(f"MAX API: Deleting subscription for {sub_url}...")
                        res = await self._delete(f"/subscriptions?url={quote(sub_url)}")
                        if not res:
                            success = False
                return success
        except Exception as e:
            logger.error(f"MAX API: Error deleting webhooks: {e}")
        return await self._delete("/subscriptions")

    async def get_updates(self, marker: Optional[int] = None) -> Optional[Dict]:
        """
        Long Polling: получить новые события.

        Args:
            marker: ID последнего обработанного события (для пагинации)

        Returns:
            Словарь с полями 'updates' (список) и 'marker' (следующий маркер)
        """
        params: Dict[str, Any] = {
            "timeout": MAX_POLLING_TIMEOUT,
        }
        if marker is not None:
            params["marker"] = marker
        return await self._get("/updates", params=params)

    async def send_message(
        self,
        chat_id: int,
        text: str,
        keyboard: Optional[List[List[Dict]]] = None,
        fmt: str = "html",
        attachment_token: Optional[str] = None,
        attachment_type: str = "audio",
        attachment_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Отправить текстовое сообщение в чат.

        Args:
            chat_id: ID чата (из события)
            text: Текст сообщения (поддерживает HTML)
            keyboard: Матрица кнопок [[{type, text, payload}, ...], ...]
            fmt: Формат текста — 'html' или 'markdown'
            attachment_token: Токен загруженного файла (из /uploads)
            attachment_type: Тип вложения (по умолчанию audio)
        """
        payload: Dict[str, Any] = {
            "text": text,
            "format": fmt,
        }

        attachments = []
        
        if keyboard:
            attachments.append({
                "type": "inline_keyboard",
                "payload": {
                    "buttons": keyboard,
                },
            })

        if attachment_token:
            att_payload = {"token": attachment_token}
            if attachment_id:
                try:
                    val = int(attachment_id)
                    att_payload["id"] = val
                    att_payload["audio_id"] = val
                except (ValueError, TypeError):
                    att_payload["id"] = attachment_id
            
            # Критически важное поле для некоторых версий API, чтобы пометить аудио как голос
            if attachment_type == "audio":
                att_payload["kind"] = "voice"
                
            attachments.append({
                "type": attachment_type,
                "payload": att_payload
            })

        if attachments:
            payload["attachments"] = attachments

        logger.info(f"MAX API: отправка сообщения в {chat_id}. Payload: {json.dumps(payload, ensure_ascii=False)}")
        url = f"{self.base_url}/messages?chat_id={chat_id}"
        session = await self._get_session()
        
        # Уменьшаем до 10 попыток (30 секунд), так как текст уже отправлен
        for attempt in range(10):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()
                    
                    text = await resp.text()
                    # Если CDN еще не обработал аудио (ошибка attachment.not.ready)
                    if resp.status == 400 and "attachment.not.ready" in text:
                        logger.info(f"MAX: Вложение еще не готово (попытка {attempt+1}/10)...")
                        await asyncio.sleep(3.0)
                        continue
                        
                    logger.error(f"MAX API POST /messages?chat_id={chat_id} → {resp.status}: {text}")
                    return None
            except Exception as e:
                logger.error(f"Сетевая ошибка при отправке сообщения MAX: {e}")
                return None
                
        logger.error("MAX: Превышен лимит попыток ожидания готовности вложения.")
        return None

    async def answer_callback(self, callback_id: str) -> Optional[Dict]:
        """
        Подтвердить нажатие inline-кнопки (чтобы убрать "загрузку" на кнопке).

        Args:
            callback_id: ID из события message_callback
        """
        return await self._post(
            f"/answers/{callback_id}",
            {"type": "callback"},
        )

    async def send_action(self, chat_id: int, action: str = "typing_on") -> Optional[Dict]:
        """
        Отправить действие (например, "печатает...").

        Args:
            action: 'typing_on' или 'sending_photo'
        """
        return await self._post(f"/chats/{chat_id}/actions", {"action": action})

    async def delete_message(self, chat_id: int, mid: str) -> bool:
        """
        Удалить сообщение.
        """
        session = await self._get_session()
        # В MAX API для удаления нужно передавать message_id
        # Пробуем передать и в URL, и в теле (некоторые версии API требуют разного)
        url = f"{self.base_url}/messages?message_id={mid}&chat_id={chat_id}"
        payload = {"message_id": mid}
        try:
            async with session.delete(url, json=payload) as resp:
                if resp.status in (200, 204):
                    logger.debug(f"MAX: Сообщение {mid} успешно удалено.")
                    return True
                text = await resp.text()
                logger.warning(f"MAX API DELETE {url} → {resp.status}: {text}")
                return False
        except Exception as e:
            logger.error(f"Ошибка при удалении сообщения MAX: {e}")
            return False

    async def edit_message(self, chat_id: int, mid: str, text: str) -> bool:
        """
        Редактировать существующее сообщение.
        """
        session = await self._get_session()
        # В MAX API для редактирования (обновления) используется PUT
        url = f"{self.base_url}/messages?message_id={mid}&chat_id={chat_id}"
        payload = {
            "message_id": mid,
            "text": text,
            "format": "html"
        }
        try:
            async with session.put(url, json=payload) as resp:
                if resp.status in (200, 204):
                    return True
                text_resp = await resp.text()
                logger.warning(f"MAX API PUT {url} → {resp.status}: {text_resp}")
                return False
        except Exception as e:
            logger.error(f"Ошибка при редактировании сообщения MAX: {e}")
            return False

    async def upload_file(self, file_bytes: bytes, filename: str = "audio.ogg", upload_type: str = "audio", kind: str = "voice") -> tuple[Optional[str], Optional[str]]:
        """
        Загружает файл в хранилище MAX и возвращает (token, file_id).
        upload_type может быть 'audio', 'image', 'file'.
        """
        session = await self._get_session()
        
        # Шаг 1: Получение URL для загрузки
        url = f"{self.base_url}/uploads?type={upload_type}"
        if kind:
            url += f"&kind={kind}"
        
        try:
            async with session.post(url) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Ошибка получения URL загрузки: {resp.status} - {text}")
                    return None, None
                
                data = await resp.json()
                upload_url = data.get("url")
                if not upload_url:
                    logger.error("URL загрузки не найден в ответе API")
                    return None, None

            # Шаг 2: Фактическая загрузка файла в хранилище (Storage)
            # КРИТИЧНО: Используем requests для формирования эталонного multipart-запроса
            if upload_type == "image":
                import requests
                import asyncio
                
                ext = filename.lower().split('.')[-1]
                if ext not in ['png', 'jpg', 'jpeg', 'gif']:
                    ext = 'png'
                mime = f"image/{ext}"
                
                def sync_upload():
                    files = {'v1': (f"image.{ext}", file_bytes, mime)}
                    return requests.post(upload_url, files=files)
                
                # Запускаем в потоке, чтобы не блокировать event loop
                resp_sync = await asyncio.to_thread(sync_upload)
                data = resp_sync.json()
                logger.info(f"ОТВЕТ MEDIA SERVER (Step 2 - Requests/v1): {resp_sync.status_code} - {data}")
                
                # Имитируем объект ответа для дальнейшей логики
                class MockResp:
                    def __init__(self, status): self.status = status
                resp = MockResp(resp_sync.status_code)
            else:
                # Для остальных оставляем aiohttp
                form = aiohttp.FormData()
                form.add_field("file", file_bytes, filename=filename, content_type="application/octet-stream")
                async with session.post(upload_url, data=form) as resp:
                    data = await resp.json()
                    logger.info(f"ОТВЕТ MEDIA SERVER (Step 2 - Multipart): {resp.status} - {data}")
            
            # Если сервер вернул 200, но в JSON ошибка
            if data.get("error_code") or data.get("error_data"):
                logger.error(f"Медиа-сервер отклонил загрузку: {data}")
                return None, None

            if resp.status == 200:
                from urllib.parse import unquote
                
                # Шаг 3: Извлекаем НАСТОЯЩИЙ токен из ответа медиа-сервера
                token = data.get("token")
                if token:
                    token = unquote(token)
                
                # Если ответ в формате {"photos": {"hash": {"token": "..."}}}
                if not token and "photos" in data:
                    photos = data["photos"]
                    if isinstance(photos, dict) and photos:
                        first_photo = next(iter(photos.values()))
                        if isinstance(first_photo, dict):
                            token = first_photo.get("token")
                            if token:
                                token = unquote(token)
                
                # ID для изображений не требуется, но для файлов/аудио извлекаем
                file_id = data.get("id") or data.get("file_id")
                
                if not token:
                    # В крайнем случае пробуем найти в URL (для обратной совместимости)
                    if "token=" in upload_url:
                        token = unquote(upload_url.split("token=")[1].split("&")[0])
                    elif "apiToken=" in upload_url:
                        token = unquote(upload_url.split("apiToken=")[1].split("&")[0])
                
                logger.info(f"Файл успешно загружен. Тип: {upload_type}, Токен получен: {bool(token)}")
                return token, str(file_id) if file_id else None
            
            logger.error(f"Ошибка при загрузке байтов в Storage: {resp.status} - {data}")
            return None, None
                    
        except Exception as e:
            logger.error(f"Сетевая ошибка при загрузке файла в MAX API: {e}")
            return None, None

    async def register_subscription(self) -> bool:
        """
        Регистрирует подписку на типы событий для Long Polling.

        Без этого MAX API не доставляет события боту.
        Нужно вызывать один раз при старте.

        Returns:
            True если подписка успешно зарегистрирована, False при ошибке.
        """
        payload = {
            "update_types": [
                "bot_started",        # Пользователь запустил бота
                "message_created",    # Входящее сообщение
                "message_callback",   # Нажатие inline-кнопки
                "message_edited",     # Редактирование сообщения
            ]
        }
        result = await self._post("/subscriptions", payload)
        if result is not None:
            logger.info(f"MAX: подписка на события зарегистрирована: {payload['update_types']}")
            return True
        else:
            logger.error("MAX: не удалось зарегистрировать подписку на события")
            return False

    async def set_bot_menu(self) -> bool:
        """
        Устанавливает системное меню бота (кнопка [ ☰ ] у поля ввода).
        """
        payload = {
            "menu": [
                {"text": "🚀 Старт", "payload": "/start"},
                {"text": "🏭 Изменить предприятие", "payload": "/change_company"},
                {"text": "🧹 Очистить историю", "payload": "/clear"}
            ]
        }
        # В некоторых версиях MAX API используется /me/menu или /bot/menu
        result = await self._post("/me/menu", payload)
        if result is not None:
            logger.info("MAX: системное меню успешно установлено")
            return True
        else:
            # Пробуем альтернативный путь если первый не сработал
            result = await self._post("/menu", payload)
            if result is not None:
                logger.info("MAX: системное меню успешно установлено (через /menu)")
                return True
            logger.error("MAX: не удалось установить системное меню")
            return False


# ── Вспомогательные функции ────────────────────────────────────────────────

def _get_main_menu_keyboard() -> List[List[Dict]]:
    """Формирует клавиатуру главного меню."""
    return [
        [{"type": "callback", "text": "🏭 Изменить предприятие", "payload": "/change_company"}],
        [{"type": "callback", "text": "🧹 Очистить историю", "payload": "/clear"}]
    ]


def _get_company_keyboard() -> List[List[Dict]]:
    """Формирует inline-клавиатуру выбора предприятия в формате MAX API."""
    buttons = []
    for key, name in COMPANIES.items():
        buttons.append([
            {
                "type": "callback",
                "text": name,
                "payload": f"company_{key}",
            }
        ])
    return buttons


def _split_long_message(text: str, max_len: int = MAX_MESSAGE_MAX_LEN) -> List[str]:
    """
    Разбивает длинный текст на части, не обрывая слова.
    Нужно если ответ ассистента превышает лимит MAX API.
    """
    if len(text) <= max_len:
        return [text]

    parts = []
    while len(text) > max_len:
        # Ищем последний пробел/перевод строки до границы
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len  # Жёсткое разрезание если нет пробелов

        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        parts.append(text)
    return parts


def _get_session_id(user_id: int) -> str:
    """Формирует уникальный session_id для MAX пользователей."""
    return f"max_{user_id}"


# ── Обработчики событий ────────────────────────────────────────────────────

async def _handle_bot_started(
    event: Dict,
    client: MaxBotClient,
    assistant: AssistantService,
):
    """
    Обработчик события bot_started (пользователь запустил бота или написал /start).
    Показывает клавиатуру выбора предприятия если оно ещё не выбрано.
    """
    chat_id: int = event.get("chat", {}).get("chat_id") or event.get("chat_id")
    user: Dict = event.get("user", {})
    user_id: int = user.get("user_id", 0)
    user_name: str = user.get("name", "")

    session_id = _get_session_id(user_id)
    logger.info(f"🔵 MAX [bot_started]: user_id={user_id}, chat_id={chat_id}, session_id={session_id}")

    try:
        if assistant.chat_history:
            user_company = await assistant.chat_history.get_user_company(session_id)
            if not user_company:
                # Предприятие не выбрано — показываем клавиатуру
                await client.send_message(
                    chat_id=chat_id,
                    text=(
                        f"👋 Привет, <b>{user_name}</b>!\n\n"
                        "Я <b>AI-ассистент TEMPO</b> — ваш корпоративный помощник.\n\n"
                        "🎯 <b>Мои возможности:</b>\n\n"
                        "📚 <b>Поиск по базе знаний</b>\n"
                        "   • Быстрый поиск информации в корпоративных документах\n"
                        "   • Точные ответы с учетом контекста\n"
                        "💬 <b>Текстовые запросы</b>\n"
                        "   • Задавайте вопросы обычным текстом\n"
                        "   • Получайте развернутые ответы\n"
                        "🎤 <b>Голосовые сообщения</b>\n"
                        "   • Отправляйте голосовые запросы\n"
                        "   • Получайте текстовые ответы\n\n"
                        "🏭 Пожалуйста, выберите ваше предприятие для получения максимально точных ответов, "
                        "для изменения предприятия выполните команду /menu:"
                    ),
                    keyboard=_get_company_keyboard(),
                )
                return
            else:
                company_name = COMPANIES.get(user_company, user_company)
                await client.send_message(
                    chat_id=chat_id,
                    text=(
                        f"👋 Привет, <b>{user_name}</b>!\n\n"
                        f"✅ Выбрано предприятие: <b>{company_name}</b> (Для изменения напишите /menu или используйте кнопки ниже)\n\n"
                        "Я <b>AI-ассистент TEMPO</b> — ваш корпоративный помощник.\n\n"
                        "🎯 <b>Мои возможности:</b>\n\n"
                        "📚 <b>Поиск по базе знаний</b>\n"
                        "   • Быстрый поиск информации в корпоративных документах\n"
                        "   • Точные ответы с учетом контекста\n"
                        "💬 <b>Текстовые запросы</b>\n"
                        "   • Задавайте вопросы обычным текстом\n"
                        "   • Получайте развернутые ответы\n"
                        "🎤 <b>Голосовые сообщения</b>\n"
                        "   • Отправляйте голосовые запросы\n\n"
                        "Задавайте вопросы — я готов помочь! 🚀"
                    ),
                    keyboard=_get_main_menu_keyboard(),
                )
        else:
            # История отключена — просто приветствие
            await client.send_message(
                chat_id=chat_id,
                text=(
                    f"👋 Привет, <b>{user_name}</b>!\n\n"
                    "Я <b>AI-ассистент TEMPO</b> — ваш корпоративный помощник.\n\n"
                    "Задавайте вопросы — я готов помочь! 🚀"
                ),
            )
    except Exception as e:
        logger.error(f"Ошибка в _handle_bot_started: {e}")


async def _handle_message_created(
    event: Dict,
    client: MaxBotClient,
    assistant: AssistantService,
):
    """
    Обработчик входящего текстового сообщения.
    Поддерживает команды /start, /clear, /change_company.
    """
    message: Dict = event.get("message", {})
    body: Dict = message.get("body", {})
    
    # Пытаемся достать текст из основного сообщения или из пересланного (link)
    text: str = (body.get("text") or "").strip()
    if not text:
        # В пересланных сообщениях текст лежит прямо в link.message.text
        text = (message.get("link", {}).get("message", {}).get("text") or "").strip()

    sender: Dict = message.get("sender", {})
    user_id: int = sender.get("user_id", 0)
    user_name: str = sender.get("name", "")

    # Ищем голосовое сообщение, если текста нет
    # Вложения могут быть в body, в корне сообщения или во вложенном объекте при пересылке (link.message)
    attachments = body.get("attachments") or \
                  message.get("attachments") or \
                  message.get("link", {}).get("message", {}).get("attachments") or []
    
    voice_url = None
    for att in attachments:
            voice_url = att.get("payload", {}).get("url")
            break

    chat_id: int = message.get("recipient", {}).get("chat_id", 0)
    session_id = _get_session_id(user_id)
    logger.info(f"🟢 [VER 2.7] MAX [message_created]: user_id={user_id}, chat_id={chat_id}")
    
    status_mid = None
    status_task = None

    def extract_mid(resp):
        if not resp: return None
        return resp.get("mid") or \
               resp.get("id") or \
               resp.get("message_id") or \
               resp.get("message", {}).get("mid") or \
               resp.get("message", {}).get("id") or \
               resp.get("message", {}).get("body", {}).get("mid")

    try:
        if not text and not voice_url:
            logger.debug(f"MAX: ⚠️ Проигнорировано сообщение. Text: '{text[:10]}', chat_id: {chat_id}")
            return

        if assistant.chat_history:
            # Обновляем время последней активности
            try:
                await assistant.chat_history.update_last_activity(session_id, "max")
            except Exception as e:
                logger.error(f"Error updating MAX user activity: {e}")

            # Проверяем блокировку
            try:
                if await assistant.chat_history.is_user_blocked(session_id):
                    await client.send_message(chat_id, "⛔ Доступ ограничен.")
                    return
            except Exception as e:
                logger.error(f"Error checking MAX user block status: {e}")

        # ── Команды ──────────────────────────────────────────────────────
        if text.lower() in ("/start", "/start@all"):
            await _handle_bot_started(
                {"chat": {"chat_id": chat_id}, "user": {"user_id": user_id, "name": user_name}},
                client,
                assistant,
            )
            return

        if text.lower() in ("/clear",):
            try:
                if assistant.chat_history:
                    await assistant.chat_history.clear_history(session_id, clear_summary=True)
                    await assistant.orchestrator.clear_memory(session_id)
                    await client.send_message(chat_id, "✅ История разговора очищена.")
                else:
                    await client.send_message(chat_id, "ℹ️ История разговоров отключена в настройках.")
            except Exception as e:
                logger.error(f"Ошибка очистки истории MAX: {e}")
                await client.send_message(chat_id, "❌ Ошибка при очистке истории.")
            return

        if text.lower() in ("/change_company",):
            await client.send_message(
                chat_id=chat_id,
                text="🏭 Выберите ваше предприятие:",
                keyboard=_get_company_keyboard(),
            )
            return

        if text.lower() in ("/menu", "меню"):
            await client.send_message(
                chat_id=chat_id,
                text="📋 <b>Главное меню:</b>",
                keyboard=_get_main_menu_keyboard(),
            )
            return

        session_id = _get_session_id(user_id)
        if assistant.chat_history:
            user_company = await assistant.chat_history.get_user_company(session_id)
            if not user_company:
                await client.send_message(
                    chat_id=chat_id,
                    text="🏭 Пожалуйста, выберите ваше предприятие перед началом работы:",
                    keyboard=_get_company_keyboard(),
                )
                return

        # 1. Начальная подготовка и статус
        await client.send_action(chat_id, "typing_on")
            
        # 2. Обработка голоса (если есть)
        if voice_url:
            status_resp = await client.send_message(chat_id, "🎤 Скачиваю и распознаю аудио...")
            status_mid = extract_mid(status_resp)
            logger.info(f"🔍 MAX: Статус 'Скачиваю' отправлен, mid={status_mid}")
            
            # Скачивание файла из CDN MAX
            async with aiohttp.ClientSession() as download_session:
                async with download_session.get(voice_url) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                    else:
                        if status_mid: await client.delete_message(chat_id, status_mid)
                        await client.send_message(chat_id, "❌ Не удалось получить аудиофайл из облака MAX.")
                        return

            # STT
            transcript = await assistant.transcribe_audio(bytes(audio_bytes))
            if not transcript:
                if status_mid: await client.delete_message(chat_id, status_mid)
                await client.send_message(chat_id, "❌ Не удалось распознать речь.")
                return
            text = transcript

        # 3. Если текста нет (ни в сообщении, ни после STT) — выходим
        if not text:
            if status_mid: await client.delete_message(chat_id, status_mid)
            return

        # 4. Создаем/обновляем статус "думает" для основного процесса LLM
        if not status_mid:
            status_resp = await client.send_message(chat_id, "🔄 Готовлю ответ...")
            status_mid = extract_mid(status_resp)
        else:
            await client.edit_message(chat_id, status_mid, "🔄 Готовлю ответ...")

        # Запускаем циклическое обновление статуса (как в Telegram)
        async def update_status_periodically(mid):
            status_texts = ["🔄 Готовлю ответ...", "⏳ Анализирую запрос...", "🔍 Поиск информации...", "📝 Формирую ответ..."]
            counter = 0
            while True:
                try:
                    await asyncio.sleep(5)
                    counter += 1
                    await client.edit_message(chat_id, mid, status_texts[counter % len(status_texts)])
                except asyncio.CancelledError: break
                except Exception: break
        
        if status_mid:
            status_task = asyncio.create_task(update_status_periodically(status_mid))

        # 5. Основной запрос к ассистенту (RAG + LLM)
        result = await assistant.process_text_query(
            query=text,
            limit=15,
            session_id=session_id,
            platform="max",
            user_name=user_name,
            user_company=user_company
        )

        # Логируем, какие документы были найдены
        docs = result.get("documents", [])
        if docs:
            source_names = [d.get("source", "unknown") for d in docs]
            logger.info(f"📚 Найдено документов: {len(docs)}. Источники: {list(set(source_names))}")
        else:
            logger.warning("📚 Документы в базе знаний не найдены!")

        # 6. Очистка статуса перед выводом ответа (теперь только через finally)
        if status_task: status_task.cancel()

        answer: str = result.get("answer", "")
        if not answer:
            await client.send_message(chat_id, "❌ Извините, не удалось получить ответ.")
            return

        # 7. Отправка итогового ответа
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        # Находим картинки в тексте ПЕРЕД тем как их удалить
        images_from_text = re.findall(image_pattern, answer)
        
        clean_answer = re.sub(image_pattern, '', answer).strip()
        parts = _split_long_message(clean_answer)
        for part in parts:
            if part:
                await client.send_message(chat_id, part)
                if len(parts) > 1: await asyncio.sleep(0.3)
                
        # 8. Отправка дополнительных документов
        documents_to_send = result.get("documents_to_send", [])
        
        # Собираем все пути к файлам, которые уже есть в списке на отправку, чтобы не дублировать
        already_sending_paths = {d.document_path for d in documents_to_send}
        
        # Добавляем картинки из текста в список на отправку
        for alt_text, img_path in images_from_text:
            if img_path.startswith('data/') and img_path not in already_sending_paths:
                from src.core.document_sender import DocumentRule
                documents_to_send.append(DocumentRule(
                    keywords=[],
                    document_path=img_path,
                    description=alt_text or "Изображение из ответа",
                    file_type="image"
                ))
                already_sending_paths.add(img_path)

        if documents_to_send:
            import os
            base_paths = ["e:/Old/bots/Worker/iTEMPO/iTEMPO_test", "/home/administrator/iTEMPO_test"]
            for doc_rule in documents_to_send:
                full_path = None
                for bp in base_paths:
                    p = os.path.join(bp, doc_rule.document_path.replace('/', os.sep))
                    if os.path.exists(p):
                        full_path = p
                        break
                
                if full_path:
                    try:
                        with open(full_path, "rb") as f:
                            file_bytes = f.read()
                        
                        file_type = doc_rule.file_type
                        if file_type == "auto":
                            ext = os.path.splitext(full_path)[1].lower()
                            if ext in ['.jpg', '.jpeg', '.png', '.gif']: file_type = "image"
                            else: file_type = "file"
                        
                        upload_type = "image" if file_type == "image" else "file"
                        filename = os.path.basename(full_path)
                        
                        token, file_id = await client.upload_file(
                            file_bytes=file_bytes, 
                            filename=filename,
                            upload_type=upload_type,
                            kind=""
                        )
                        
                        if token:
                            # Для изображений ID не передается, только токен
                            final_id = None if upload_type == "image" else file_id
                            await client.send_message(
                                chat_id, 
                                doc_rule.description, 
                                attachment_token=token,
                                attachment_type=upload_type,
                                attachment_id=final_id
                            )
                    except Exception as e:
                        logger.error(f"MAX bot: Error sending document {doc_rule.document_path}: {e}")

        return

    except AssistantError as e:
        # Если пришла ошибка блокировки сессии
        if "wait for the response" in str(e).lower() or "already processing" in str(e).lower():
            await client.send_message(
                chat_id, 
                "⏳ Подождите — ваш предыдущий запрос ещё обрабатывается. Я отвечу на него совсем скоро!"
            )
        else:
            await client.send_message(chat_id, f"⚠️ {str(e)}")
    except Exception as e:
        logger.error(f"Ошибка в _handle_message_created MAX: {e}", exc_info=True)
        await client.send_message(chat_id, "❌ Произошла ошибка при обработке.")
    finally:
        # Гарантируем очистку статуса в любой ситуации
        if status_task and not status_task.done():
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass
        
        # Удаляем статусное сообщение, если оно было создано
        if status_mid:
            logger.info(f"🧹 MAX: Попытка удаления статусного сообщения {status_mid}...")
            await client.delete_message(chat_id, status_mid)


async def _handle_message_callback(
    event: Dict,
    client: MaxBotClient,
    assistant: AssistantService,
):
    """
    Обработчик нажатия inline-кнопки (callback от keyboard).
    Используется для выбора предприятия.
    """
    callback_id: str = event.get("callback", {}).get("callback_id", "")
    payload: str = event.get("callback", {}).get("payload", "")

    callback_obj = event.get("callback") or {}
    user_obj = callback_obj.get("user") or event.get("user") or {}
    
    user_id: int = user_obj.get("user_id") or user_obj.get("id") or \
                  event.get("user_id") or event.get("sender_id") or 0
    
    # Пытаемся достать chat_id из разных мест (зависит от типа чата и версии API)
    message_obj = event.get("message", {})
    chat_id: int = message_obj.get("recipient", {}).get("chat_id") or \
                  message_obj.get("chat", {}).get("chat_id") or \
                  event.get("chat_id", 0)

    session_id = _get_session_id(user_id)
    # Трассировка для финальной проверки
    if user_id == 0:
        logger.warning(f"⚠️ MAX [message_callback]: user_id всё еще 0! event keys: {list(event.keys())}")
    else:
        logger.info(f"🟡 MAX [message_callback]: Успешно извлечен user_id={user_id}, session_id={session_id}")

    # Подтверждаем нажатие (убирает "загрузку" на кнопке)
    if callback_id:
        await client.answer_callback(callback_id)

    # 1. Обработка системных команд из меню
    if payload == "/start":
        await _handle_bot_started(event, client, assistant)
        return

    if payload == "/clear":
        try:
            if assistant.chat_history:
                await assistant.chat_history.clear_history(session_id, clear_summary=True)
                if chat_id:
                    await client.send_message(chat_id, "✅ История разговора очищена.")
            else:
                if chat_id:
                    await client.send_message(chat_id, "ℹ️ История разговоров отключена в настройках.")
        except Exception as e:
            logger.error(f"Ошибка очистки истории MAX через меню: {e}")
            if chat_id:
                await client.send_message(chat_id, "❌ Ошибка при очистке истории.")
        return

    if payload == "/change_company":
        if chat_id:
            await client.send_message(
                chat_id=chat_id,
                text="🏭 Выберите ваше предприятие:",
                keyboard=_get_company_keyboard(),
            )
        return

    # 2. Обработка выбора компании
    if not payload.startswith("company_"):
        return

    company_id = payload.split("company_", 1)[1]
    company_name = COMPANIES.get(company_id, company_id)
    session_id = _get_session_id(user_id)

    try:
        if assistant.chat_history:
            await assistant.chat_history.set_user_company(session_id, company_id)
            if chat_id:
                await client.send_message(
                    chat_id=chat_id,
                    text=(
                        f"✅ Предприятие <b>{company_name}</b> успешно установлено!\n\n"
                        "Теперь вы можете задавать свои вопросы."
                    ),
                )
        else:
            if chat_id:
                await client.send_message(chat_id, "❌ Ошибка: модуль истории не подключен.")
    except Exception as e:
        logger.error(f"Ошибка установки компании MAX (user={user_id}): {e}")
        if chat_id:
            await client.send_message(chat_id, "❌ Произошла ошибка при сохранении настроек.")


# ── Диспетчер событий ──────────────────────────────────────────────────────

async def _dispatch_event(
    event: Dict,
    client: MaxBotClient,
    assistant: AssistantService,
):
    """Определяет тип события и вызывает соответствующий обработчик."""
    update_type: str = event.get("update_type", "")

    try:
        if update_type == "bot_started":
            await _handle_bot_started(event, client, assistant)

        elif update_type == "message_created":
            await _handle_message_created(event, client, assistant)

        elif update_type == "message_callback":
            await _handle_message_callback(event, client, assistant)

        else:
            # Неизвестный тип события — игнорируем
            logger.debug(f"MAX: неизвестный update_type='{update_type}', пропускаем")

    except Exception as e:
        logger.error(f"Необработанное исключение в dispatch_event ({update_type}): {e}", exc_info=True)


# ── Основной Long Polling цикл ─────────────────────────────────────────────

async def run_max_bot(config: Config, assistant: Optional[AssistantService] = None):
    """
    Запускает MAX бота в режиме Long Polling.

    Это корутина — запускается через asyncio.run() или asyncio.create_task().

    Args:
        config: Конфигурация приложения (должен быть заполнен config.max_token)
        assistant: Существующий экземпляр AssistantService (опционально).
                   Если не передан — создаётся новый.
    """
    if not config.max_token:
        logger.error("❌ MAX_TOKEN не задан в .env — MAX бот не будет запущен.")
        return

    logger.info("🚀 Запуск MAX бота...")

    # Создаём ассистента если не передан
    if assistant is None:
        assistant = AssistantService(config)

    client = MaxBotClient(token=config.max_token)

    # Проверяем токен и настраиваем меню
    try:
        me = await client.get_me()
        if me:
            bot_name = me.get("name", "Unknown")
            bot_id = me.get("user_id", "?")
            # Логируем успешную авторизацию
            logger.info(f"✅ MAX бот авторизован: {bot_name} (ID: {bot_id})")
        else:
            logger.error("❌ MAX API: не удалось получить информацию о боте. Проверьте MAX_TOKEN.")
            await client.close()
            return
    except Exception as e:
        logger.error(f"❌ Ошибка проверки MAX токена: {e}")
        await client.close()
        return

    # Если задан Webhook URL — переходим в режим Webhook
    if config.max_webhook_url:
        logger.info(f"🔗 MAX бот: включен режим Webhook. Установка URL: {config.max_webhook_url}")
        success = await client.set_webhook(config.max_webhook_url, secret=config.max_webhook_secret)
        if success:
            logger.info("✅ MAX Webhook успешно установлен.")
            try:
                # В режиме вебхука просто ждем вечно, не блокируя цикл
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                logger.info("MAX бот: получен сигнал остановки.")
            finally:
                await client.delete_webhook()
                await client.close()
                logger.info("MAX бот остановлен.")
            return
        else:
            logger.error("❌ Не удалось установить MAX Webhook. Возврат к Long Polling (fallback).")

    # В противном случае используем Long Polling
    logger.info("📡 MAX бот начал прослушивание событий (Long Polling)...")
    
    # Принудительно отключаем webhook на серверах MAX, чтобы Long Polling заработал корректно
    await client.delete_webhook()

    marker: Optional[int] = None
    retry_count = 0

    try:
        while True:
            try:
                data = await client.get_updates(marker=marker)

                if data is None:
                    # Сетевая ошибка — ждём и повторяем
                    retry_count += 1
                    wait = min(MAX_RETRY_DELAY * retry_count, 60)
                    logger.warning(f"MAX: get_updates вернул None (попытка {retry_count}), повтор через {wait}с")
                    await asyncio.sleep(wait)
                    continue

                retry_count = 0  # Сбрасываем счётчик при успехе

                updates: List[Dict] = data.get("updates", [])
                new_marker: Optional[int] = data.get("marker")
                
                # Добавляем лог для отладки
                if updates:
                    logger.info(f"✅ MAX: Получено {len(updates)} обновлений! (marker {marker} -> {new_marker})")
                    for u in updates:
                        logger.info(f"Update: {u.get('update_type')}")
                else:
                    logger.info(f" MAX: пусто (marker {marker} -> {new_marker})")

                # Обновляем маркер если получили
                if new_marker is not None:
                    marker = new_marker

                # Обрабатываем события конкурентно (не блокируя друг друга)
                if updates:
                    tasks = [
                        asyncio.create_task(_dispatch_event(ev, client, assistant))
                        for ev in updates
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Короткая пауза если не было событий (снижаем нагрузку)
                if not updates:
                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                logger.info("MAX бот: получен сигнал остановки.")
                break
            except Exception as e:
                retry_count += 1
                wait = min(MAX_RETRY_DELAY * retry_count, 60)
                logger.error(f"MAX: ошибка в polling цикле (попытка {retry_count}): {e}", exc_info=True)
                await asyncio.sleep(wait)

    finally:
        await client.close()
        logger.info("MAX бот остановлен.")

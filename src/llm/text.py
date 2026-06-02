"""
Async текстовый LLM через Gemini.

ИСПРАВЛЕНО: Прокси через ClientManager, НЕ os.environ.
ДОБАВЛЕНО: Retry-логика с fallback между API ключами и моделями.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from google.genai import types

from src.core.clients import ClientManager
from src.core.config import Config
from src.core.exceptions import LLMError
from src.core.models_loader import TextModelConfig
from src.llm.model_fallback_manager import ModelFallbackManager

logger = logging.getLogger(__name__)


class TextLLMService:
    """
    Async текстовый LLM через Gemini.

    ИСПРАВЛЕНО: Прокси передается через ClientManager.
    ДОБАВЛЕНО: Автоматическое переключение API ключей и моделей при ошибках.
    """

    def __init__(self, config: Config, model_config: TextModelConfig = None):
        self.config = config
        self.client_manager = ClientManager.get_instance(config)
        # Используем конфиг из YAML если не передан явно
        self.model_config = model_config or config.text_model_config

        # Менеджер fallback моделей для 503 ошибок
        self.model_fallback_manager = None
        if config.text_model_fallbacks:
            self.model_fallback_manager = ModelFallbackManager(
                primary_model=config.text_model,
                fallback_models=config.text_model_fallbacks,
            )

    def _get_safety_settings(self) -> list:
        """Safety settings для отключения фильтрации (если задано в model_config)."""
        if not self.model_config.disable_safety:
            return []
        return [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        ]

    async def _call_api(self, contents, generation_config, operation_name: str = "LLM call"):
        """
        Unified Gemini API call point with retry/fallback.
        """
        return await self._retry_with_fallback(
            lambda model, api_version, api_key: self._execute_api_call(contents, generation_config, model, api_version, api_key),
            operation_name,
        )

    async def _call_api_stream(self, contents, generation_config, operation_name: str = "LLM stream"):
        """
        Gemini API streaming call with retry/fallback at stream start.
        """
        return await self._retry_with_fallback(
            lambda model, api_version, api_key: self._execute_api_call_stream(contents, generation_config, model, api_version, api_key),
            operation_name,
        )

    async def _execute_api_call(self, contents, generation_config, model_name: str, api_version: str, api_key: str):
        """Execute single API call (without retry)."""
        # Передаём api_key явно, чтобы get_gemini_client создал клиент под нужный ключ
        client = self.client_manager.get_gemini_client(api_version, api_key=api_key)
        model = self._get_full_model_name(model_name)
        return await client.aio.models.generate_content(model=model, contents=contents, config=generation_config)

    async def _execute_api_call_stream(self, contents, generation_config, model_name: str, api_version: str, api_key: str):
        """Execute single API call in streaming mode (without retry)."""
        client = self.client_manager.get_gemini_client(api_version, api_key=api_key)
        model = self._get_full_model_name(model_name)
        return await client.aio.models.generate_content_stream(model=model, contents=contents, config=generation_config)

    @staticmethod
    def _parse_retry_delay(error_str: str, default: float = 10.0) -> float:
        """
        Извлекает retryDelay из текста ошибки 429.

        Google возвращает задержку в формате:
          'retryDelay': '31s'
          Please retry in 31.008674714s.
        """
        import re

        # Пробуем "Please retry in Xs"
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str, re.IGNORECASE)
        if match:
            return min(float(match.group(1)), 60.0)  # cap 60s

        # Пробуем "'retryDelay': 'Xs'"
        match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", error_str, re.IGNORECASE)
        if match:
            return min(float(match.group(1)), 60.0)

        return default

    @staticmethod
    def _is_rpm_limit(error_lower: str) -> bool:
        """
        Определяет, является ли ошибка 429 лимитом в минуту (RPM), а не в день (RPD).

        Google включает quotaId в текст ошибки, например:
        - 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier' → RPM
        - 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' → RPD
        """
        # Приоритет: парсим quotaId (самый надёжный индикатор)
        if "perminute" in error_lower:
            return True
        if "perday" in error_lower:
            return False

        # Вторичные индикаторы
        if "daily" in error_lower or "rpd" in error_lower:
            return False

        # По умолчанию считаем RPM (безопаснее — подождём, а не сдадимся)
        return True

    async def _retry_with_fallback(
        self,
        operation: Callable,
        operation_name: str = "LLM operation",
        max_retries: int = 100,
        initial_model: Optional[str] = None,
        initial_api_version: Optional[str] = None,
    ) -> Any:
        """
        Retry с fallback между API ключами (429) и моделями (503).
        Изолировано от глобального конфига для поддержки конкурентных запросов.

        Стратегия обработки 429:
        - RPM (лимит в минуту): asyncio.sleep(retryDelay), затем повтор с тем же ключом/моделью.
          Ротация ключей бессмысленна, так как RPM — проектный лимит.
        - RPD (дневной лимит): пометить модель как исчерпанную, ротация на fallback.
        """
        # Всегда начинаем с лучшей доступной модели на текущий момент
        current_model = initial_model
        current_api_version = initial_api_version
        if self.model_fallback_manager:
            if not current_model or not self.model_fallback_manager.is_model_available(current_model):
                current_model = self.model_fallback_manager.get_best_available_model()
                # Сбрасываем версию API, чтобы получить правильную версию для fallback-модели
                current_api_version = None
        if not current_model:
            current_model = self.config.text_model

        # Приоритет версии
        if not current_api_version and self.model_fallback_manager:
            current_api_version = self.model_fallback_manager.get_api_version_for_model(current_model)
        if not current_api_version:
            current_api_version = self.config.text_api_version

        last_error = None

        try:
            # Увеличен таймаут для батчевых операций, которые могут ждать RPM cooldown
            overall_timeout = 600.0  # 10 минут
            start_time = time.time()

            for attempt in range(max_retries):
                # Check if we've exceeded overall timeout
                if time.time() - start_time > overall_timeout:
                    logger.error(f"Overall timeout of {overall_timeout}s exceeded for {operation_name}")
                    raise LLMError(f"Timeout: {operation_name} took longer than {overall_timeout}s")

                # Get current API key ONCE per attempt
                current_api_key = (
                    self.client_manager.api_key_manager.get_current_key()
                    if self.client_manager.api_key_manager
                    else self.config.gemini_api_key
                )

                try:
                    # Log current model and API version
                    logger.debug(
                        "Retry attempt %d/%d — модель: %s (API: %s)",
                        attempt + 1, max_retries, current_model, current_api_version,
                    )

                    # Таймауты по модели (в секундах).
                    # Если модель не в словаре — используется DEFAULT_TIMEOUT.
                    MODEL_TIMEOUTS = {
                        "gemini-2.5-flash-lite": 40.0,          # стабильная, но может давать 20-35с при нагрузке
                        "gemini-3.1-flash-lite": 40.0,          # стабильная версия
                    }
                    DEFAULT_TIMEOUT = 30.0
                    timeout = MODEL_TIMEOUTS.get(current_model, DEFAULT_TIMEOUT)

                    try:
                        logger.debug("Вызов API: модель=%s timeout=%.0fs", current_model, timeout)
                        response = await asyncio.wait_for(
                            operation(current_model, current_api_version, current_api_key),
                            timeout=timeout,
                        )
                        # Успешный ответ — сбрасываем fallback-менеджер на primary,
                        # чтобы следующий запрос снова начинал с основной модели.
                        if self.model_fallback_manager:
                            self.model_fallback_manager.reset_to_primary()
                        return response, current_model
                    except asyncio.TimeoutError as te:
                        logger.debug(
                            "⏰ Таймаут %.0fs для модели %s (попытка %d/%d) — переключаю на fallback",
                            timeout, current_model, attempt + 1, max_retries,
                        )
                        akm = self.client_manager.api_key_manager
                        # Таймаут = модель перегружена, переключаемся на fallback-модель
                        if self.model_fallback_manager:
                            new_model = self.model_fallback_manager.rotate_model(
                                current_model, f"timeout {timeout}s"
                            )
                            if new_model:
                                current_model = new_model
                                current_api_version = self.model_fallback_manager.get_api_version_for_model(new_model)
                                if akm:
                                    akm.reset_exhausted_keys()
                                last_error = te
                                continue
                        # Нет fallback — просто помечаем ключ и пробуем снова
                        if akm:
                            akm.mark_key_exhausted(current_api_key, f"timeout {timeout}s на {current_model}")
                        last_error = te
                        continue

                except Exception as e:
                    error_str = str(e)
                    error_lower = error_str.lower()
                    last_error = e

                    # --- 401/403: Authentication / Permission errors ---
                    is_auth_error = (
                        "401" in error_str
                        or "403" in error_str
                        or "unauthenticated" in error_lower
                        or "permission_denied" in error_lower
                        or "invalid" in error_lower
                        or "not active" in error_lower
                        or "deleted or disabled" in error_lower
                    )
                    if is_auth_error:
                        logger.warning(
                            "🔑 Ошибка авторизации (401/403) для API ключа (модель: %s): %s",
                            current_model, error_str
                        )
                        akm = self.client_manager.api_key_manager
                        if akm:
                            # Помечаем ключ как исчерпанный
                            akm.mark_key_exhausted(current_api_key, f"auth error: {error_str}")
                            # Ротируем на следующий ключ
                            new_key = akm.rotate_key(f"auth error: {error_str}")
                            if new_key:
                                logger.debug(
                                    "🔄 Ошибка авторизации: Переключаюсь на следующий API ключ: %s",
                                    akm.get_masked_key(new_key)
                                )
                                continue
                        raise LLMError(f"Ошибка авторизации API: {error_str}") from e

                    # --- 429: Rate limit ---
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        is_rpm = self._is_rpm_limit(error_lower)
                        retry_delay = self._parse_retry_delay(error_str, default=15.0)

                        if is_rpm:
                            # =============================================
                            # RPM (лимит в минуту) — ждём и повторяем.
                            # Ротация ключей бессмысленна: RPM — проектный лимит,
                            # общий для всех ключей одного проекта.
                            # =============================================
                            logger.debug(
                                "⏳ RPM лимит для %s (модель: %s, retry_delay: %.0fs), попытка %d/%d",
                                operation_name, current_model, retry_delay, attempt + 1, max_retries,
                            )

                            # Помечаем текущую модель на кулдаун ПЕРЕД ротацией
                            if self.model_fallback_manager:
                                self.model_fallback_manager.mark_model_rpm_limit(current_model, int(retry_delay))

                                # Пробуем переключиться на доступную модель
                                new_model = self.model_fallback_manager.rotate_model(
                                    current_model, f"RPM limit, retry in {retry_delay:.0f}s"
                                )
                                if new_model:
                                    # Есть доступная модель — переключаемся без ожидания
                                    current_model = new_model
                                    current_api_version = self.model_fallback_manager.get_api_version_for_model(new_model)
                                    akm = self.client_manager.api_key_manager
                                    if akm:
                                        akm.reset_exhausted_keys()
                                    logger.debug(
                                        "🔄 RPM: переключаюсь на доступную модель %s (API: %s)",
                                        new_model, current_api_version,
                                    )
                                    continue

                                # Все модели заблокированы — ждём минимальный cooldown
                                wait_time = self.model_fallback_manager.get_min_cooldown_wait()
                                if wait_time is not None and wait_time > 0:
                                    logger.info(
                                        "💤 Все модели в RPM кулдауне. Жду %.0fs...",
                                        wait_time,
                                    )
                                    await asyncio.sleep(wait_time)
                                    akm = self.client_manager.api_key_manager
                                    if akm:
                                        akm.reset_exhausted_keys()
                                    # После сна пересчитываем лучшую модель
                                    current_model = self.model_fallback_manager.get_best_available_model()
                                    current_api_version = self.model_fallback_manager.get_api_version_for_model(current_model)
                                    continue
                                elif wait_time == 0.0:
                                    # Есть доступная модель (ситуация гонки с другими задачами)
                                    current_model = self.model_fallback_manager.get_best_available_model()
                                    current_api_version = self.model_fallback_manager.get_api_version_for_model(current_model)
                                    akm = self.client_manager.api_key_manager
                                    if akm:
                                        akm.reset_exhausted_keys()
                                    continue

                            # Нет model_fallback_manager или все в RPD — ждём и пробуем тот же
                            logger.info(
                                "💤 Нет fallback-моделей. Жду %.0fs перед повтором...",
                                retry_delay,
                            )
                            await asyncio.sleep(retry_delay)
                            akm = self.client_manager.api_key_manager
                            if akm:
                                akm.reset_exhausted_keys()
                            continue
                        else:
                            # =============================================
                            # RPD (дневной лимит) — модель исчерпана на сегодня.
                            # Помечаем и переключаемся на fallback.
                            # =============================================
                            logger.debug(
                                "🚫 RPD лимит (дневной) для модели %s, попытка %d/%d",
                                current_model, attempt + 1, max_retries,
                            )

                            akm = self.client_manager.api_key_manager
                            if akm:
                                # Помечаем ключ как исчерпанный для этой модели
                                akm.mark_key_exhausted(current_api_key, f"RPD limit on {current_model}")
                                # Пытаемся найти следующий доступный ключ
                                new_key = akm.rotate_key(f"RPD limit on {current_model}")
                                if new_key:
                                    logger.debug(
                                        "🔄 RPD: Переключаюсь на следующий API ключ: %s для модели %s",
                                        akm.get_masked_key(new_key), current_model
                                    )
                                    continue

                            # Если ключи исчерпаны (или их нет), помечаем модель как daily exhausted и переключаемся на fallback
                            if self.model_fallback_manager:
                                self.model_fallback_manager.mark_model_daily_exhausted(current_model)

                                new_model = self.model_fallback_manager.rotate_model(
                                    current_model, "RPD daily limit reached on all keys"
                                )
                                if new_model:
                                    current_model = new_model
                                    current_api_version = self.model_fallback_manager.get_api_version_for_model(new_model)
                                    if akm:
                                        akm.reset_exhausted_keys()
                                    logger.debug(
                                        "🔄 RPD: переключаюсь на fallback -> %s (API: %s)",
                                        new_model, current_api_version,
                                    )
                                    continue

                                # Все модели заблокированы. Проверяем, есть ли RPM-кулдауны.
                                # Если есть — стоит подождать (RPM кулдаун < 1 мин).
                                wait_time = self.model_fallback_manager.get_min_cooldown_wait()
                                if wait_time is not None and wait_time > 0:
                                    logger.info(
                                        "💤 Все модели заблокированы, но есть RPM-кулдаун (%.0fs). Жду...",
                                        wait_time,
                                    )
                                    await asyncio.sleep(wait_time)
                                    current_model = self.model_fallback_manager.get_best_available_model()
                                    current_api_version = self.model_fallback_manager.get_api_version_for_model(current_model)
                                    if akm:
                                        akm.reset_exhausted_keys()
                                    continue

                            raise LLMError(
                                f"Дневной лимит исчерпан для всех моделей: {operation_name}."
                            ) from e

                    # --- Server errors / Bad Request / Not Found: switch model ---
                    error_lower = error_str.lower()
                    is_timeout = isinstance(e, asyncio.TimeoutError) or "timeout" in error_lower
                    is_rotation_trigger = (
                        "503" in error_str
                        or "UNAVAILABLE" in error_str
                        or "overloaded" in error_lower
                        or "404" in error_str  # Модель не найдена в этой версии API или регионе
                        or "400" in error_str  # Некорректные параметры для данной модели
                        or "RemoteProtocolError" in error_str
                        or "Server disconnected" in error_str
                        or "disconnected without" in error_lower
                        or "CancelledError" in error_str
                        or is_timeout
                    )

                    if is_rotation_trigger and self.model_fallback_manager:
                        # Помечаем модель на короткий RPM кулдаун при 503/недоступности
                        cooldown_secs = 10 if ("503" in error_str or "unavailable" in error_lower) else 15
                        self.model_fallback_manager.mark_model_rpm_limit(current_model, cooldown_secs)

                        new_model = self.model_fallback_manager.rotate_model(
                            current_model, f"503/error: {operation_name}"
                        )
                        if new_model:
                            current_model = new_model
                            current_api_version = self.model_fallback_manager.get_api_version_for_model(new_model)
                            akm = self.client_manager.api_key_manager
                            if akm:
                                akm.reset_exhausted_keys()
                            continue

                        # Если все модели оказались заблокированы/в кулдауне, ждем перед повтором
                        wait_time = self.model_fallback_manager.get_min_cooldown_wait()
                        if wait_time is not None and wait_time > 0:
                            logger.info(
                                "💤 Все модели перегружены или недоступны (503). Жду %.0fs перед повторной попыткой...",
                                wait_time
                            )
                            await asyncio.sleep(wait_time)
                            akm = self.client_manager.api_key_manager
                            if akm:
                                akm.reset_exhausted_keys()
                            current_model = self.model_fallback_manager.get_best_available_model()
                            current_api_version = self.model_fallback_manager.get_api_version_for_model(current_model)
                            continue

                        raise LLMError(f"Все модели перегружены: {operation_name}") from e

                    raise LLMError(f"Ошибка {operation_name}: {e}") from e

            raise LLMError(f"Все попытки {operation_name} исчерпаны") from last_error

        finally:
            # Ничего не нужно восстанавливать, так как мы не меняли глобальный конфиг!
            pass

    async def query(
        self,
        context_chunks: List[str],
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Генерация ответа через Gemini с поддержкой истории и инструментов."""
        try:
            truncated_chunks = self._truncate_chunks(context_chunks, self.model_config.max_context_chars)
            context_str = "\n\n".join(truncated_chunks) if truncated_chunks else "Информация не найдена."
            system_prompt = self._get_system_prompt(context_str)

            generation_config = types.GenerateContentConfig(
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
                safety_settings=self._get_safety_settings() or None,
            )

            if tools:
                generation_config.tools = [
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool["name"],
                                description=tool["description"],
                                parameters=tool.get("parameters", {}),
                            )
                            for tool in tools
                        ]
                    )
                ]

            # Формирование contents
            contents = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {
                    "role": "model",
                    "parts": [{"text": "Контекст принят. Ожидаю вопрос."}],
                },
            ]

            if history:
                for msg in history:
                    expected_role = "user" if len(contents) % 2 == 0 else "model"
                    role = "user" if msg["role"] == "user" else "model"
                    if role == expected_role:
                        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                    else:
                        contents[-1]["parts"][0]["text"] += f"\n\n{msg['content']}"

            if contents[-1]["role"] == "user":
                contents.append({"role": "model", "parts": [{"text": "Слушаю вас."}]})
            contents.append({"role": "user", "parts": [{"text": question}]})

            response, model_name = await self._call_api(contents, generation_config, "text query")

            result = {"text": "", "tool_calls": [], "model": model_name}
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    logger.debug(f"🔍 Response part: {part}")
                    if hasattr(part, "text") and part.text:
                        result["text"] = part.text
                    elif hasattr(part, "function_call") and part.function_call:
                        logger.info(f"🎯 Function call detected: {part.function_call.name}")
                        result["tool_calls"].append(
                            {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args),
                            }
                        )
            if not result["tool_calls"] and not result["text"]:
                result["text"] = response.text if hasattr(response, "text") else ""
            return result

        except Exception as e:
            logger.exception("Error in text query")
            raise LLMError(f"Gemini API error: {e}") from e

    async def query_stream(
        self,
        context_chunks: List[str],
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """Потоковая генерация ответа через Gemini."""
        try:
            truncated_chunks = self._truncate_chunks(context_chunks, self.model_config.max_context_chars)
            context_str = "\n\n".join(truncated_chunks) if truncated_chunks else "Информация не найдена."
            system_prompt = self._get_system_prompt(context_str)

            generation_config = types.GenerateContentConfig(
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
                safety_settings=self._get_safety_settings() or None,
            )

            # Формирование contents
            contents = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {
                    "role": "model",
                    "parts": [{"text": "Контекст принят. Ожидаю вопрос."}],
                },
            ]

            if history:
                for msg in history:
                    expected_role = "user" if len(contents) % 2 == 0 else "model"
                    role = "user" if msg["role"] == "user" else "model"
                    if role == expected_role:
                        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                    else:
                        contents[-1]["parts"][0]["text"] += f"\n\n{msg['content']}"

            if contents[-1]["role"] == "user":
                contents.append({"role": "model", "parts": [{"text": "Слушаю вас."}]})
            contents.append({"role": "user", "parts": [{"text": question}]})

            stream_data = await self._call_api_stream(contents, generation_config, "text stream query")
            stream, model_name = stream_data

            async for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.exception("Error in text streaming query")
            raise LLMError(f"Gemini Streaming API error: {e}") from e

    def _truncate_chunks(self, chunks: List[str], max_chars: int) -> List[str]:
        """Обрезает список чанков так, чтобы сумма их длин не превышала лимит."""
        result = []
        total = 0
        for chunk in chunks:
            if total + len(chunk) > max_chars and result:
                break
            result.append(chunk)
            total += len(chunk) + 2
        return result

    async def decontextualize_query(self, query: str, history: List[Dict[str, str]], company_name: Optional[str] = None) -> str:
        """
        Переписывает запрос с учетом истории и текущей компании.
        """
        if not history or len(query) < 3:
            return query

        try:
            recent_history = "\n".join(
                [
                    f"{'Пользователь' if msg['role'] == 'user' else 'Ассистент'}: {msg['content'][:200]}"
                    for msg in history[-5:]
                ]
            )

            company_info = f"\nТЕКУЩАЯ КОМПАНИЯ ПОЛЬЗОВАТЕЛЯ: {company_name}" if company_name else ""

            prompt = f"""Ты — эксперт по анализу поисковых намерений в ГК ТЭМПО.
Твоя задача: превратить вопрос пользователя в ОДИН идеальный поисковый запрос.

ПРАВИЛА:
1. СОХРАНЯЙ ФОКУС: Если пользователь спрашивает о конкретной вещи (адрес, телефон, ФИО), поисковый запрос должен быть сфокусирован ИМЕННО на этом. 
2. УМНОЕ РАСШИРЕНИЕ: Добавляй уточняющие слова (адрес, телефон, график) только если они помогают найти ПРЯМОЙ ответ на вопрос. 
3. НЕ ПОДМЕНЯЙ ТЕМУ: Если спросили "где АБК-2", не нужно искать "как устроиться на работу". 
4. ОСТОРОЖНОСТЬ С КОНТЕКСТОМ: Если новый вопрос очень общий (например, "Что мне делать?"), проверь, не является ли он резкой сменой темы. Если в самом вопросе есть сильное ключевое слово (травма, пожар, адрес), ИГНОРИРУЙ предыдущий контекст про отдел кадров.
5. ФОРМАТ: Только текст запроса.

ПРИМЕРЫ:
Вопрос: "где абк-2?" -> Запрос: "адрес местоположение АБК-2 ГК ТЭМПО как добраться ссылка на карту"
Вопрос: "хочу на работу" -> Запрос: "трудоустройство в ГК ТЭМПО отдел кадров контакты документы адрес"

ДИАЛОГ:
{recent_history}

НОВЫЙ ВОПРОС: {query}
ПОИСКОВЫЙ ЗАПРОС:"""

            rewritten = await self.generate(prompt, temperature=0.0)
            clean_rewritten = rewritten.strip().replace('"', "").replace("'", "")

            # Валидация: если результат подозрительно короткий или пустой — возвращаем оригинал
            if len(clean_rewritten) > 2:
                logger.info(f"Query de-contextualized: '{query}' -> '{clean_rewritten}'")
                return clean_rewritten
            return query

        except Exception as e:
            logger.warning(f"Failed to decontextualize: {e}")
            return query

    def _prepare_context(self, chunks: List[str]) -> str:
        """Подготовка контекста с ограничением размера."""
        if not chunks:
            return "Нет доступной информации в базе знаний."

        max_chars = self.model_config.max_context_chars
        truncated = self._truncate_chunks(chunks, max_chars)
        return "\n\n".join(truncated)

    def _get_system_prompt(self, context: str) -> str:
        """System prompt from model_config with time information."""
        from src.utils.time_utils import format_time_for_prompt, get_current_time_info

        # Get current time information
        time_info = get_current_time_info()
        formatted_time = format_time_for_prompt(time_info)

        # Format the prompt with both time and context
        return self.model_config.system_prompt_template.format(
            time_info=formatted_time,
            context=context
        )

    def _get_full_model_name(self, model_name: str) -> str:
        """Формирует полное имя модели для API."""
        if not model_name.startswith("models/"):
            return f"models/{model_name}"
        return model_name

    async def generate_structured(
        self, 
        prompt: str, 
        response_schema: Any, 
        model_override: str = None, 
        temperature: float = 0.0,
        api_version: str = "v1beta"
    ) -> Any:
        """
        Генерация структурированного ответа с использованием Pydantic схемы.
        Поддерживает автоматическую ротацию ключей и повторы.
        """
        try:
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_schema,
            )

            target_model = model_override or self.config.text_model
            
            response, _ = await self._retry_with_fallback(
                lambda model, api_ver, api_key: self._execute_api_call(prompt, generation_config, model, api_ver, api_key),
                "structured generation",
                initial_model=target_model,
                initial_api_version=api_version
            )
            
            if hasattr(response, "parsed") and response.parsed:
                return response.parsed
                
            # Ручной парсинг если SDK не справился
            import json
            return response_schema(**json.loads(response.text))
            
        except Exception as e:
            logger.exception(f"Structured generation failed: {e}")
            raise LLMError(f"Structured generation failed: {e}") from e

    async def generate(self, prompt: str, model_override: str = None, temperature: float = None) -> str:
        """Генерация текста по произвольному промпту."""
        try:
            generation_config = types.GenerateContentConfig(
                temperature=self.model_config.temperature if temperature is None else temperature,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
            )

            # Используем локальную модель если задан override, иначе из конфига
            target_model = model_override or self.config.text_model
            target_api_version = self.model_fallback_manager.get_api_version_for_model(target_model) if self.model_fallback_manager else self.config.text_api_version

            # Для generate напрямую вызываем версию с параметрами, чтобы избежать ротации в конфиге
            # Но мы можем использовать _retry_with_fallback, если обновим его поддержку начальной модели.
            # Пока сделаем проще: прямой вызов через _execute_api_call с ручным retry если нужно,
            # или просто обновим _call_api.

            response, _ = await self._retry_with_fallback(
                lambda model, api_version, api_key: self._execute_api_call(prompt, generation_config, model, api_version, api_key),
                "text generation",
                initial_model=target_model,
                initial_api_version=target_api_version
            )
            return response.text
        except Exception as e:
            raise LLMError(f"Generation failed: {e}") from e

    async def summarize(self, messages: List[Dict[str, any]]) -> str:
        """
        Генерация краткого резюме разговора.

        Args:
            messages: Список сообщений [{"role": "...", "content": "..."}]

        Returns:
            str: Краткое резюме (3-5 пунктов)
        """
        try:
            conversation_text = "\n".join(
                [f"{'Пользователь' if msg['role'] == 'user' else 'Ассистент'}: {msg['content']}" for msg in messages]
            )
            summarize_prompt = (
                "Ты должен создать краткое резюме следующего разговора.\n"
                "Сохрани ключевые факты, темы и важные детали.\n"
                "Формат: 3-5 пунктов, каждый на новой строке.\n\n"
                f"РАЗГОВОР:\n{conversation_text}\n\nРЕЗЮМЕ:"
            )
            generation_config = types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                safety_settings=self._get_safety_settings() or None,
            )
            response, _ = await self._call_api(summarize_prompt, generation_config, "summarization")
            return response.text.strip()
        except Exception as e:
            raise LLMError(f"Summarization error: {e}") from e

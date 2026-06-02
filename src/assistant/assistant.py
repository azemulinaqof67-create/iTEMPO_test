"""
Единый сервис ассистента.

Объединяет RAG + LLM в единую точку входа.
Устраняет дубликацию из bot.py и server.py.
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from cachetools import TTLCache

from src.core.config import Config
from src.core.document_sender import DocumentSender
from src.core.exceptions import AssistantError, AudioError, LLMError, SearchError
from src.helpdesk.ticketing import (
    TicketContext,
    TicketCreationResult,
    TicketingService,
)
from src.helpdesk.ticketing_logic import should_offer_ticket
from src.llm.audio import AudioLLMService
from src.llm.text import TextLLMService
from src.rag.retrieval.search import SearchResult, SearchService
from src.storage.chat_history import ChatHistoryManager
from src.tools.contact_search import ContactSearchTool
from src.utils.request_logger import RequestLogger
from src.services.weather_service import get_weather

logger = logging.getLogger(__name__)

from src.core.constants import COMPANIES as COMPANY_NAMES


class AssistantService:
    """
    ЕДИНАЯ ТОЧКА ВХОДА для всех запросов к ассистенту.

    Устраняет дубликацию из bot.py и server.py.
    Полностью async архитектура.
    """

    def __init__(self, config: Config):
        self.config = config
        from src.agents.orchestrator import AgentOrchestrator
        self.orchestrator = AgentOrchestrator(config)
        
        self.search = SearchService(config) # Остается для обратной совместимости
        self.text_llm = TextLLMService(config)
        self.audio_llm = AudioLLMService(config)
        self.contact_search = ContactSearchTool()
        self.ticketing = TicketingService(config)
        self.chat_history = ChatHistoryManager(config) if config.chat_history_enabled else None
        self.document_sender = DocumentSender()
        
        # Request logger for detailed logging
        self.request_logger = RequestLogger()
        
        # Session locks for concurrent request management
        self._session_locks = TTLCache(maxsize=10000, ttl=3600)
        self._locks_lock = asyncio.Lock()
        
        # Active requests tracking
        self._active_requests = {}  # session_id -> timestamp
        self._active_requests_lock = asyncio.Lock()
        self._request_timeout = 300  # 5 minutes TTL for active requests

    async def initialize(self):
        """Прогрев всех тяжелых индексов (BM25, FuzzyMatcher)."""
        logger.info("--- INITIALIZING ASSISTANT SERVICE ---")
        await self.orchestrator.initialize()
        await self.search.initialize()
        logger.info("--- ASSISTANT SERVICE READY ---")

    def reload_services(self):
        """
        Горячая перезагрузка всех сервисов при изменении конфигурации.
        """
        logger.info("Reloading assistant services...")
        from src.agents.orchestrator import AgentOrchestrator
        self.orchestrator = AgentOrchestrator(self.config)

        # Пересоздаем LLM сервисы
        self.text_llm = TextLLMService(self.config)
        self.audio_llm = AudioLLMService(self.config)

        # Принудительная перезагрузка конфигурации в ClientManager
        from src.core.clients import ClientManager
        ClientManager.reload_config(self.config)

        logger.info("Services reloaded successfully")

    async def is_request_active(self, session_id: Optional[str]) -> bool:
        if not session_id: return False
        async with self._active_requests_lock:
            if session_id in self._active_requests:
                timestamp = self._active_requests[session_id]
                if time.time() - timestamp < self._request_timeout:
                    return True
                else:
                    del self._active_requests[session_id]
            return False

    async def set_request_active(self, session_id: Optional[str]):
        if not session_id: return
        async with self._active_requests_lock:
            self._active_requests[session_id] = time.time()

    async def clear_request_active(self, session_id: Optional[str]):
        if not session_id: return
        async with self._active_requests_lock:
            self._active_requests.pop(session_id, None)

    async def process_text_query(
        self,
        query: str,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        platform: str = "api",
        user_name: Optional[str] = None,
        user_company: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Публичный метод для обработки текстовых запросов с блокировкой сессии.
        """
        if await self.is_request_active(session_id):
            raise AssistantError("Пожалуйста, подождите, я еще обрабатываю ваш предыдущий запрос...")

        await self.set_request_active(session_id)
        try:
            return await self._process_text_query_impl(query, limit, session_id, platform, user_name, user_company)
        finally:
            await self.clear_request_active(session_id)

    async def _process_text_query_impl(
        self,
        query: str,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        platform: str = "api",
        user_name: Optional[str] = None,
        user_company: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            start_time = time.perf_counter()
            logger.info(f"--- AGENTIC PROCESSOR START: {query[:50]}... ---")

            # 1. Запуск графа (Оркестратор сам разберется с историей через thread_id)
            thread_id = session_id or "default_thread"
            result_state = await self.orchestrator.process_query(query, thread_id=thread_id, user_company=user_company)
            
            answer = result_state.get("answer", "Извините, не удалось получить ответ.")
            # Санитизация для корректного отображения в Telegram (HTML)
            answer = self._sanitize_response(answer)
            
            # Извлекаем контекст для метаданных/логов
            clean_context = []
            raw_results = result_state.get("search_results", "")
            if isinstance(raw_results, str):
                clean_context = [raw_results]
            elif isinstance(raw_results, list):
                clean_context = raw_results

            model_name = result_state.get("model", "gemini-2.0-flash")
            total_duration = time.perf_counter() - start_time
            logger.info(f"✅ [AGENT SUCCESS] Answer generated in {total_duration:.2f}s")

            # 2. Дополнительные документы (схемы, PDF), если есть в базе
            documents_to_send = []
            if getattr(self.config, "enable_document_sender", True):
                resolved_query = result_state.get("query", query)
                # Ищем документы и по оригинальному, и по разрешенному запросу (без дубликатов)
                docs_original = self.document_sender.find_documents(query)
                docs_resolved = self.document_sender.find_documents(resolved_query)
                
                # Объединяем списки, сохраняя уникальность
                seen_paths = set()
                for doc in docs_original + docs_resolved:
                    if doc.document_path not in seen_paths:
                        # Проверка физического наличия файла
                        if self.document_sender.document_exists(doc.document_path):
                            documents_to_send.append(doc)
                            seen_paths.add(doc.document_path)
                        else:
                            logger.warning(f"Document rule triggered but file NOT found: {doc.document_path}")

            # 3. Сохранение в постоянную БД истории (для логов и веб-панели)
            if self.chat_history and session_id:
                metadata = {"rag_used": bool(clean_context), "platform": platform, "agentic": True}
                await self.chat_history.save_message(session_id, platform, "user", query, metadata)
                await self.chat_history.save_message(session_id, platform, "assistant", answer, metadata)

            return {
                "answer": answer,
                "model": model_name,
                "context": clean_context,
                "raw_results": raw_results,
                "documents": [], # AgentState пока не хранит объекты документов отдельно
                "documents_to_send": documents_to_send,
                "retrieval_status": "success" if clean_context else "not_found",
                "resolution_status": "unknown",
                "ticket_offer_available": should_offer_ticket(answer),
                "ticket_created": False,
                "ticket_number": None,
                "ticket_draft_saved": False,
                "ticket_creation_reason": None,
            }

        except Exception as e:
            logger.exception("Unexpected error in agentic processor")
            raise AssistantError(f"Внутренняя ошибка ассистента: {e}") from e

        except (SearchError, LLMError) as e:
            logger.error(f"Error: {e}")
            raise AssistantError(str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error")
            raise AssistantError(f"Внутренняя ошибка: {e}") from e

    async def process_text_query_stream(self, *args, **kwargs):
        # Legacy placeholder or stream implementation
        async for chunk in self._process_text_query_stream_impl(*args, **kwargs):
            yield chunk

    async def _process_text_query_stream_impl(
        self,
        query: str,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        platform: str = "api",
        user_name: Optional[str] = None,
    ):
        try:
            logger.info(f"Processing text query stream: {query[:50]}...")
            history_messages = []
            summary = None
            user_company = None
            if self.chat_history and session_id:
                user_company = await self.chat_history.get_user_company(session_id)
                history_messages = await self.chat_history.get_history(session_id)
                summary_stats = await self.chat_history.get_summary_stats(session_id)
                summary = summary_stats["summary"] if summary_stats else None
                summarized_count = summary_stats["messages_count"] if summary_stats else 0

                if await self.chat_history.check_summarization_needed(session_id):
                    old_messages = await self.chat_history.get_old_messages_for_summarization(
                        session_id, 
                        keep_recent=self.config.memory.max_history_messages,
                        offset=summarized_count
                    )
                    if old_messages:
                        new_delta_summary = await self.text_llm.summarize(old_messages)
                        if new_delta_summary:
                            combined_summary = f"{summary}\n\n{new_delta_summary}" if summary else new_delta_summary
                            await self.chat_history.save_summary(session_id, combined_summary, summarized_count + len(old_messages))
                            summary = combined_summary
            
            search_query = query
            company_display_name = COMPANY_NAMES.get(user_company, user_company) if user_company else None
            
            if history_messages:
                search_query = await self.text_llm.decontextualize_query(query, history_messages, company_name=company_display_name)
            elif company_display_name and len(query) > 3:
                search_query = await self.text_llm.decontextualize_query(query, [], company_name=company_display_name)

            result: SearchResult = await self.search.search(search_query, limit, company_id=user_company)
            clean_context = self.search.clean_scores(result.chunks)

            history = None
            if history_messages:
                history = self.chat_history.format_history_for_llm(history_messages, summary)
                history = self.chat_history.truncate_history_by_tokens(history, self.config.max_context_tokens)

            enriched_query = query
            if user_name:
                enriched_query = f"[Текущий пользователь: {user_name}]\n\nВопрос: {query}"

            full_answer = ""
            async for chunk in self.text_llm.query_stream(clean_context, enriched_query, history=history):
                full_answer += chunk
                yield chunk

            if self.chat_history and session_id:
                metadata = {"rag_used": bool(clean_context), "platform": platform, "streaming": True}
                await self.chat_history.save_message(session_id, platform, "user", query, metadata)
                await self.chat_history.save_message(session_id, platform, "assistant", full_answer, metadata)

        except Exception as e:
            logger.exception("Unexpected streaming error")
            raise AssistantError(f"Внутренняя ошибка при стриминге: {e}") from e

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Перевод аудио (ogg/mp3) в текст с использованием GEMINI_AUDIO_MODEL.
        Не генерирует голосовой ответ, только текст.
        """
        try:
            pcm_data = await self.audio_llm._convert_ogg_to_pcm(audio_bytes)
            transcript = await self.audio_llm.transcribe_audio_from_pcm(pcm_data)
            return self._sanitize_response(transcript)
        except Exception as e:
            logger.exception("Error during audio transcription")
            raise AssistantError(f"Не удалось распознать голосовое сообщение: {e}") from e


    async def process_voice_query(
        self,
        ogg_bytes: bytes,
        system_prompt: Optional[str] = None,
        use_rag: bool = True,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        platform: str = "api",
        audio_format: str = "ogg",
    ) -> tuple[bytes, Optional[str], List[str]]:
        start_time = time.perf_counter()
        user_id = session_id or "unknown"
        if await self.is_request_active(session_id):
            raise AssistantError("Please wait for the response to your previous message...")
        await self.set_request_active(session_id)
        try:
            result = await self._process_voice_query_impl(ogg_bytes, system_prompt, use_rag, limit, session_id, platform, audio_format)
            return result
        finally:
            await self.clear_request_active(session_id)

    async def _process_voice_query_impl(
        self,
        ogg_bytes: bytes,
        system_prompt: Optional[str] = None,
        use_rag: bool = True,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        platform: str = "api",
        audio_format: str = "ogg",
    ) -> tuple[bytes, Optional[str], List[str]]:
        start_time = time.perf_counter()
        history_context = ""
        user_company = None
        if self.chat_history and session_id:
            user_company = await self.chat_history.get_user_company(session_id)
            summary = await self.chat_history.get_summary(session_id)
            history_messages = await self.chat_history.get_history(session_id)
            if summary: history_context = f"\n\nРЕЗЮМЕ ПРЕДЫДУЩЕГО РАЗГОВОРА:\n{summary}\n"
            if history_messages:
                recent = "\n".join([f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}" for m in history_messages[-5:]])
                history_context += f"\nПОСЛЕДНИЕ СООБЩЕНИЯ:\n{recent}\n"

        pcm_data = await self.audio_llm._convert_ogg_to_pcm(ogg_bytes)
        extracted_links = set()
        
        current_system_prompt = system_prompt
        if current_system_prompt is None:
            # Маппинг для промпта
            company_display_name = COMPANY_NAMES.get(user_company, user_company) if user_company else "ГК ТЭМПО"
            
            if use_rag:
                current_system_prompt = f"ТЕКУЩАЯ КОМПАНИЯ: {company_display_name}\nКОНТЕКСТ ДИАЛОГА:\n{history_context}\n\nИНСТРУКЦИИ:\n{self.audio_llm.model_config.tool_prompt_template}"
            else:
                current_system_prompt = (history_context or "") + self.audio_llm.model_config.system_prompt_template.format(context="Контекст не предоставлен.")

        if use_rag:
            all_search_context = []
            async def search_callback(query: str) -> List[str]:
                nonlocal all_search_context
                search_result: SearchResult = await self.search.search(query, self.config.rag.search_limit or 10, company_id=user_company)
                # Сохраняем оригинал (со ссылками) для формирования кнопок в Telegram
                all_search_context.extend(search_result.chunks)
                
                # Вырезаем ссылки из текста, который видит LLM, чтобы она не могла их зачитать
                # Используем заглушку, чтобы ИИ знал о наличии ссылки, но не видел URL
                clean_chunks = [re.sub(r'https?://\S+', '[ссылка прикреплена в чате]', chunk) for chunk in search_result.chunks]
                
                return self._truncate_context_by_chunks(SearchService.clean_scores(clean_chunks), self.audio_llm.model_config.max_voice_context_chars)

            # Коллбэк для поиска контактов через голосовой режим
            async def contact_callback(query: str) -> str:
                return await self.contact_search.search(query, target_company=user_company)
 
            audio_response, transcript, user_query = await self.audio_llm.process_voice_with_tools(
                pcm_data,
                search_callback=search_callback,
                system_prompt=current_system_prompt,
                format=audio_format,
                contact_callback=contact_callback,
            )
            if transcript:
                extracted_links.update(self._extract_links(transcript))
                # Если в ответе упоминается ссылка, ищем её во ВСЕМ накопленном контексте этого хода
                if any(w in transcript.lower() for w in ["ссылк", "маршрут", "карт", "локаци"]):
                    logger.info(f"DEBUG: All search context chunks seen in this turn ({len(all_search_context)}):")
                    for idx, chunk in enumerate(all_search_context):
                        logger.info(f"  Chunk {idx}: {chunk[:100]}...")
                    
                    for chunk in all_search_context:
                        extracted_links.update(self._extract_links(chunk))
            
            if self.chat_history and session_id:
                metadata = {"rag_used": True, "platform": platform, "voice": True}
                await self.chat_history.save_message(session_id, platform, "user", user_query or "[Голос]", metadata)
                await self.chat_history.save_message(session_id, platform, "assistant", transcript or "[Голос]", metadata)
            
            return audio_response, self._sanitize_response(transcript), list(extracted_links)
        else:
            audio_response, transcript = await self.audio_llm.process_voice_from_pcm(pcm_data, current_system_prompt, format=audio_format)
            if self.chat_history and session_id:
                await self.chat_history.save_message(session_id, platform, "user", "[Голос]", {"voice": True})
                await self.chat_history.save_message(session_id, platform, "assistant", transcript or "[Голос]", {"voice": True})
            return audio_response, self._sanitize_response(transcript), list(extracted_links)

    def _truncate_context_by_chunks(self, chunks: List[str], max_chars: int) -> List[str]:
        result = []
        total = 0
        for chunk in chunks:
            if total + len(chunk) > max_chars and result: break
            result.append(chunk)
            total += len(chunk)
        return result

    async def create_helpdesk_ticket(self, query: str, assistant_answer: str, session_id: Optional[str] = None, **kwargs) -> TicketCreationResult:
        summary = None
        try:
            history_messages = await self.chat_history.get_history(session_id) if self.chat_history and session_id else []
            dialog = "\n".join([f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}" for m in history_messages[-10:]]) if history_messages else f"U: {query}\nA: {assistant_answer}"
            summary = await self.text_llm.generate(f"Кратко опиши суть проблемы для тикета на основе диалога:\n{dialog}", temperature=0.1)
        except: pass
        ctx = TicketContext(query=query, assistant_answer=assistant_answer, summary=summary, **kwargs)
        return await self.ticketing.create_ticket(ctx)

    def _sanitize_response(self, text: str) -> str:
        if not text: return ""
        
        # Удаляем блоки рассуждений <thought>...</thought>
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 1. Заменяем структурные HTML-теги, которые не поддерживает Telegram
        # Преобразуем li в списки с маркером
        text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        # Удаляем контейнеры списков ul и ol
        text = re.sub(r'</?(?:ul|ol)[^>]*>', '', text, flags=re.IGNORECASE)
        # Абзацы p заменяем на переносы строк
        text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL | re.IGNORECASE)
        # Заголовки h1-h6 преобразуем в жирный текст
        text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'<b>\1</b>\n\n', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Обрабатываем простые таблицы: строки разделяем переносами, ячейки td/th - пробелами
        text = re.sub(r'<tr[^>]*>(.*?)</tr>', r'\1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<t[dh][^>]*>(.*?)</t[dh]>', r' \1 ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'</?(?:table|thead|tbody|tfoot)[^>]*>', '', text, flags=re.IGNORECASE)
        
        # Удаляем контейнеры div и span
        text = re.sub(r'</?(?:div|span)[^>]*>', '', text, flags=re.IGNORECASE)

        # 2. Обрабатываем markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        # Очистка wiki-ссылок вида [[Текст|ссылка]] -> Текст
        text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\1', text)
        # Очистка wiki-ссылок вида [[Текст]] -> Текст
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        
        # Устраняем лишние (3+) подряд идущие переносы строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _extract_links(self, text: str) -> List[str]:
        if not text: return []
        URL_PATTERN = r"(https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*)"
        found = re.findall(URL_PATTERN, text)
        return [l.rstrip(".,!?;:)\u2026\u00bb\u00ab\"'") for l in found if len(l) >= 11]

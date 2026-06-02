"""
Async голосовой LLM через Gemini Live API.

ИСПРАВЛЕНО: Прокси через ClientManager.
"""

import asyncio
import io
import logging
from typing import Awaitable, Callable, List, Optional

from google.genai import types
from pydub import AudioSegment

from src.core.clients import ClientManager
from src.core.config import Config
from src.core.exceptions import AudioError
from src.core.models_loader import AudioModelConfig

logger = logging.getLogger(__name__)

# Function calling tool для RAG-поиска
SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": "Поиск в корпоративной базе знаний TEMPO. Используй эту функцию для ответов на вопросы о компании, процессах, расположении офисов, правилах и документах.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Поисковый запрос на основе вопроса пользователя.",
            }
        },
        "required": ["query"],
    },
}

# Function calling tool для поиска контактов и телефонов
CONTACT_TOOL = {
    "name": "search_contacts",
    "description": "Поиск сотрудников, их телефонных номеров и должностей в корпоративной базе контактов. Используй ЭТУ функцию, когда нужно найти: телефон конкретного человека, должность сотрудника, в каком отделе работает человек.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Имя сотрудника, фамилия или должность для поиска. Например: 'Шириева', 'помощник директора', 'бухгалтер Технотрон'.",
            }
        },
        "required": ["query"],
    },
}

# Настройка ffmpeg для pydub
try:
    from static_ffmpeg import add_paths

    add_paths()
    logger.info("FFmpeg added to PATH")
except Exception as e:
    import shutil
    if shutil.which("ffmpeg"):
        logger.warning(f"static-ffmpeg initialization failed ({e}), but system ffmpeg is available in PATH.")
    else:
        logger.error(f"Failed to configure FFmpeg: {e}")
        raise AudioError(f"FFmpeg initialization failed: {e}") from e


class AudioLLMService:
    """
    Async голосовой LLM через Gemini Live API.

    ИСПРАВЛЕНО: Прокси через ClientManager.
    """

    def __init__(self, config: Config, model_config: AudioModelConfig = None):
        self.config = config
        self.client_manager = ClientManager.get_instance(config)
        # Используем конфиг из YAML если не передан явно
        self.model_config = model_config or config.audio_model_config

    async def transcribe_audio(self, ogg_bytes: bytes) -> str:
        """
        Транскрибирует аудио в текст через Gemini Live API.

        Args:
            ogg_bytes: OGG/Opus аудио

        Returns:
            str: Транскрибированный текст
        """
        pcm_data = await self._convert_ogg_to_pcm(ogg_bytes)
        return await self.transcribe_audio_from_pcm(pcm_data)

    async def transcribe_audio_from_pcm(self, pcm_data: bytes) -> str:
        """
        Транскрибирует PCM аудио в текст через Gemini Live API.

        Args:
            pcm_data: PCM аудио данные

        Returns:
            str: Транскрибированный текст
        """
        try:
            client = self.client_manager.get_gemini_client_for_live_api()
            model_name = self._get_full_model_name(self.config.audio_model)

            # Конфигурация с транскрипцией (требуется AUDIO модальность)
            # VAD отключен для поддержки явного управления активностью
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                input_audio_transcription={},
                enable_affective_dialog=self.model_config.enable_affective_dialog,
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
                ),
            )

            logger.info(f"Transcribing audio via {model_name}")

            transcript_text = ""

            async def _receive_transcription():
                nonlocal transcript_text
                async with client.aio.live.connect(model=model_name, config=config) as session:
                    # Activity markers для консистентности с _call_live_api
                    await session.send_realtime_input(activity_start=types.ActivityStart())

                    await session.send_realtime_input(
                        audio=types.Blob(
                            mime_type=f"audio/pcm;rate={self.model_config.input_sample_rate}",
                            data=pcm_data,
                        )
                    )

                    await session.send_realtime_input(activity_end=types.ActivityEnd())

                    # Получаем транскрипцию
                    async for msg in session.receive():
                        if msg.server_content:
                            # Транскрипция входного аудио
                            if msg.server_content.input_transcription:
                                chunk = msg.server_content.input_transcription.text
                                if chunk:
                                    transcript_text += chunk
                                    logger.info(f"Transcription chunk: {chunk}")

                            # Завершение
                            if msg.server_content.turn_complete:
                                break

            try:
                await asyncio.wait_for(
                    _receive_transcription(),
                    timeout=self.model_config.transcription_timeout,
                )
            except asyncio.TimeoutError as err:
                logger.error(f"Transcription timeout ({self.model_config.transcription_timeout}s)")
                raise AudioError(
                    f"Транскрипция не завершилась за {self.model_config.transcription_timeout} секунд"
                ) from err

            if not transcript_text:
                raise AudioError("Failed to transcribe audio")

            return transcript_text

        except AudioError:
            raise
        except Exception as e:
            logger.exception("Transcription error")
            raise AudioError(f"Transcription failed: {e}") from e

    async def process_voice(self, ogg_bytes: bytes, system_prompt: Optional[str] = None) -> tuple[bytes, Optional[str]]:
        """
        Обработка голосового сообщения.

        Args:
            ogg_bytes: OGG/Opus (Telegram формат)
            system_prompt: Системный промпт

        Returns:
            tuple[bytes, Optional[str]]: (OGG/Opus ответ, транскрипция)
        """
        pcm_data = await self._convert_ogg_to_pcm(ogg_bytes)
        return await self.process_voice_from_pcm(pcm_data, system_prompt)

    async def process_voice_from_pcm(self, pcm_data: bytes, system_prompt: Optional[str] = None, format: str = "ogg") -> tuple[bytes, Optional[str]]:
        """
        Обработка голосового сообщения из PCM данных.
        """
        try:
            # 1. Live API
            response_pcm, transcript = await self._call_live_api(pcm_data, system_prompt)

            # 2. PCM → Audio (OGG, MP3 or WAV)
            if format == "mp3":
                audio_response = await self._convert_pcm_to_mp3(response_pcm)
            elif format == "wav":
                audio_response = await self._convert_pcm_to_wav(response_pcm)
            else:
                audio_response = await self._convert_pcm_to_ogg(response_pcm)

            return audio_response, transcript

        except AudioError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in voice processing")
            raise AudioError(f"Voice processing failed: {e}") from e

    async def process_voice_with_tools(
        self,
        pcm_data: bytes,
        search_callback: Callable[[str], Awaitable[List[str]]],
        system_prompt: str,
        format: str = "ogg",
        contact_callback: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> tuple[bytes, Optional[str], Optional[str]]:
        """
        Обработка голосового сообщения с Function Calling для RAG.

        Использует одно WebSocket-соединение вместо двух:
        - Модель сама решает когда вызвать search_knowledge_base
        - Получаем транскрипцию ответа для логирования

        Args:
            pcm_data: PCM аудио данные
            search_callback: Async функция для RAG-поиска
            system_prompt: Системный промпт

        Returns:
            tuple[bytes, Optional[str], Optional[str]]: (OGG ответ, транскрипция ответа, запрос пользователя)
        """
        _MAX_RETRIES = 3
        _RETRY_DELAY = 1.5  # секунд между попытками

        # Сетевые ошибки, которые имеет смысл повторять
        _NETWORK_ERRORS = (ConnectionResetError, ConnectionAbortedError, OSError)

        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await self._process_voice_with_tools_once(
                    pcm_data, search_callback, system_prompt, format=format,
                    contact_callback=contact_callback,
                )
            except AudioError:
                raise  # Логические ошибки (нет аудио и т.д.) — не повторяем
            except asyncio.TimeoutError as e:
                last_error = e
                logger.error(
                    "Live API timeout на попытке %d/%d", attempt, _MAX_RETRIES
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_DELAY)
            except _NETWORK_ERRORS as e:
                last_error = e
                logger.warning(
                    "Сетевая ошибка на попытке %d/%d: %s. Повтор через %.1f с...",
                    attempt, _MAX_RETRIES, e, _RETRY_DELAY,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_DELAY)
            except Exception as e:
                # Неизвестные ошибки — не повторяем
                logger.exception("Unexpected error in voice processing with tools")
                raise AudioError(f"Voice processing with tools failed: {e}") from e

        raise AudioError(
            f"Voice processing with tools failed after {_MAX_RETRIES} attempts: {last_error}"
        )

    async def _process_voice_with_tools_once(
        self,
        pcm_data: bytes,
        search_callback: Callable[[str], Awaitable[List[str]]],
        system_prompt: str,
        format: str = "ogg",
        contact_callback: Optional[Callable[[str], Awaitable[str]]] = None,
    ) -> tuple[bytes, Optional[str], Optional[str]]:
        """Один проход обработки голосового сообщения (без retry)."""
        try:
            client = self.client_manager.get_gemini_client_for_live_api()
            model_name = self._get_full_model_name(self.config.audio_model)

            # Конфигурация с tools
            tool_declarations = [SEARCH_TOOL]
            if contact_callback is not None:
                tool_declarations.append(CONTACT_TOOL)
            tools = [{"function_declarations": tool_declarations}] if self.model_config.enable_tools else []

            config = types.LiveConnectConfig(
                response_modalities=self.model_config.response_modalities,
                input_audio_transcription={},
                output_audio_transcription={} if self.model_config.enable_output_transcription else None,
                tools=tools,
                system_instruction=system_prompt,
                enable_affective_dialog=self.model_config.enable_affective_dialog,
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.model_config.voice_name
                        )
                    ),
                    language_code="ru-RU",
                ),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        disabled=self.model_config.vad_disabled
                    )
                ),
                generation_config=types.GenerationConfig(
                    temperature=self.model_config.temperature,
                    top_p=self.model_config.top_p,
                    top_k=self.model_config.top_k,
                    max_output_tokens=self.model_config.max_output_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=self.model_config.thinking_budget),
                ),
            )

            logger.info(f"Connecting to Live API with tools: {model_name}")

            async with client.aio.live.connect(model=model_name, config=config) as session:
                logger.info("Connected. Sending audio...")

                # Отправка аудио
                await session.send_realtime_input(activity_start=types.ActivityStart())
                await session.send_realtime_input(
                    audio=types.Blob(
                        mime_type=f"audio/pcm;rate={self.model_config.input_sample_rate}",
                        data=pcm_data,
                    )
                )
                await session.send_realtime_input(activity_end=types.ActivityEnd())

                logger.info(f"Sent {len(pcm_data)} bytes, waiting for response...")

                # Сбор ответа
                chunks = []
                output_transcript = None
                input_transcript = None
                captured_user_query = None

                async def _receive():
                    nonlocal output_transcript, input_transcript, captured_user_query
                    async for response in session.receive():
                        # Логируем все типы сообщений на INFO уровне для отладки
                        if response.server_content:
                            content_type = []
                            if response.server_content.input_transcription:
                                content_type.append("input_transcription")
                            if response.server_content.output_transcription:
                                content_type.append("output_transcription")
                            if response.server_content.model_turn:
                                content_type.append("model_turn")
                            if response.server_content.turn_complete:
                                content_type.append("turn_complete")
                            if response.server_content.interrupted:
                                content_type.append("interrupted")
                            if content_type:
                                logger.debug(f"Event: {', '.join(content_type)}")

                        if response.tool_call:
                            logger.debug("Event: tool_call (processing...)")

                        # Обработка tool_call (ДО проверки server_content!)
                        if response.tool_call:
                            logger.info(f"Model requested tool call: {response.tool_call}")
                            function_responses = []

                            for fc in response.tool_call.function_calls:
                                if fc.name == "search_knowledge_base":
                                    query = fc.args.get("query", "")
                                    logger.info(f"Searching knowledge base: {query}")
                                    captured_user_query = query

                                    try:
                                        results = await search_callback(query)
                                        context = "\n\n".join(results) if results else "Информация не найдена"

                                        function_responses.append(
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"context": context},
                                            )
                                        )
                                        logger.info(f"Found {len(results)} results, sending to model...")
                                    except Exception as e:
                                        logger.error(f"Search failed: {e}")
                                        function_responses.append(
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"context": "Ошибка поиска"},
                                            )
                                        )

                                elif fc.name == "search_contacts" and contact_callback is not None:
                                    query = fc.args.get("query", "")
                                    logger.info(f"Searching contacts: {query}")

                                    try:
                                        result_text = await contact_callback(query)
                                        function_responses.append(
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"result": result_text},
                                            )
                                        )
                                        logger.info(f"Contact search result: {result_text[:100]}...")
                                    except Exception as e:
                                        logger.error(f"Contact search failed: {e}")
                                        function_responses.append(
                                            types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"result": "Ошибка поиска контактов"},
                                            )
                                        )

                            if function_responses:
                                await session.send_tool_response(function_responses=function_responses)
                                logger.info("Tool response sent, waiting for model response...")

                        # Проверка server_content
                        if not response.server_content:
                            continue

                        # Транскрипция входного аудио
                        if response.server_content.input_transcription:
                            chunk = response.server_content.input_transcription.text
                            if chunk:
                                if input_transcript is None:
                                    input_transcript = ""
                                input_transcript += chunk
                                logger.info(f"Input transcript chunk: {chunk}")

                        # Обработка прерывания
                        if response.server_content.interrupted:
                            logger.warning("Interrupted, clearing chunks")
                            chunks.clear()
                            continue

                        # Сбор аудио
                        if response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data and isinstance(part.inline_data.data, bytes):
                                    chunks.append(part.inline_data.data)

                        # Транскрипция ответа
                        if response.server_content.output_transcription:
                            transcript_chunk = response.server_content.output_transcription.text
                            if output_transcript is None:
                                output_transcript = ""
                            output_transcript += transcript_chunk
                            logger.debug(f"Output transcript chunk: {transcript_chunk[:50]}...")

                        # Завершение
                        if response.server_content.turn_complete:
                            logger.info(f"Turn complete: {len(chunks)} chunks")
                            return

                try:
                    await asyncio.wait_for(_receive(), timeout=self.model_config.response_timeout)
                except asyncio.TimeoutError as err:
                    logger.error(f"API response timeout ({self.model_config.response_timeout}s)")
                    raise AudioError(f"API не ответил за {self.model_config.response_timeout} секунд") from err

            if not chunks:
                raise AudioError("No audio response from API")

            total_bytes = sum(len(c) for c in chunks)
            logger.info(f"Received: {len(chunks)} chunks, {total_bytes} bytes")

            # Конвертация в нужный формат (MP3 для MAX, OGG для остальных)
            response_pcm = b"".join(chunks)
            if format == "mp3":
                final_audio = await self._convert_pcm_to_mp3(response_pcm)
            elif format == "wav":
                final_audio = await self._convert_pcm_to_wav(response_pcm)
            else:
                final_audio = await self._convert_pcm_to_ogg(response_pcm)

            # Определяем, что считать запросом пользователя:
            # ВСЕГДА используем input_transcript (реальные слова пользователя)
            # captured_user_query оставляем только для логов
            final_user_query = input_transcript

            if output_transcript:
                logger.info(f"Final output transcript: {output_transcript[:500]}...")

            return final_audio, output_transcript, final_user_query

        except AudioError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in voice processing with tools")
            raise AudioError(f"Voice processing with tools failed: {e}") from e

    async def _convert_ogg_to_pcm(self, ogg_bytes: bytes) -> bytes:
        """Audio(OGG/Opus/m4a/mp3) → PCM (input_sample_rate, mono, sample_width из model_config)"""
        loop = asyncio.get_event_loop()

        def _convert():
            # FFmpeg отлично определяет формат автоматически по сигнатуре
            try:
                audio = AudioSegment.from_file(io.BytesIO(ogg_bytes))
            except Exception:
                # Если авто-определение сломается, пробуем как OGG (для Telegram)
                audio = AudioSegment.from_file(io.BytesIO(ogg_bytes), format="ogg")
                
            audio = (
                audio.set_frame_rate(self.model_config.input_sample_rate)
                .set_channels(self.model_config.channels)
                .set_sample_width(self.model_config.sample_width)
            )
            return audio.raw_data

        return await loop.run_in_executor(None, _convert)

    def _get_full_model_name(self, model_name: str) -> str:
        """
        Формирует полное имя модели для API.

        Args:
            model_name: Короткое имя модели (gemini-2.5-flash-native-audio-dialog)

        Returns:
            str: Полное имя для API (models/gemini-2.5-flash-native-audio-dialog)
        """
        if not model_name.startswith("models/"):
            return f"models/{model_name}"
        return model_name

    async def _call_live_api(self, pcm_data: bytes, system_prompt: Optional[str]) -> tuple[bytes, Optional[str]]:
        """
        Вызов Gemini Live API для предзаписанного аудио.

        ВАЖНО: Последовательная обработка (send → receive), не параллельная.
        VAD отключен для предзаписанного аудио.
        """
        client = self.client_manager.get_gemini_client_for_live_api()
        # Имя модели из Config (env)
        model_name = self._get_full_model_name(self.config.audio_model)

        # Конфигурация из model_config
        config = types.LiveConnectConfig(
            response_modalities=self.model_config.response_modalities,
            enable_affective_dialog=self.model_config.enable_affective_dialog,
            output_audio_transcription={} if self.model_config.enable_output_transcription else None,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.model_config.voice_name
                    )
                ),
                language_code="ru-RU",
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=self.model_config.vad_disabled)
            ),
            generation_config=types.GenerationConfig(
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
                max_output_tokens=self.model_config.max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=self.model_config.thinking_budget),
            ),
        )

        if system_prompt:
            config.system_instruction = system_prompt

        logger.info(f"Connecting to Live API: {model_name}")

        async with client.aio.live.connect(model=model_name, config=config) as session:
            logger.info("Connected. Sending audio...")

            # 1. ОТПРАВКА: activity_start → аудио → activity_end (для disabled VAD)
            await session.send_realtime_input(activity_start=types.ActivityStart())

            await session.send_realtime_input(
                audio=types.Blob(
                    mime_type=f"audio/pcm;rate={self.model_config.input_sample_rate}",
                    data=pcm_data,
                )
            )

            # Сигнализируем конец активности (при disabled VAD используется activity_end)
            await session.send_realtime_input(activity_end=types.ActivityEnd())

            logger.info(f"Sent {len(pcm_data)} bytes, waiting for response...")

            # 2. ПОЛУЧЕНИЕ: Собираем все чанки до turn_complete (с таймаутом)
            chunks = []
            output_transcript = ""

            async def _receive():
                nonlocal output_transcript
                async for response in session.receive():
                    if not response.server_content:
                        continue

                    if response.server_content.interrupted:
                        logger.warning("Interrupted, clearing chunks")
                        chunks.clear()
                        continue

                    if response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                chunks.append(part.inline_data.data)

                    # Транскрипция ответа
                    if response.server_content.output_transcription:
                        transcript_chunk = response.server_content.output_transcription.text
                        output_transcript += transcript_chunk

                    if response.server_content.turn_complete:
                        logger.info(f"Turn complete: {len(chunks)} chunks")
                        return

            try:
                await asyncio.wait_for(_receive(), timeout=self.model_config.response_timeout)
            except asyncio.TimeoutError as err:
                logger.error(f"API response timeout ({self.model_config.response_timeout}s)")
                raise AudioError(f"API не ответил за {self.model_config.response_timeout} секунд") from err

        if not chunks:
            raise AudioError("No audio response from API")

        total_bytes = sum(len(c) for c in chunks)
        logger.info(f"Received: {len(chunks)} chunks, {total_bytes} bytes")
        return b"".join(chunks), output_transcript or None

    async def _convert_pcm_to_ogg(self, pcm_bytes: bytes) -> bytes:
        """PCM → OGG/Opus (output_sample_rate из model_config)"""
        loop = asyncio.get_event_loop()

        def _convert():
            audio = AudioSegment(
                data=pcm_bytes,
                sample_width=self.model_config.sample_width,
                frame_rate=self.model_config.output_sample_rate,
                channels=self.model_config.channels,
            )
            buffer = io.BytesIO()
            audio.export(
                buffer,
                format="ogg",
                codec="libopus",
                bitrate=self.model_config.ogg_bitrate,
            )
            return buffer.getvalue()

        return await loop.run_in_executor(None, _convert)

    async def _convert_pcm_to_mp3(self, pcm_bytes: bytes) -> bytes:
        """PCM → MP3 (Mono, 16kHz для MAX)"""
        loop = asyncio.get_event_loop()

        def _convert():
            audio = AudioSegment(
                data=pcm_bytes,
                sample_width=self.model_config.sample_width,
                frame_rate=self.model_config.output_sample_rate,
                channels=self.model_config.channels,
            )
            # Форсируем моно и 16кГц для совместимости с MAX/OK
            audio = audio.set_channels(1).set_frame_rate(16000)
            buffer = io.BytesIO()
            audio.export(buffer, format="mp3", bitrate="32k")
            return buffer.getvalue()

        return await loop.run_in_executor(None, _convert)

    async def _convert_pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        """PCM → WAV (без сжатия)"""
        loop = asyncio.get_event_loop()

        def _convert():
            audio = AudioSegment(
                data=pcm_bytes,
                sample_width=self.model_config.sample_width,
                frame_rate=self.model_config.output_sample_rate,
                channels=self.model_config.channels,
            )
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            return buffer.getvalue()

        return await loop.run_in_executor(None, _convert)

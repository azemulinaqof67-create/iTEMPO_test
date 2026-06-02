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
        """
        pcm_data = await self._convert_ogg_to_pcm(ogg_bytes)
        return await self.transcribe_audio_from_pcm(pcm_data)

    async def transcribe_audio_from_pcm(self, pcm_data: bytes) -> str:
        """
        Транскрибирует PCM аудио в текст через Gemini Live API.
        """
        try:
            client = self.client_manager.get_gemini_client_for_live_api()
            model_name = self._get_full_model_name(self.config.audio_model)

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
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                    await session.send_realtime_input(
                        audio=types.Blob(
                            mime_type=f"audio/pcm;rate={self.model_config.input_sample_rate}",
                            data=pcm_data,
                        )
                    )
                    await session.send_realtime_input(activity_end=types.ActivityEnd())

                    async for msg in session.receive():
                        if msg.server_content:
                            if msg.server_content.input_transcription:
                                chunk = msg.server_content.input_transcription.text
                                if chunk:
                                    transcript_text += chunk
                            if msg.server_content.turn_complete:
                                break

            await asyncio.wait_for(_receive_transcription(), timeout=self.model_config.transcription_timeout)
            return transcript_text

        except Exception as e:
            logger.exception("Transcription error")
            raise AudioError(f"Transcription failed: {e}") from e

    async def process_voice(self, ogg_bytes: bytes, system_prompt: Optional[str] = None) -> tuple[bytes, Optional[str]]:
        pcm_data = await self._convert_ogg_to_pcm(ogg_bytes)
        return await self.process_voice_from_pcm(pcm_data, system_prompt)

    async def process_voice_from_pcm(self, pcm_data: bytes, system_prompt: Optional[str] = None, format: str = "ogg") -> tuple[bytes, Optional[str]]:
        try:
            response_pcm, transcript = await self._call_live_api(pcm_data, system_prompt)
            if format == "mp3":
                audio_response = await self._convert_pcm_to_mp3(response_pcm)
            elif format == "wav":
                audio_response = await self._convert_pcm_to_wav(response_pcm)
            else:
                audio_response = await self._convert_pcm_to_ogg(response_pcm)
            return audio_response, transcript
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
        Обработка голосового сообщения с Function Calling для RAG и механизмом RETRY.
        """
        client = self.client_manager.get_gemini_client_for_live_api()
        model_name = self._get_full_model_name(self.config.audio_model)

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
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.model_config.voice_name)
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

        max_retries = 2
        retry_count = 0
        chunks = []
        output_transcript = ""
        input_transcript = ""
        captured_user_query = None

        while retry_count <= max_retries:
            try:
                async with client.aio.live.connect(model=model_name, config=config) as session:
                    logger.info(f"Connected to Live API (Attempt {retry_count + 1})")
                    
                    # Отправка аудио
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                    await session.send_realtime_input(
                        audio=types.Blob(
                            mime_type=f"audio/pcm;rate={self.model_config.input_sample_rate}",
                            data=pcm_data,
                        )
                    )
                    await session.send_realtime_input(activity_end=types.ActivityEnd())

                    # Сброс данных перед получением
                    chunks = []
                    output_transcript = ""
                    input_transcript = ""

                    async def _receive():
                        nonlocal output_transcript, input_transcript, captured_user_query
                        try:
                            async for response in session.receive():
                                if response.server_content:
                                    if response.server_content.input_transcription:
                                        chunk = response.server_content.input_transcription.text
                                        if chunk:
                                            input_transcript += chunk

                                    if response.server_content.interrupted:
                                        chunks.clear()
                                        continue

                                    if response.server_content.model_turn:
                                        for part in response.server_content.model_turn.parts:
                                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                                chunks.append(part.inline_data.data)

                                    if response.server_content.output_transcription:
                                        output_transcript += response.server_content.output_transcription.text

                                    if response.server_content.turn_complete:
                                        return

                                if response.tool_call:
                                    function_responses = []
                                    for fc in response.tool_call.function_calls:
                                        if fc.name == "search_knowledge_base":
                                            query = fc.args.get("query", "")
                                            captured_user_query = query
                                            results = await search_callback(query)
                                            context = "\n\n".join(results) if results else "Информация не найдена"
                                            function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"context": context}))
                                        elif fc.name == "search_contacts" and contact_callback is not None:
                                            query = fc.args.get("query", "")
                                            result_text = await contact_callback(query)
                                            function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": result_text}))

                                    if function_responses:
                                        await session.send_tool_response(function_responses=function_responses)
                        except Exception as e:
                            logger.error(f"Session receive error: {e}")
                            raise

                    await asyncio.wait_for(_receive(), timeout=self.model_config.response_timeout)
                    break # Успешно завершили

            except Exception as e:
                retry_count += 1
                err_msg = str(e)
                if ("1006" in err_msg or "abnormal closure" in err_msg or isinstance(e, asyncio.TimeoutError)) and retry_count <= max_retries:
                    logger.warning(f"Live API connection lost. Retrying {retry_count}/{max_retries}...")
                    await asyncio.sleep(1)
                    continue
                raise AudioError(f"Voice processing with tools failed after {retry_count} attempts: {e}")

        # Финальная обработка аудио
        if not chunks:
            raise AudioError("No audio response from API")

        response_pcm = b"".join(chunks)
        if format == "mp3":
            final_audio = await self._convert_pcm_to_mp3(response_pcm)
        elif format == "wav":
            final_audio = await self._convert_pcm_to_wav(response_pcm)
        else:
            final_audio = await self._convert_pcm_to_ogg(response_pcm)

        return final_audio, output_transcript or None, input_transcript or None

    async def _convert_ogg_to_pcm(self, ogg_bytes: bytes) -> bytes:
        loop = asyncio.get_event_loop()
        def _convert():
            try:
                audio = AudioSegment.from_file(io.BytesIO(ogg_bytes))
            except Exception:
                audio = AudioSegment.from_file(io.BytesIO(ogg_bytes), format="ogg")
            audio = audio.set_frame_rate(self.model_config.input_sample_rate).set_channels(self.model_config.channels).set_sample_width(self.model_config.sample_width)
            return audio.raw_data
        return await loop.run_in_executor(None, _convert)

    def _get_full_model_name(self, model_name: str) -> str:
        if not model_name.startswith("models/"):
            return f"models/{model_name}"
        return model_name

    async def _call_live_api(self, pcm_data: bytes, system_prompt: Optional[str]) -> tuple[bytes, Optional[str]]:
        client = self.client_manager.get_gemini_client_for_live_api()
        model_name = self._get_full_model_name(self.config.audio_model)
        config = types.LiveConnectConfig(
            response_modalities=self.model_config.response_modalities,
            enable_affective_dialog=self.model_config.enable_affective_dialog,
            output_audio_transcription={} if self.model_config.enable_output_transcription else None,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.model_config.voice_name)
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
            ),
            system_instruction=system_prompt,
        )

        async with client.aio.live.connect(model=model_name, config=config) as session:
            await session.send_realtime_input(activity_start=types.ActivityStart())
            await session.send_realtime_input(
                audio=types.Blob(
                    mime_type=f"audio/pcm;rate={self.model_config.input_sample_rate}",
                    data=pcm_data,
                )
            )
            await session.send_realtime_input(activity_end=types.ActivityEnd())

            chunks = []
            output_transcript = ""
            async def _receive():
                nonlocal output_transcript
                async for response in session.receive():
                    if response.server_content:
                        if response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    chunks.append(part.inline_data.data)
                        if response.server_content.output_transcription:
                            output_transcript += response.server_content.output_transcription.text
                        if response.server_content.turn_complete:
                            return

            await asyncio.wait_for(_receive(), timeout=self.model_config.response_timeout)
            return b"".join(chunks), output_transcript or None

    async def _convert_pcm_to_ogg(self, pcm_bytes: bytes) -> bytes:
        loop = asyncio.get_event_loop()
        def _convert():
            audio = AudioSegment(data=pcm_bytes, sample_width=self.model_config.sample_width, frame_rate=self.model_config.output_sample_rate, channels=self.model_config.channels)
            buffer = io.BytesIO()
            audio.export(buffer, format="ogg", codec="libopus", bitrate=self.model_config.ogg_bitrate)
            return buffer.getvalue()
        return await loop.run_in_executor(None, _convert)

    async def _convert_pcm_to_mp3(self, pcm_bytes: bytes) -> bytes:
        loop = asyncio.get_event_loop()
        def _convert():
            audio = AudioSegment(data=pcm_bytes, sample_width=self.model_config.sample_width, frame_rate=self.model_config.output_sample_rate, channels=self.model_config.channels)
            audio = audio.set_channels(1).set_frame_rate(16000)
            buffer = io.BytesIO()
            audio.export(buffer, format="mp3", bitrate="32k")
            return buffer.getvalue()
        return await loop.run_in_executor(None, _convert)

    async def _convert_pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        loop = asyncio.get_event_loop()
        def _convert():
            audio = AudioSegment(data=pcm_bytes, sample_width=self.model_config.sample_width, frame_rate=self.model_config.output_sample_rate, channels=self.model_config.channels)
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            return buffer.getvalue()
        return await loop.run_in_executor(None, _convert)

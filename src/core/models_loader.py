"""
Загрузчик конфигурации моделей из YAML
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.exceptions import ConfigError


@dataclass
class TextModelConfig:
    """Конфигурация текстовой модели"""

    name: Optional[str] = "gemini-3.5-flash"
    api_version: Optional[str] = None
    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 64
    max_context_chars: int = 20000
    disable_safety: bool = True
    system_prompt_template: str = ""
    fallbacks: List[str] = field(default_factory=list)


@dataclass
class AudioModelConfig:
    """Конфигурация голосовой модели"""

    name: Optional[str] = "gemini-2.5-flash-native-audio-latest"
    api_version: str = "v1alpha"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: float = 64.0
    thinking_budget: int = 8192
    max_output_tokens: int = 8192
    voice_name: str = "Kore"
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2
    ogg_bitrate: str = "64k"
    vad_disabled: bool = True
    enable_output_transcription: bool = True
    enable_tools: bool = True
    enable_affective_dialog: bool = False
    response_timeout: float = 120.0
    transcription_timeout: float = 30.0
    max_voice_context_chars: int = 50000
    system_prompt_template: str = ""
    tool_prompt_template: str = ""
    response_modalities: list[str] = field(default_factory=lambda: ["AUDIO"])


@dataclass
class EmbeddingModelConfig:
    """Конфигурация модели эмбеддингов"""

    name: Optional[str] = "gemini-embedding-2"
    api_version: str = "v1beta"


@dataclass
class ContextualRetrievalConfig:
    """Конфигурация контекстуального поиска"""

    model: Optional[str] = "gemini-3.5-flash"  # дефолт из yaml перетрёт это
    max_doc_chars: int = 12000
    parallelism: int = 4


@dataclass
class ModelsConfig:
    """Полная конфигурация всех моделей"""

    text_model: TextModelConfig
    audio_model: AudioModelConfig
    embedding_model: EmbeddingModelConfig
    contextual_retrieval: ContextualRetrievalConfig


def load_models_config(yaml_path: str = "models_config.yaml") -> ModelsConfig:
    """
    Загружает конфигурацию моделей из YAML файла.

    Args:
        yaml_path: Путь к YAML файлу

    Returns:
        ModelsConfig: Конфигурация всех моделей

    Raises:
        ConfigError: Если файл не найден или структура некорректна
    """
    path = Path(yaml_path)

    if not path.exists():
        raise ConfigError(f"Файл конфигурации {yaml_path} не найден")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Ошибка чтения YAML: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError("YAML должен содержать словарь верхнего уровня")

    # Парсинг текстовой модели
    try:
        text_data = data.get("text_model", {})
        text_model = TextModelConfig(
            name=text_data.get("name", "gemini-3.5-flash"),
            api_version=text_data.get("api_version"),
            temperature=text_data.get("generation", {}).get("temperature", 0.5),
            top_p=text_data.get("generation", {}).get("top_p", 0.95),
            top_k=text_data.get("generation", {}).get("top_k", 64),
            max_context_chars=text_data.get("limits", {}).get("max_context_chars", 20000),
            disable_safety=text_data.get("safety", {}).get("disabled", True),
            system_prompt_template=text_data.get("prompt", {}).get("template", ""),
            fallbacks=text_data.get("fallbacks", []),
        )
    except Exception as e:
        raise ConfigError(f"Ошибка парсинга text_model: {e}") from e

    # Парсинг голосовой модели
    try:
        audio_data = data.get("audio_model", {})
        audio_model = AudioModelConfig(
            name=audio_data.get("name", "gemini-2.5-flash-native-audio-latest"),
            api_version=audio_data.get("api_version", "v1alpha"),
            temperature=audio_data.get("generation", {}).get("temperature", 1.0),
            top_p=audio_data.get("generation", {}).get("top_p", 0.95),
            top_k=float(audio_data.get("generation", {}).get("top_k", 64.0)),
            thinking_budget=audio_data.get("generation", {}).get("thinking_budget", 8192),
            max_output_tokens=audio_data.get("generation", {}).get("max_output_tokens", 8192),
            voice_name=audio_data.get("voice", {}).get("name", "Kore"),
            input_sample_rate=audio_data.get("audio", {}).get("input_sample_rate", 16000),
            output_sample_rate=audio_data.get("audio", {}).get("output_sample_rate", 24000),
            channels=audio_data.get("audio", {}).get("channels", 1),
            sample_width=audio_data.get("audio", {}).get("sample_width", 2),
            ogg_bitrate=audio_data.get("audio", {}).get("ogg_bitrate", "64k"),
            vad_disabled=audio_data.get("features", {}).get("vad_disabled", True),
            enable_output_transcription=audio_data.get("features", {}).get("enable_output_transcription", True),
            enable_tools=audio_data.get("features", {}).get("enable_tools", True),
            enable_affective_dialog=audio_data.get("features", {}).get("enable_affective_dialog", False),
            response_timeout=audio_data.get("limits", {}).get("response_timeout", 120.0),
            transcription_timeout=audio_data.get("limits", {}).get("transcription_timeout", 30.0),
            max_voice_context_chars=audio_data.get("limits", {}).get("max_voice_context_chars", 50000),
            system_prompt_template=audio_data.get("prompt", {}).get("template", ""),
            tool_prompt_template=audio_data.get("prompt", {}).get("tool_prompt_template", ""),
            response_modalities=audio_data.get("response_modalities", ["AUDIO"]),
        )
    except Exception as e:
        raise ConfigError(f"Ошибка парсинга audio_model: {e}") from e

    # Парсинг модели эмбеддингов
    try:
        embedding_data = data.get("embedding_model", {})
        embedding_model = EmbeddingModelConfig(
            name=embedding_data.get("name", "gemini-embedding-2"),
            api_version=embedding_data.get("api_version", "v1beta"),
        )
    except Exception as e:
        raise ConfigError(f"Ошибка парсинга embedding_model: {e}") from e

    # Парсинг контекстуального поиска
    try:
        contextual_data = data.get("contextual_retrieval", {})
        contextual_retrieval = ContextualRetrievalConfig(
            model=contextual_data.get("model", "gemini-3.5-flash"),
            max_doc_chars=contextual_data.get("max_doc_chars", 12000),
            parallelism=contextual_data.get("parallelism", 4),
        )
    except Exception as e:
        raise ConfigError(f"Ошибка парсинга contextual_retrieval: {e}") from e

    return ModelsConfig(
        text_model=text_model,
        audio_model=audio_model,
        embedding_model=embedding_model,
        contextual_retrieval=contextual_retrieval,
    )


def apply_env_overrides(config: ModelsConfig, env_overrides: Dict[str, Any]) -> ModelsConfig:
    """
    Применяет переопределения из .env к конфигурации моделей.

    Args:
        config: Базовая конфигурация из YAML
        env_overrides: Словарь с переопределениями из .env

    Returns:
        ModelsConfig: Конфигурация с применёнными переопределениями
    """
    # Text model overrides
    if "text_model_name" in env_overrides:
        config.text_model.name = env_overrides["text_model_name"]
    if "text_api_version" in env_overrides:
        config.text_model.api_version = env_overrides["text_api_version"]
    if "text_temperature" in env_overrides:
        config.text_model.temperature = float(env_overrides["text_temperature"])
    if "text_top_p" in env_overrides:
        config.text_model.top_p = float(env_overrides["text_top_p"])
    if "text_top_k" in env_overrides:
        config.text_model.top_k = int(env_overrides["text_top_k"])

    # Audio model overrides
    if "audio_model_name" in env_overrides:
        config.audio_model.name = env_overrides["audio_model_name"]
    if "audio_api_version" in env_overrides:
        config.audio_model.api_version = env_overrides["audio_api_version"]
    if "audio_temperature" in env_overrides:
        config.audio_model.temperature = float(env_overrides["audio_temperature"])
    if "audio_voice_name" in env_overrides:
        config.audio_model.voice_name = env_overrides["audio_voice_name"]

    # Embedding model overrides
    if "embedding_model_name" in env_overrides:
        config.embedding_model.name = env_overrides["embedding_model_name"]
    if "embedding_api_version" in env_overrides:
        config.embedding_model.api_version = env_overrides["embedding_api_version"]

    # Contextual retrieval overrides
    if "contextual_model" in env_overrides:
        config.contextual_retrieval.model = env_overrides["contextual_model"]
    if "contextual_max_doc_chars" in env_overrides:
        config.contextual_retrieval.max_doc_chars = int(env_overrides["contextual_max_doc_chars"])
    if "contextual_parallelism" in env_overrides:
        config.contextual_retrieval.parallelism = int(env_overrides["contextual_parallelism"])

    return config

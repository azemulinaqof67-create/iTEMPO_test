"""
Thread-safe менеджер клиентов с async support.

ИСПРАВЛЕНИЯ:
- Thread-safe синглтон с Lock
- Прокси передается в HTTP-клиент, НЕ изменяет os.environ
- Проверка доступности прокси без побочных эффектов
- GeminiEmbedder избавлен от дублирующей retry-логики
- Добавлены публичные recreate_gemini_client / recreate_embedder_client
"""

import logging
import socket
import threading
from threading import Lock
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import httpx
import numpy as np
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from src.core.config import Config
from src.core.exceptions import ConfigError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proxy helper
# ---------------------------------------------------------------------------




class GeminiEmbedder:
    """
    Обертка для Gemini Embedding API (gemini-embedding-001).

    Особенности:
    - Поддержка Matryoshka Representation Learning (MRL)
    - Автоматическая нормализация для размерностей != 3072
    - Рекомендуемые размерности: 768, 1536, 3072
    - Поддержка task_type для оптимизации под разные задачи

    Документация: https://ai.google.dev/gemini-api/docs/embeddings
    """

    def __init__(self, model_name: str, gemini_client, output_dimensionality: int = 1536):
        """
        Args:
            model_name: Название модели (рекомендуется "gemini-embedding-001")
            gemini_client: Экземпляр genai.Client
            output_dimensionality: Размерность выходных векторов (768/1536/3072)
        """
        self.model_name = model_name
        self.gemini_client = gemini_client
        self.output_dimensionality = output_dimensionality

        # Проверка рекомендуемых размерностей
        if output_dimensionality not in [128, 256, 512, 768, 1536, 2048, 3072]:
            print(
                f"⚠️ Warning: output_dimensionality={output_dimensionality} не в списке рекомендуемых "
                f"(768, 1536, 3072). Качество может быть ниже."
            )

    def encode(self, texts: Union[str, list[str]], **kwargs) -> np.ndarray:
        """
        Генерация embeddings для текстов с автоматической нормализацией.

        Retry/fallback при 429 выполняет вышестоящий код (EmbeddingService).
        Этот метод делает ровно один вызов API и пробрасывает исключение наверх.

        Args:
            texts: Строка или список строк
            **kwargs:
                - task_type: Тип задачи (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY,
                            SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING)
                - normalize: Принудительная нормализация (по умолчанию True для != 3072)

        Returns:
            np.ndarray: Нормализованный массив embeddings формы (n_texts, output_dimensionality)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([]).reshape(0, self.output_dimensionality)

        task_type = kwargs.get("task_type", "RETRIEVAL_DOCUMENT")
        normalize = kwargs.get("normalize", self.output_dimensionality != 3072)

        try:
            config = types.EmbedContentConfig(output_dimensionality=self.output_dimensionality, task_type=task_type)
            result = self.gemini_client.models.embed_content(model=self.model_name, contents=texts, config=config)
        except Exception as e:
            raise ConfigError(f"Ошибка при генерации embeddings через Gemini API: {e}") from e

        # Обработка результата
        embeddings = []
        if hasattr(result, "embeddings"):
            source_embeddings = result.embeddings
        elif hasattr(result, "embedding"):
            source_embeddings = [result]
        else:
            source_embeddings = result if isinstance(result, list) else [result]

        for embedding in source_embeddings:
            if hasattr(embedding, "values"):
                emb_values = embedding.values
            elif hasattr(embedding, "embedding"):
                emb_values = embedding.embedding
            else:
                emb_values = embedding

            if isinstance(emb_values, (list, tuple)):
                embeddings.append(list(emb_values))
            elif isinstance(emb_values, np.ndarray):
                embeddings.append(emb_values.tolist())
            else:
                embeddings.append(emb_values)

        result_array = np.array(embeddings, dtype=np.float32)

        if len(texts) == 1 and result_array.ndim == 1:
            result_array = result_array.reshape(1, -1)

        if normalize:
            result_array = self._normalize_embeddings(result_array)

        return result_array

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Нормализация векторов для улучшения качества semantic similarity.

        Gemini embedding-001 с MRL требует нормализации для размерностей != 3072.
        Нормализованные векторы обеспечивают более точное сравнение по направлению,
        а не по величине.

        Args:
            embeddings: Массив векторов формы (n, dim)

        Returns:
            np.ndarray: Нормализованные векторы (L2 norm = 1.0)
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Избегаем деления на 0
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms


# Глобальные переменные для строгого синглтона Qdrant (вне класса)
_shared_qdrant: Optional[QdrantClient] = None
_qdrant_lock = Lock()

class ClientManager:
    """
    Thread-safe менеджер клиентов с async support.

    Управляет жизненным циклом QdrantClient, SentenceTransformer и Gemini Client.
    """

    _local = threading.local()
    _instance_lock = Lock()

    def __init__(self, config: Config):
        self.config = config
        self._embedder: Optional[Union["SentenceTransformer", "GeminiEmbedder"]] = None  # noqa: F821
        self._gemini: Optional[object] = None  # По умолчанию (для совместимости)
        self._gemini_clients: Dict[str, object] = {} # Кеш клиентов по api_version
        self._gemini_live: Optional[object] = None  # genai.Client for Live API
        self._embedder_api_key: Optional[str] = None  # API key for current _embedder
        self._gemini_http_client: Optional[httpx.Client] = None
        self._http_client: Optional[httpx.Client] = None # Добавлено для совместимости с get_http_client
        self._init_lock = Lock()

        # Менеджер API ключей для fallback
        from src.llm.api_key_manager import ApiKeyManager

        # Gemini сообщает "retry in 56s" при RPM-лимите.
        # Сбрасываем исчерпанные ключи не раньше чем через 65 секунд,
        # чтобы не пробовать тот же ключ до снятия лимита.
        reset_interval = 65 if config.api_keys and len(config.api_keys) > 1 else 3600
        # Enable auto_rotate for per-request key rotation
        self.api_key_manager = ApiKeyManager(config.api_keys, reset_interval=reset_interval, auto_rotate=True) if config.api_keys else None

    @classmethod
    def get_instance(cls, config: Optional[Config] = None) -> "ClientManager":
        """
        Потокобезопасный синглтон, уникальный для каждого потока (thread-local).
        Это предотвращает ошибки "Future attached to a different loop" 
        при использовании aiohttp внутри genai.Client.
        """
        if getattr(cls._local, "instance", None) is None:
            if config is None:
                raise ConfigError("Config required for first initialization")
            cls._local.instance = cls(config)
        return cls._local.instance

    @classmethod
    def reset(cls):
        """Сброс синглтона (для тестов)"""
        if getattr(cls._local, "instance", None):
            cls._local.instance.close_all()
            cls._local.instance = None

    @classmethod
    def reload_config(cls, config: Config):
        """Принудительная перезагрузка конфигурации синглтона"""
        if getattr(cls._local, "instance", None):
            cls._local.instance.close_all()
        cls._local.instance = cls(config)

    def get_gemini_http_client(self) -> Optional[httpx.Client]:
        """
        Возвращает httpx.Client с прокси для Gemini API.
        """
        if self._gemini_http_client is not None:
            return self._gemini_http_client

        proxy = self.config.https_proxy
        force = self.config.force_proxy

        if not proxy:
            logger.debug("🌐 ВНИМАНИЕ: Прокси для Gemini не задан в .env, будет использовано прямое соединение.")
            return None

        if not force:
            # Проверяем доступность
            parsed = urlparse(proxy)
            if not parsed.hostname or not parsed.port:
                logger.warning("⚠️ Некорректный proxy URL: %s. Пропускаю.", proxy)
                return None
            try:
                with socket.create_connection((parsed.hostname, parsed.port), timeout=2.0):
                    pass
            except OSError as e:
                logger.warning("⚠️ Proxy %s недоступен (%s). Использую прямое соединение.", proxy, e)
                return None

        logger.info("🔒 ИНИЦИАЛИЗАЦИЯ: Gemini API будет работать через прокси: %s", proxy)
        self._gemini_http_client = httpx.Client(proxy=proxy, timeout=60.0)
        return self._gemini_http_client

    # ------------------------------------------------------------------
    # Публичные методы безопасного пересоздания клиентов
    # (используются из text.py и embeddings.py вместо прямого доступа к _gemini)
    # ------------------------------------------------------------------

    def recreate_gemini_client(self, api_key: str, api_version: Optional[str] = None) -> None:
        """
        Атомарная замена ГЛОБАЛЬНОГО Gemini text-клиента.
        Обычно вызывается при инициализации или полной перегрузке.
        """
        new_client = self.create_gemini_client(api_key, api_version)
        with self._init_lock:
            self._gemini = new_client
        logger.info("🔄 Глобальный Gemini text-клиент обновлен")

    def create_gemini_client(self, api_key: str, api_version: Optional[str] = None):
        """Создает новый экземпляр клиента без сохранения в глобальное состояние."""
        resolved_version = api_version or (
            self.config.text_api_version or self._get_api_version_for_model(self.config.text_model)
        )
        http_options = types.HttpOptions(
            api_version=resolved_version,
            httpxClient=self.get_gemini_http_client(),
        )
        return genai.Client(api_key=api_key, http_options=http_options)

    def recreate_embedder_client(self, api_key: str) -> None:
        """
        Атомарная замена embedding-клиента (thread-safe).

        Вызывается при смене API-ключа из EmbeddingService при 429.
        """
        http_options = types.HttpOptions(
            api_version=self.config.embedding_api_version,
            httpxClient=self.get_gemini_http_client(),
        )
        new_gemini_client = genai.Client(api_key=api_key, http_options=http_options)
        new_embedder = GeminiEmbedder(
            model_name=self.config.embedding_model,
            gemini_client=new_gemini_client,
            output_dimensionality=self.config.vector_size,
        )
        with self._init_lock:
            self._embedder = new_embedder
        logger.info("🔄 Embedding-клиент пересоздан")

    def reload_clients(self):
        """
        Горячая перезагрузка клиентов при изменении конфигурации.

        Создает новые клиенты БЕЗ lock, потом атомарно заменяет старые.
        Это предотвращает блокировку других операций.
        """
        logger.info("🔄 Горячая перезагрузка клиентов...")

        # 1. Создаем новые клиенты ВНЕ lock (не блокируем текущие операции)
        new_gemini = None
        new_gemini_live = None
        new_embedder = None
        new_http_client = None

        api_key = self.api_key_manager.get_current_key() if self.api_key_manager else self.config.gemini_api_key

        if self._gemini:
            logger.info("  - Создание нового Gemini Client (модель: %s)", self.config.text_model)
            api_version = self.config.text_api_version or self._get_api_version_for_model(self.config.text_model)
            http_options = types.HttpOptions(
                api_version=api_version, 
                httpxClient=self.get_gemini_http_client()
            )
            new_gemini = genai.Client(api_key=api_key, http_options=http_options)
            logger.info("    ✓ Новый Gemini Client создан")

        if self._embedder:
            logger.info("  - Создание нового Embedder (модель: %s)", self.config.embedding_model)
            http_options = types.HttpOptions(
                api_version=self.config.embedding_api_version, 
                httpxClient=self.get_gemini_http_client()
            )
            gemini_client = genai.Client(api_key=api_key, http_options=http_options)
            new_embedder = GeminiEmbedder(
                model_name=self.config.embedding_model,
                gemini_client=gemini_client,
                output_dimensionality=self.config.vector_size,
            )
            logger.info("    ✓ Новый Embedder создан")

        if self._gemini_live:
            logger.info("  - Создание нового Live API Client")
            http_options = types.HttpOptions(api_version=self.config.live_api_version, httpxClient=self.get_gemini_http_client())
            new_gemini_live = genai.Client(api_key=api_key, http_options=http_options)
            logger.info("    ✓ Новый Live API Client создан")

        if self._http_client and self.config.https_proxy:
            new_http_client = self._create_http_client()

        # 2. Атомарная замена (быстрая операция с lock)
        with self._init_lock:
            if new_gemini:
                self._gemini = new_gemini
            if new_gemini_live:
                self._gemini_live = new_gemini_live
            if new_embedder:
                self._embedder = new_embedder
            if self._http_client:
                old_http = self._http_client
                self._http_client = new_http_client
                if old_http:
                    try:
                        old_http.close()
                    except Exception:
                        pass

        logger.info("✅ Клиенты обновлены!")

    def preload_models(self):
        """Предзагрузка моделей."""
        logger.info("🔄 Предзагрузка моделей...")
        try:
            logger.info("  - Embedding модель: %s", self.config.embedding_model)
            logger.info("  - Gemini Client будет инициализирован при первом использовании")
            logger.info("✅ Модели готовы!")
        except Exception as e:
            logger.warning("⚠️ Ошибка при предзагрузке: %s", e)
            raise

    def get_qdrant_client(self) -> QdrantClient:
        """Thread-safe lazy init QdrantClient (shared across ALL instances and threads)."""
        global _shared_qdrant
        if _shared_qdrant is None:
            with _qdrant_lock:
                if _shared_qdrant is None:
                    path = str(self.config.storage_path)
                    logger.info("--- [QDRANT INIT] Creating shared client (path: %s) ---", path)
                    _shared_qdrant = QdrantClient(path=path)
        return _shared_qdrant

    @classmethod
    def close_qdrant(cls):
        """Явное закрытие общего клиента Qdrant."""
        global _shared_qdrant
        with _qdrant_lock:
            if _shared_qdrant:
                logger.info("--- [QDRANT CLOSE] Closing shared client ---")
                _shared_qdrant = None

    def get_embedder(self, model_name: Optional[str] = None, api_key: Optional[str] = None) -> GeminiEmbedder:
        """
        Thread-safe lazy init embedder.

        Использует Gemini Embedding API.
        """
        target_model = model_name or self.config.embedding_model
        
        # Если передан конкретный ключ или модель и они отличаются от текущих - сбрасываем кэш
        if (api_key and self._embedder_api_key != api_key) or (model_name and self._embedder and self._embedder.model_name != model_name):
            with self._init_lock:
                if (api_key and self._embedder_api_key != api_key) or (model_name and self._embedder and self._embedder.model_name != model_name):
                    self._embedder = None

        if self._embedder is None:
            with self._init_lock:
                if self._embedder is None:
                    try:
                        from google import genai
                        from google.genai import types
                    except ImportError as e:
                        raise ConfigError("google-genai SDK не установлен") from e

                    http_options = types.HttpOptions(
                        api_version=self.config.embedding_api_version,
                        httpxClient=self.get_gemini_http_client(),
                    )
                    
                    resolved_key = api_key or (
                        self.api_key_manager.get_current_key() if self.api_key_manager else self.config.gemini_api_key
                    )
                    
                    gemini_client = genai.Client(api_key=resolved_key, http_options=http_options)

                    self._embedder = GeminiEmbedder(
                        model_name=target_model,
                        gemini_client=gemini_client,
                        output_dimensionality=self.config.vector_size,
                    )
                    self._embedder_api_key = resolved_key
                    logger.debug("Используется Gemini Embedding API: %s (key: ...%s, dim: %d)", target_model, resolved_key[-4:], self.config.vector_size)

        return self._embedder

    def get_gemini_client(self, api_version: Optional[str] = None, api_key: Optional[str] = None):
        """
        Создаёт (или возвращает кешированный) Gemini Client.

        Кеш учитывает пару (api_version, api_key), поэтому ротация ключей
        гарантированно создаёт новый клиент под нужный ключ.

        Args:
            api_version: Версия API (v1 / v1beta). По умолчанию из конфига.
            api_key: API-ключ. По умолчанию из api_key_manager / конфига.
        """
        target_version = api_version or self.config.text_api_version or self._get_api_version_for_model(self.config.text_model)
        resolved_key = api_key or (
            self.api_key_manager.get_current_key() if self.api_key_manager else self.config.gemini_api_key
        )

        cache_key = f"{target_version}::{resolved_key}"

        if cache_key not in self._gemini_clients:
            with self._init_lock:
                if cache_key not in self._gemini_clients:
                    try:
                        from google import genai
                        from google.genai import types
                    except ImportError as e:
                        raise ConfigError("google-genai SDK не установлен") from e

                    logger.debug("    → Создание клиента api_version=%s key=...%s", target_version, resolved_key[-4:])
                    http_options = types.HttpOptions(
                        api_version=target_version, 
                        httpxClient=self.get_gemini_http_client()
                    )
                    client = genai.Client(api_key=resolved_key, http_options=http_options)
                    self._gemini_clients[cache_key] = client

                    # Для обратной совместимости сохраняем первый созданный клиент в _gemini
                    if self._gemini is None:
                        self._gemini = client

        return self._gemini_clients[cache_key]

    def get_gemini_client_for_live_api(self):
        """
        Thread-safe lazy init Gemini Client для Live API с кешированием.

        Live API требует v1beta версию API для WebSocket соединений.
        """
        if self._gemini_live is None:
            with self._init_lock:
                if self._gemini_live is None:
                    # Live API требует v1beta для WebSocket соединений
                    http_options = types.HttpOptions(
                        api_version=self.config.live_api_version,
                        httpxClient=self.get_gemini_http_client(),
                    )
                    api_key = (
                        self.api_key_manager.get_current_key() if self.api_key_manager else self.config.gemini_api_key
                    )
                    self._gemini_live = genai.Client(api_key=api_key, http_options=http_options)
                    logger.debug(
                        "    → Gemini Live API Client создан (api_version=%s)",
                        self.config.live_api_version,
                    )

        return self._gemini_live

    def _get_api_version_for_model(self, model_name: str) -> str:
        """
        Определяет API версию на основе названия модели.

        Args:
            model_name: Название модели

        Returns:
            str: API версия ('v1beta' для gemma, 'v1' для gemini)
        """
        if "gemma" in model_name.lower():
            return "v1beta"
        return "v1"

    def _create_http_client(self) -> Optional[httpx.Client]:
        """
        Создание HTTP-клиента с прокси.

        Поддерживает HTTP/HTTPS/SOCKS5 прокси через переменные окружения.

        Returns:
            httpx.Client или None, если прокси не настроен
        """
        if not self.config.https_proxy:
            return None

        proxy = self.config.https_proxy

        # Проверка доступности (если не forced)
        if not self.config.force_proxy:
            parsed = urlparse(proxy)
            if not parsed.hostname or not parsed.port:
                logger.warning("Некорректный HTTPS_PROXY: %s. Пропускаю.", proxy)
                return None

            try:
                with socket.create_connection((parsed.hostname, parsed.port), timeout=3.0):
                    pass
            except OSError as e:
                logger.warning("Proxy %s недоступен (%s). Игнорируем.", proxy, e)
                return None

        logger.debug("Using Proxy: %s (via environment variables)", proxy)

        # httpx автоматически подхватывает прокси из переменных окружения
        # SOCKS5 поддерживается через python-socks
        return httpx.Client(timeout=60.0)

    def get_http_client(self) -> Optional[httpx.Client]:
        """Получить HTTP-клиент с прокси (если настроен)"""
        if self._http_client is None and self.config.https_proxy:
            self._http_client = self._create_http_client()
        return self._http_client

    def close_all(self):
        """Закрытие всех ресурсов"""
        with self._init_lock:
            self.close_qdrant()

            # SentenceTransformer не требует закрытия
            # genai.Client обычно не требует явного закрытия
            self._gemini = None
            self._gemini_clients = {}
            self._gemini_live = None
            self._embedder = None

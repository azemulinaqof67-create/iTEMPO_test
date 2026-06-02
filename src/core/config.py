"""
Централизованная конфигурация с валидацией
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

from src.core.exceptions import ConfigError
from src.core.models_loader import (
    AudioModelConfig,
    ContextualRetrievalConfig,
    EmbeddingModelConfig,
    TextModelConfig,
    apply_env_overrides,
    load_models_config,
)

logger = logging.getLogger(__name__)


def _get_optional_int(env_name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.getenv(env_name)
    if val:
        try:
            return int(val)
        except ValueError:
            return default
    return default


@dataclass
class RAGConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_size: int = 1536
    collection_name: str = "documents_v2"
    search_limit: int = 10
    fetch_limit: int = 40
    use_contextual_retrieval: bool = True
    use_hybrid_search: bool = True
    use_rag_fusion: bool = True
    use_llm_rerank: bool = True
    use_semantic_chunking: bool = True
    use_parent_child_chunks: bool = True
    use_fallback: bool = True
    enable_metrics: bool = True
    use_incremental_updates: bool = True
    contextual_text_model: str = "gemini-3.5-flash"
    contextual_max_doc_chars: int = 12000
    contextual_parallelism: int = 4
    vector_fetch_limit: int = 250
    bm25_fetch_limit: int = 250
    fusion_fetch_limit: int = 40
    rrf_k: int = 60
    query_variations: int = 3
    rerank_top_k: int = 25
    rerank_max_docs: int = 50
    rerank_doc_chars: int = 1000
    semantic_chunk_size: int = 1000
    semantic_similarity_threshold: float = 0.75
    parent_chunk_size: int = 6000
    child_chunk_size: int = 400
    bm25_min_token_len: int = 2
    document_hashes_path: Path = field(default_factory=lambda: Path("qdrant_storage_v2") / "document_hashes.json")

    @classmethod
    def from_env(cls, current: Optional["RAGConfig"] = None, yaml_data: Optional[dict] = None) -> "RAGConfig":
        c = current if current else cls()
        y = yaml_data or {}
        rag_y = y.get("rag", {})
        qdrant_y = y.get("qdrant", {})

        def _get_bool(env_name: str, yaml_dict: dict, yaml_key: str, default: bool) -> bool:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val == "1"
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return bool(val)
            return default

        def _get_int(env_name: str, yaml_dict: dict, yaml_key: str, default: int) -> int:
            env_val = os.getenv(env_name)
            if env_val is not None:
                try:
                    return int(env_val)
                except ValueError:
                    pass
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return int(val)
            return default

        def _get_float(env_name: str, yaml_dict: dict, yaml_key: str, default: float) -> float:
            env_val = os.getenv(env_name)
            if env_val is not None:
                try:
                    return float(env_val)
                except ValueError:
                    pass
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return float(val)
            return default

        def _get_str(env_name: str, yaml_dict: dict, yaml_key: str, default: str) -> str:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return str(val)
            return default

        return cls(
            chunk_size=_get_int("CHUNK_SIZE", rag_y, "chunk_size", c.chunk_size),
            chunk_overlap=_get_int("CHUNK_OVERLAP", rag_y, "chunk_overlap", c.chunk_overlap),
            vector_size=_get_int("VECTOR_SIZE", qdrant_y, "vector_size", c.vector_size),
            collection_name=_get_str("COLLECTION_NAME", qdrant_y, "collection_name", c.collection_name),
            search_limit=_get_int("SEARCH_LIMIT", rag_y, "search_limit", c.search_limit),
            fetch_limit=_get_int("FETCH_LIMIT", rag_y, "fetch_limit", c.fetch_limit),
            use_contextual_retrieval=_get_bool("USE_CONTEXTUAL_RETRIEVAL", rag_y, "use_contextual_retrieval", c.use_contextual_retrieval),
            use_hybrid_search=_get_bool("USE_HYBRID_SEARCH", rag_y, "use_hybrid_search", c.use_hybrid_search),
            use_rag_fusion=_get_bool("USE_RAG_FUSION", rag_y, "use_rag_fusion", c.use_rag_fusion),
            use_llm_rerank=_get_bool("USE_LLM_RERANK", rag_y, "use_llm_rerank", c.use_llm_rerank),
            use_semantic_chunking=_get_bool("USE_SEMANTIC_CHUNKING", rag_y, "use_semantic_chunking", c.use_semantic_chunking),
            use_parent_child_chunks=_get_bool("USE_PARENT_CHILD_CHUNKS", rag_y, "use_parent_child_chunks", c.use_parent_child_chunks),
            use_fallback=_get_bool("USE_FALLBACK", rag_y, "use_fallback", c.use_fallback),
            enable_metrics=_get_bool("ENABLE_METRICS", rag_y, "enable_metrics", c.enable_metrics),
            use_incremental_updates=_get_bool("USE_INCREMENTAL_UPDATES", rag_y, "use_incremental_updates", c.use_incremental_updates),
            contextual_text_model=_get_str("CONTEXTUAL_TEXT_MODEL", rag_y, "contextual_text_model", c.contextual_text_model),
            contextual_max_doc_chars=_get_int("CONTEXTUAL_MAX_DOC_CHARS", rag_y, "contextual_max_doc_chars", c.contextual_max_doc_chars),
            contextual_parallelism=_get_int("CONTEXTUAL_PARALLELISM", rag_y, "contextual_parallelism", c.contextual_parallelism),
            vector_fetch_limit=_get_int("VECTOR_FETCH_LIMIT", rag_y, "vector_fetch_limit", c.vector_fetch_limit),
            bm25_fetch_limit=_get_int("BM25_FETCH_LIMIT", rag_y, "bm25_fetch_limit", c.bm25_fetch_limit),
            fusion_fetch_limit=_get_int("FUSION_FETCH_LIMIT", rag_y, "fusion_fetch_limit", c.fusion_fetch_limit),
            rrf_k=_get_int("RRF_K", rag_y, "rrf_k", c.rrf_k),
            query_variations=_get_int("QUERY_VARIATIONS", rag_y, "query_variations", c.query_variations),
            rerank_top_k=_get_int("RERANK_TOP_K", rag_y, "rerank_top_k", c.rerank_top_k),
            rerank_max_docs=_get_int("RERANK_MAX_DOCS", rag_y, "rerank_max_docs", c.rerank_max_docs),
            rerank_doc_chars=_get_int("RERANK_DOC_CHARS", rag_y, "rerank_doc_chars", c.rerank_doc_chars),
            semantic_chunk_size=_get_int("SEMANTIC_CHUNK_SIZE", rag_y, "semantic_chunk_size", c.semantic_chunk_size),
            semantic_similarity_threshold=_get_float("SEMANTIC_SIMILARITY_THRESHOLD", rag_y, "semantic_similarity_threshold", c.semantic_similarity_threshold),
            parent_chunk_size=_get_int("PARENT_CHUNK_SIZE", rag_y, "parent_chunk_size", c.parent_chunk_size),
            child_chunk_size=_get_int("CHILD_CHUNK_SIZE", rag_y, "child_chunk_size", c.child_chunk_size),
            bm25_min_token_len=_get_int("BM25_MIN_TOKEN_LEN", rag_y, "bm25_min_token_len", c.bm25_min_token_len),
            document_hashes_path=Path(_get_str("DOCUMENT_HASHES_PATH", qdrant_y, "document_hashes_path", str(c.document_hashes_path))),
        )


@dataclass
class HelpdeskConfig:
    intraservice_base_url: str = ""
    intraservice_login: str = ""
    intraservice_password: str = ""
    intraservice_default_service_id: Optional[int] = None
    intraservice_default_type_id: Optional[int] = None
    intraservice_initial_status_id: Optional[int] = None
    intraservice_medium_priority_id: Optional[int] = None
    intraservice_allow_create_without_status: bool = True
    intraservice_all_service_code: str = "All"
    auto_assign_enabled: bool = False
    ticket_confidence_threshold: float = 0.70
    ticket_long_context_chars: int = 4000
    ticket_auto_mark_enabled: bool = True
    ticket_create_on_behalf: bool = True
    helpdesk_auth_mode: str = "negotiate"
    helpdesk_allow_basic_fallback: bool = False
    helpdesk_require_delegation: bool = True
    helpdesk_identity_source: str = "login"
    helpdesk_spn: str = ""
    helpdesk_negotiate_force_kerberos: bool = False
    intraservice_ssl_verify: Union[bool, str, None] = None
    intraservice_ssl_cert_path: Optional[str] = None

    @classmethod
    def from_env(cls, current: Optional["HelpdeskConfig"] = None) -> "HelpdeskConfig":
        c = current if current else cls()
        ssl_verify_raw = os.getenv("INTRASERVICE_SSL_VERIFY")
        ssl_verify = c.intraservice_ssl_verify
        if ssl_verify_raw is not None:
            ssl_verify_raw = ssl_verify_raw.strip().lower()
            if ssl_verify_raw in ("0", "false", "no"):
                ssl_verify = False
            elif ssl_verify_raw in ("1", "true", "yes"):
                ssl_verify = True
            else:
                ssl_verify = ssl_verify_raw

        return cls(
            intraservice_base_url=os.getenv("INTRASERVICE_BASE_URL", c.intraservice_base_url).strip(),
            intraservice_login=os.getenv("INTRASERVICE_LOGIN", c.intraservice_login).strip(),
            intraservice_password=os.getenv("INTRASERVICE_PASSWORD", c.intraservice_password).strip(),
            intraservice_default_service_id=_get_optional_int(
                "INTRASERVICE_DEFAULT_SERVICE_ID", c.intraservice_default_service_id
            ),
            intraservice_default_type_id=_get_optional_int(
                "INTRASERVICE_DEFAULT_TYPE_ID", c.intraservice_default_type_id
            ),
            intraservice_initial_status_id=_get_optional_int(
                "INTRASERVICE_INITIAL_STATUS_ID", c.intraservice_initial_status_id
            ),
            intraservice_medium_priority_id=_get_optional_int(
                "INTRASERVICE_MEDIUM_PRIORITY_ID", c.intraservice_medium_priority_id
            ),
            intraservice_allow_create_without_status=os.getenv(
                "INTRASERVICE_ALLOW_CREATE_WITHOUT_STATUS",
                "1" if c.intraservice_allow_create_without_status else "0",
            )
            == "1",
            intraservice_all_service_code=os.getenv("INTRASERVICE_ALL_SERVICE_CODE", c.intraservice_all_service_code),
            auto_assign_enabled=os.getenv("AUTO_ASSIGN_ENABLED", "1" if c.auto_assign_enabled else "0") == "1",
            ticket_confidence_threshold=float(
                os.getenv("TRIAGE_CONFIDENCE_THRESHOLD", str(c.ticket_confidence_threshold))
            ),
            ticket_long_context_chars=int(os.getenv("TICKET_LONG_CONTEXT_CHARS", str(c.ticket_long_context_chars))),
            ticket_auto_mark_enabled=os.getenv("TICKET_AUTO_MARK_ENABLED", "1" if c.ticket_auto_mark_enabled else "0")
            == "1",
            ticket_create_on_behalf=os.getenv("TICKET_CREATE_ON_BEHALF", "1" if c.ticket_create_on_behalf else "0")
            == "1",
            helpdesk_auth_mode=os.getenv("HELPDESK_AUTH_MODE", c.helpdesk_auth_mode).strip().lower(),
            helpdesk_allow_basic_fallback=os.getenv(
                "HELPDESK_ALLOW_BASIC_FALLBACK",
                "1" if c.helpdesk_allow_basic_fallback else "0",
            )
            == "1",
            helpdesk_require_delegation=os.getenv(
                "HELPDESK_REQUIRE_DELEGATION",
                "1" if c.helpdesk_require_delegation else "0",
            )
            == "1",
            helpdesk_identity_source=os.getenv("HELPDESK_IDENTITY_SOURCE", c.helpdesk_identity_source).strip().lower(),
            helpdesk_spn=os.getenv("HELPDESK_SPN", c.helpdesk_spn).strip(),
            helpdesk_negotiate_force_kerberos=os.getenv(
                "HELPDESK_NEGOTIATE_FORCE_KERBEROS",
                "1" if c.helpdesk_negotiate_force_kerberos else "0",
            )
            == "1",
            intraservice_ssl_verify=ssl_verify,
            intraservice_ssl_cert_path=os.getenv(
                "INTRASERVICE_SSL_CERT_PATH",
                c.intraservice_ssl_cert_path if c.intraservice_ssl_cert_path else "",
            ).strip()
            or None,
        )


@dataclass
class MemoryConfig:
    chat_history_enabled: bool = True
    chat_history_db_path: Path = field(default_factory=lambda: Path("chat_history.db"))
    max_history_messages: int = 20
    summarization_threshold: int = 50
    max_context_tokens: int = 8000
    enable_auto_summarization: bool = True

    @classmethod
    def from_env(cls, current: Optional["MemoryConfig"] = None, yaml_data: Optional[dict] = None) -> "MemoryConfig":
        c = current if current else cls()
        y = yaml_data or {}
        mem_y = y.get("memory", {})

        def _get_bool(env_name: str, yaml_key: str, default: bool) -> bool:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val == "1"
            val = mem_y.get(yaml_key)
            if val is not None:
                return bool(val)
            return default

        def _get_int(env_name: str, yaml_key: str, default: int) -> int:
            env_val = os.getenv(env_name)
            if env_val is not None:
                try:
                    return int(env_val)
                except ValueError:
                    pass
            val = mem_y.get(yaml_key)
            if val is not None:
                return int(val)
            return default

        def _get_str(env_name: str, yaml_key: str, default: str) -> str:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val
            val = mem_y.get(yaml_key)
            if val is not None:
                return str(val)
            return default

        return cls(
            chat_history_enabled=_get_bool("CHAT_HISTORY_ENABLED", "chat_history_enabled", c.chat_history_enabled),
            chat_history_db_path=Path(_get_str("CHAT_HISTORY_DB_PATH", "chat_history_db_path", str(c.chat_history_db_path))),
            max_history_messages=_get_int("MAX_HISTORY_MESSAGES", "max_history_messages", c.max_history_messages),
            summarization_threshold=_get_int("SUMMARIZATION_THRESHOLD", "summarization_threshold", c.summarization_threshold),
            max_context_tokens=_get_int("MAX_CONTEXT_TOKENS", "max_context_tokens", c.max_context_tokens),
            enable_auto_summarization=_get_bool("ENABLE_AUTO_SUMMARIZATION", "enable_auto_summarization", c.enable_auto_summarization),
        )


@dataclass
class Config:
    """Централизованная конфигурация приложения"""

    telegram_token: str
    gemini_api_key: str
    embedding_model: str
    text_model: str
    audio_model: str
    embedding_api_version: str
    text_api_version: Optional[str]
    live_api_version: str
    embedding_models: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    text_model_fallbacks: List[str] = field(default_factory=list)
    text_model_config: Optional[TextModelConfig] = None
    audio_model_config: Optional[AudioModelConfig] = None
    embedding_model_config: Optional[EmbeddingModelConfig] = None
    contextual_retrieval_config: Optional[ContextualRetrievalConfig] = None
    data_path: Path = field(default_factory=lambda: Path("data"))
    storage_path: Path = field(default_factory=lambda: Path("qdrant_storage"))
    https_proxy: Optional[str] = None
    force_proxy: bool = False
    database_url: str = ""
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    telegram_whitelist: List[int] = field(default_factory=list)
    enable_document_sender: bool = True
    default_city: str = "Набережные Челны"
    max_token: Optional[str] = None  # Токен бота мессенджера MAX
    max_webhook_url: Optional[str] = None
    max_webhook_secret: Optional[str] = None
    admin_enabled: bool = True
    admin_port: int = 8080
    admin_password: str = "tempo_admin_2024"

    rag: RAGConfig = field(default_factory=RAGConfig)
    helpdesk: HelpdeskConfig = field(default_factory=HelpdeskConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def reload(self, env_file: str = ".env", models_yaml: str = "models_config.yaml"):
        """Горячая перезагрузка конфигурации из .env без перезапуска."""
        load_dotenv(env_file, override=False)

        yaml_data = {}
        yaml_path = Path(models_yaml)
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Ошибка загрузки YAML конфигурации {models_yaml}: {e}")

        qdrant_y = yaml_data.get("qdrant", {})
        app_y = yaml_data.get("app", {})

        def _get_bool(env_name: str, yaml_dict: dict, yaml_key: str, default: bool) -> bool:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val == "1"
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return bool(val)
            return default

        def _get_int(env_name: str, yaml_dict: dict, yaml_key: str, default: int) -> int:
            env_val = os.getenv(env_name)
            if env_val is not None:
                try:
                    return int(env_val)
                except ValueError:
                    pass
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return int(val)
            return default

        def _get_str(env_name: str, yaml_dict: dict, yaml_key: str, default: str) -> str:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return str(val)
            return default

        self.gemini_api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)
        self.telegram_token = os.getenv("TELEGRAM_TOKEN", self.telegram_token)
        self.default_city = _get_str("DEFAULT_CITY", app_y, "default_city", self.default_city)
        # Собираем все ключи вида GEMINI_API_KEY_X, сортируя их по числовому индексу
        env_keys = []
        for k, v in os.environ.items():
            if k.startswith("GEMINI_API_KEY_") and v.strip():
                suffix = k[len("GEMINI_API_KEY_"):]
                if suffix.isdigit():
                    env_keys.append((int(suffix), v.strip()))
        env_keys.sort(key=lambda x: x[0])
        api_keys = [v for idx, v in env_keys]

        if self.gemini_api_key and self.gemini_api_key not in api_keys:
            api_keys.insert(0, self.gemini_api_key)

        if api_keys:
            self.api_keys = api_keys
            if not self.gemini_api_key:
                self.gemini_api_key = api_keys[0]
            logger.info(f"🔄 Перезагружено {len(api_keys)} API ключей")
        self.text_model = os.getenv("GEMINI_TEXT_MODEL", self.text_model)
        self.audio_model = os.getenv("GEMINI_AUDIO_MODEL", self.audio_model)
        raw_embedding_models = os.getenv("GEMINI_EMBEDDING_MODEL", ",".join(self.embedding_models) if self.embedding_models else self.embedding_model)
        if raw_embedding_models:
            self.embedding_models = [m.strip() for m in raw_embedding_models.split(",") if m.strip()]
            self.embedding_model = self.embedding_models[0]
        self.embedding_api_version = os.getenv("EMBEDDING_API_VERSION", self.embedding_api_version)
        text_api_version = os.getenv("TEXT_API_VERSION")
        self.text_api_version = text_api_version if text_api_version else None
        self.live_api_version = os.getenv("LIVE_API_VERSION", self.live_api_version)
        self.rag = RAGConfig.from_env(self.rag, yaml_data=yaml_data)
        self.helpdesk = HelpdeskConfig.from_env(self.helpdesk)
        self.memory = MemoryConfig.from_env(self.memory, yaml_data=yaml_data)
        self.database_url = os.getenv("DATABASE_URL", self.database_url)
        self.db_pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", str(self.db_pool_min_size)))
        self.db_pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", str(self.db_pool_max_size)))
        # Поддержка обоих префиксов для гибкости
        self.https_proxy = os.getenv("BOT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or self.https_proxy
        self.storage_path = Path(_get_str("STORAGE_PATH", qdrant_y, "storage_path", str(self.storage_path)))
        force_proxy_val = os.getenv("BOT_FORCE_PROXY") or os.getenv("FORCE_PROXY")
        self.force_proxy = (force_proxy_val == "1") if force_proxy_val is not None else self.force_proxy
        self.max_webhook_url = os.getenv("MAX_WEBHOOK_URL", self.max_webhook_url)
        self.max_webhook_secret = os.getenv("MAX_WEBHOOK_SECRET", self.max_webhook_secret)

    @classmethod
    def from_env(cls, env_file: str = ".env", models_yaml: str = "models_config.yaml") -> "Config":
        """Загрузка конфигурации из .env файла с валидацией."""
        load_dotenv(env_file, override=False)
        telegram_token = os.getenv("TELEGRAM_TOKEN")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not telegram_token:
            raise ConfigError("TELEGRAM_TOKEN не задан в .env")
        # Собираем все ключи вида GEMINI_API_KEY_X, сортируя их по числовому индексу
        env_keys = []
        for k, v in os.environ.items():
            if k.startswith("GEMINI_API_KEY_") and v.strip():
                suffix = k[len("GEMINI_API_KEY_"):]
                if suffix.isdigit():
                    env_keys.append((int(suffix), v.strip()))
        env_keys.sort(key=lambda x: x[0])
        api_keys = [v for idx, v in env_keys]

        if gemini_api_key and gemini_api_key not in api_keys:
            api_keys.insert(0, gemini_api_key)

        if not api_keys:
            raise ConfigError("GEMINI_API_KEY или GEMINI_API_KEY_1 не задан в .env")

        if not gemini_api_key:
            gemini_api_key = api_keys[0]
        logger.info(f"🔑 Загружено {len(api_keys)} API ключей")

        # Загрузка YAML конфигурации
        yaml_data = {}
        yaml_path = Path(models_yaml)
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Ошибка загрузки YAML конфигурации {models_yaml}: {e}")

        # Читаем несекретные параметры из .env или yaml (app и qdrant секции)
        qdrant_y = yaml_data.get("qdrant", {})
        app_y = yaml_data.get("app", {})

        def _get_bool(env_name: str, yaml_dict: dict, yaml_key: str, default: bool) -> bool:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val == "1"
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return bool(val)
            return default

        def _get_int(env_name: str, yaml_dict: dict, yaml_key: str, default: int) -> int:
            env_val = os.getenv(env_name)
            if env_val is not None:
                try:
                    return int(env_val)
                except ValueError:
                    pass
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return int(val)
            return default

        def _get_str(env_name: str, yaml_dict: dict, yaml_key: str, default: str) -> str:
            env_val = os.getenv(env_name)
            if env_val is not None:
                return env_val
            val = yaml_dict.get(yaml_key)
            if val is not None:
                return str(val)
            return default

        models_config = load_models_config(models_yaml)
        text_model_fallbacks = models_config.text_model.fallbacks
        fallback_models_str = os.getenv("GEMINI_TEXT_MODEL_FALLBACK", "")
        if fallback_models_str:
            text_model_fallbacks = [m.strip() for m in fallback_models_str.split(",") if m.strip()]
        env_overrides = {}
        if os.getenv("GEMINI_TEXT_MODEL"):
            tm = os.getenv("GEMINI_TEXT_MODEL", "").split(",")[0].strip()
            if tm:
                env_overrides["text_model_name"] = tm
        if os.getenv("TEXT_API_VERSION"):
            env_overrides["text_api_version"] = os.getenv("TEXT_API_VERSION")
        if os.getenv("TEXT_TEMPERATURE"):
            env_overrides["text_temperature"] = os.getenv("TEXT_TEMPERATURE")
        if os.getenv("TEXT_TOP_P"):
            env_overrides["text_top_p"] = os.getenv("TEXT_TOP_P")
        if os.getenv("TEXT_TOP_K"):
            env_overrides["text_top_k"] = os.getenv("TEXT_TOP_K")
        if os.getenv("GEMINI_AUDIO_MODEL"):
            env_overrides["audio_model_name"] = os.getenv("GEMINI_AUDIO_MODEL")
        if os.getenv("LIVE_API_VERSION"):
            env_overrides["audio_api_version"] = os.getenv("LIVE_API_VERSION")
        if os.getenv("AUDIO_TEMPERATURE"):
            env_overrides["audio_temperature"] = os.getenv("AUDIO_TEMPERATURE")
        if os.getenv("AUDIO_VOICE_NAME"):
            env_overrides["audio_voice_name"] = os.getenv("AUDIO_VOICE_NAME")
        if os.getenv("GEMINI_EMBEDDING_MODEL"):
            raw_models = os.getenv("GEMINI_EMBEDDING_MODEL", "")
            env_overrides["embedding_model_name"] = [m.strip() for m in raw_models.split(",") if m.strip()][0]
        if os.getenv("EMBEDDING_API_VERSION"):
            env_overrides["embedding_api_version"] = os.getenv("EMBEDDING_API_VERSION")
        if os.getenv("CONTEXTUAL_TEXT_MODEL"):
            env_overrides["contextual_model"] = os.getenv("CONTEXTUAL_TEXT_MODEL")
        if os.getenv("CONTEXTUAL_MAX_DOC_CHARS"):
            env_overrides["contextual_max_doc_chars"] = os.getenv("CONTEXTUAL_MAX_DOC_CHARS")
        if os.getenv("CONTEXTUAL_PARALLELISM"):
            env_overrides["contextual_parallelism"] = os.getenv("CONTEXTUAL_PARALLELISM")
        models_config = apply_env_overrides(models_config, env_overrides)
        # Поддержка обоих префиксов
        https_proxy = os.getenv("BOT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY")
        force_proxy_env = os.getenv("BOT_FORCE_PROXY") or os.getenv("FORCE_PROXY")
        force_proxy = force_proxy_env == "1"
        whitelist_str = os.getenv("TELEGRAM_WHITELIST", "")
        enable_document_sender = _get_bool("ENABLE_DOCUMENT_SENDER", app_y, "enable_document_sender", True)
        whitelist = []
        if whitelist_str:
            for x in whitelist_str.split(","):
                if x.strip().isdigit():
                    whitelist.append(int(x.strip()))
        
        default_city = _get_str("DEFAULT_CITY", app_y, "default_city", "Набережные Челны")
        rag_conf = RAGConfig.from_env(yaml_data=yaml_data)
        if models_config.contextual_retrieval:
            rag_conf.contextual_text_model = models_config.contextual_retrieval.model or "gemini-3.5-flash"
            rag_conf.contextual_max_doc_chars = models_config.contextual_retrieval.max_doc_chars
            rag_conf.contextual_parallelism = models_config.contextual_retrieval.parallelism
        return cls(
            telegram_token=telegram_token,
            gemini_api_key=gemini_api_key,
            api_keys=api_keys,
            text_model_fallbacks=text_model_fallbacks,
            embedding_model=models_config.embedding_model.name or "gemini-embedding-2",
            # embedding_models строится из уже обработанного models_config (с учётом env override'ов)
            embedding_models=[models_config.embedding_model.name or "gemini-embedding-2"],
            text_model=models_config.text_model.name or "gemini-3.5-flash",
            audio_model=models_config.audio_model.name or "gemini-2.5-flash-native-audio-latest",
            embedding_api_version=models_config.embedding_model.api_version,
            text_api_version=models_config.text_model.api_version,
            live_api_version=models_config.audio_model.api_version,
            text_model_config=models_config.text_model,
            audio_model_config=models_config.audio_model,
            embedding_model_config=models_config.embedding_model,
            contextual_retrieval_config=models_config.contextual_retrieval,
            https_proxy=https_proxy,
            force_proxy=force_proxy,
            telegram_whitelist=whitelist,
            enable_document_sender=enable_document_sender,
            database_url=os.getenv("DATABASE_URL", ""),
            db_pool_min_size=int(os.getenv("DB_POOL_MIN_SIZE", "1")),
            db_pool_max_size=int(os.getenv("DB_POOL_MAX_SIZE", "10")),
            default_city=default_city,
            max_token=os.getenv("MAX_TOKEN"),
            storage_path=Path(_get_str("STORAGE_PATH", qdrant_y, "storage_path", "qdrant_storage")),
            rag=rag_conf,
            helpdesk=HelpdeskConfig.from_env(),
            memory=MemoryConfig.from_env(yaml_data=yaml_data),
            max_webhook_url=os.getenv("MAX_WEBHOOK_URL"),
            max_webhook_secret=os.getenv("MAX_WEBHOOK_SECRET"),
            admin_enabled=_get_bool("ADMIN_ENABLED", app_y, "admin_enabled", True),
            admin_port=_get_int("ADMIN_PORT", app_y, "admin_port", 8080),
            admin_password=_get_str("ADMIN_PASSWORD", app_y, "admin_password", "tempo_admin_2024"),
        )

    # Property Aliases for backward compatibility
    @property
    def chunk_size(self) -> int:
        return self.rag.chunk_size

    @chunk_size.setter
    def chunk_size(self, v):
        self.rag.chunk_size = v

    @property
    def chunk_overlap(self) -> int:
        return self.rag.chunk_overlap

    @chunk_overlap.setter
    def chunk_overlap(self, v):
        self.rag.chunk_overlap = v

    @property
    def vector_size(self) -> int:
        return self.rag.vector_size

    @vector_size.setter
    def vector_size(self, v):
        self.rag.vector_size = v

    @property
    def collection_name(self) -> str:
        return self.rag.collection_name

    @collection_name.setter
    def collection_name(self, v):
        self.rag.collection_name = v

    @property
    def search_limit(self) -> int:
        return self.rag.search_limit

    @search_limit.setter
    def search_limit(self, v):
        self.rag.search_limit = v

    @property
    def fetch_limit(self) -> int:
        return self.rag.fetch_limit

    @fetch_limit.setter
    def fetch_limit(self, v):
        self.rag.fetch_limit = v

    @property
    def use_contextual_retrieval(self) -> bool:
        return self.rag.use_contextual_retrieval

    @use_contextual_retrieval.setter
    def use_contextual_retrieval(self, v):
        self.rag.use_contextual_retrieval = v

    @property
    def use_hybrid_search(self) -> bool:
        return self.rag.use_hybrid_search

    @use_hybrid_search.setter
    def use_hybrid_search(self, v):
        self.rag.use_hybrid_search = v

    @property
    def use_rag_fusion(self) -> bool:
        return self.rag.use_rag_fusion

    @use_rag_fusion.setter
    def use_rag_fusion(self, v):
        self.rag.use_rag_fusion = v

    @property
    def use_llm_rerank(self) -> bool:
        return self.rag.use_llm_rerank

    @use_llm_rerank.setter
    def use_llm_rerank(self, v):
        self.rag.use_llm_rerank = v

    @property
    def use_semantic_chunking(self) -> bool:
        return self.rag.use_semantic_chunking

    @use_semantic_chunking.setter
    def use_semantic_chunking(self, v):
        self.rag.use_semantic_chunking = v

    @property
    def use_parent_child_chunks(self) -> bool:
        return self.rag.use_parent_child_chunks

    @use_parent_child_chunks.setter
    def use_parent_child_chunks(self, v):
        self.rag.use_parent_child_chunks = v

    @property
    def use_fallback(self) -> bool:
        return self.rag.use_fallback

    @use_fallback.setter
    def use_fallback(self, v):
        self.rag.use_fallback = v

    @property
    def enable_metrics(self) -> bool:
        return self.rag.enable_metrics

    @enable_metrics.setter
    def enable_metrics(self, v):
        self.rag.enable_metrics = v

    @property
    def use_incremental_updates(self) -> bool:
        return self.rag.use_incremental_updates

    @use_incremental_updates.setter
    def use_incremental_updates(self, v):
        self.rag.use_incremental_updates = v

    @property
    def contextual_text_model(self) -> str:
        return self.rag.contextual_text_model

    @contextual_text_model.setter
    def contextual_text_model(self, v):
        self.rag.contextual_text_model = v

    @property
    def contextual_max_doc_chars(self) -> int:
        return self.rag.contextual_max_doc_chars

    @contextual_max_doc_chars.setter
    def contextual_max_doc_chars(self, v):
        self.rag.contextual_max_doc_chars = v

    @property
    def contextual_parallelism(self) -> int:
        return self.rag.contextual_parallelism

    @contextual_parallelism.setter
    def contextual_parallelism(self, v):
        self.rag.contextual_parallelism = v

    @property
    def vector_fetch_limit(self) -> int:
        return self.rag.vector_fetch_limit

    @vector_fetch_limit.setter
    def vector_fetch_limit(self, v):
        self.rag.vector_fetch_limit = v

    @property
    def bm25_fetch_limit(self) -> int:
        return self.rag.bm25_fetch_limit

    @bm25_fetch_limit.setter
    def bm25_fetch_limit(self, v):
        self.rag.bm25_fetch_limit = v

    @property
    def fusion_fetch_limit(self) -> int:
        return self.rag.fusion_fetch_limit

    @fusion_fetch_limit.setter
    def fusion_fetch_limit(self, v):
        self.rag.fusion_fetch_limit = v

    @property
    def rrf_k(self) -> int:
        return self.rag.rrf_k

    @rrf_k.setter
    def rrf_k(self, v):
        self.rag.rrf_k = v

    @property
    def query_variations(self) -> int:
        return self.rag.query_variations

    @query_variations.setter
    def query_variations(self, v):
        self.rag.query_variations = v

    @property
    def rerank_top_k(self) -> int:
        return self.rag.rerank_top_k

    @rerank_top_k.setter
    def rerank_top_k(self, v):
        self.rag.rerank_top_k = v

    @property
    def rerank_max_docs(self) -> int:
        return self.rag.rerank_max_docs

    @rerank_max_docs.setter
    def rerank_max_docs(self, v):
        self.rag.rerank_max_docs = v

    @property
    def rerank_doc_chars(self) -> int:
        return self.rag.rerank_doc_chars

    @rerank_doc_chars.setter
    def rerank_doc_chars(self, v):
        self.rag.rerank_doc_chars = v

    @property
    def parent_chunk_size(self) -> int:
        return self.rag.parent_chunk_size

    @parent_chunk_size.setter
    def parent_chunk_size(self, v):
        self.rag.parent_chunk_size = v

    @property
    def child_chunk_size(self) -> int:
        return self.rag.child_chunk_size

    @child_chunk_size.setter
    def child_chunk_size(self, v):
        self.rag.child_chunk_size = v

    @property
    def semantic_chunk_size(self) -> int:
        return self.rag.semantic_chunk_size

    @semantic_chunk_size.setter
    def semantic_chunk_size(self, v):
        self.rag.semantic_chunk_size = v

    @property
    def semantic_similarity_threshold(self) -> float:
        return self.rag.semantic_similarity_threshold

    @semantic_similarity_threshold.setter
    def semantic_similarity_threshold(self, v):
        self.rag.semantic_similarity_threshold = v

    @property
    def bm25_min_token_len(self) -> int:
        return self.rag.bm25_min_token_len

    @bm25_min_token_len.setter
    def bm25_min_token_len(self, v):
        self.rag.bm25_min_token_len = v

    @property
    def document_hashes_path(self) -> Path:
        return self.rag.document_hashes_path

    @document_hashes_path.setter
    def document_hashes_path(self, v):
        self.rag.document_hashes_path = v

    @property
    def intraservice_base_url(self) -> str:
        return self.helpdesk.intraservice_base_url

    @intraservice_base_url.setter
    def intraservice_base_url(self, v):
        self.helpdesk.intraservice_base_url = v

    @property
    def intraservice_login(self) -> str:
        return self.helpdesk.intraservice_login

    @intraservice_login.setter
    def intraservice_login(self, v):
        self.helpdesk.intraservice_login = v

    @property
    def intraservice_password(self) -> str:
        return self.helpdesk.intraservice_password

    @intraservice_password.setter
    def intraservice_password(self, v):
        self.helpdesk.intraservice_password = v

    @property
    def intraservice_default_service_id(self) -> Optional[int]:
        return self.helpdesk.intraservice_default_service_id

    @intraservice_default_service_id.setter
    def intraservice_default_service_id(self, v):
        self.helpdesk.intraservice_default_service_id = v

    @property
    def intraservice_default_type_id(self) -> Optional[int]:
        return self.helpdesk.intraservice_default_type_id

    @intraservice_default_type_id.setter
    def intraservice_default_type_id(self, v):
        self.helpdesk.intraservice_default_type_id = v

    @property
    def intraservice_initial_status_id(self) -> Optional[int]:
        return self.helpdesk.intraservice_initial_status_id

    @intraservice_initial_status_id.setter
    def intraservice_initial_status_id(self, v):
        self.helpdesk.intraservice_initial_status_id = v

    @property
    def intraservice_medium_priority_id(self) -> Optional[int]:
        return self.helpdesk.intraservice_medium_priority_id

    @intraservice_medium_priority_id.setter
    def intraservice_medium_priority_id(self, v):
        self.helpdesk.intraservice_medium_priority_id = v

    @property
    def intraservice_allow_create_without_status(self) -> bool:
        return self.helpdesk.intraservice_allow_create_without_status

    @intraservice_allow_create_without_status.setter
    def intraservice_allow_create_without_status(self, v):
        self.helpdesk.intraservice_allow_create_without_status = v

    @property
    def intraservice_all_service_code(self) -> str:
        return self.helpdesk.intraservice_all_service_code

    @intraservice_all_service_code.setter
    def intraservice_all_service_code(self, v):
        self.helpdesk.intraservice_all_service_code = v

    @property
    def auto_assign_enabled(self) -> bool:
        return self.helpdesk.auto_assign_enabled

    @auto_assign_enabled.setter
    def auto_assign_enabled(self, v):
        self.helpdesk.auto_assign_enabled = v

    @property
    def ticket_confidence_threshold(self) -> float:
        return self.helpdesk.ticket_confidence_threshold

    @ticket_confidence_threshold.setter
    def ticket_confidence_threshold(self, v):
        self.helpdesk.ticket_confidence_threshold = v

    @property
    def ticket_long_context_chars(self) -> int:
        return self.helpdesk.ticket_long_context_chars

    @ticket_long_context_chars.setter
    def ticket_long_context_chars(self, v):
        self.helpdesk.ticket_long_context_chars = v

    @property
    def ticket_auto_mark_enabled(self) -> bool:
        return self.helpdesk.ticket_auto_mark_enabled

    @ticket_auto_mark_enabled.setter
    def ticket_auto_mark_enabled(self, v):
        self.helpdesk.ticket_auto_mark_enabled = v

    @property
    def ticket_create_on_behalf(self) -> bool:
        return self.helpdesk.ticket_create_on_behalf

    @ticket_create_on_behalf.setter
    def ticket_create_on_behalf(self, v):
        self.helpdesk.ticket_create_on_behalf = v

    @property
    def helpdesk_auth_mode(self) -> str:
        return self.helpdesk.helpdesk_auth_mode

    @helpdesk_auth_mode.setter
    def helpdesk_auth_mode(self, v):
        self.helpdesk.helpdesk_auth_mode = v

    @property
    def helpdesk_allow_basic_fallback(self) -> bool:
        return self.helpdesk.helpdesk_allow_basic_fallback

    @helpdesk_allow_basic_fallback.setter
    def helpdesk_allow_basic_fallback(self, v):
        self.helpdesk.helpdesk_allow_basic_fallback = v

    @property
    def helpdesk_require_delegation(self) -> bool:
        return self.helpdesk.helpdesk_require_delegation

    @helpdesk_require_delegation.setter
    def helpdesk_require_delegation(self, v):
        self.helpdesk.helpdesk_require_delegation = v

    @property
    def helpdesk_identity_source(self) -> str:
        return self.helpdesk.helpdesk_identity_source

    @helpdesk_identity_source.setter
    def helpdesk_identity_source(self, v):
        self.helpdesk.helpdesk_identity_source = v

    @property
    def helpdesk_spn(self) -> str:
        return self.helpdesk.helpdesk_spn

    @helpdesk_spn.setter
    def helpdesk_spn(self, v):
        self.helpdesk.helpdesk_spn = v

    @property
    def helpdesk_negotiate_force_kerberos(self) -> bool:
        return self.helpdesk.helpdesk_negotiate_force_kerberos

    @helpdesk_negotiate_force_kerberos.setter
    def helpdesk_negotiate_force_kerberos(self, v):
        self.helpdesk.helpdesk_negotiate_force_kerberos = v

    @property
    def intraservice_ssl_verify(self) -> Union[bool, str, None]:
        return self.helpdesk.intraservice_ssl_verify

    @intraservice_ssl_verify.setter
    def intraservice_ssl_verify(self, v):
        self.helpdesk.intraservice_ssl_verify = v

    @property
    def intraservice_ssl_cert_path(self) -> Optional[str]:
        return self.helpdesk.intraservice_ssl_cert_path

    @intraservice_ssl_cert_path.setter
    def intraservice_ssl_cert_path(self, v):
        self.helpdesk.intraservice_ssl_cert_path = v

    @property
    def chat_history_enabled(self) -> bool:
        return self.memory.chat_history_enabled

    @chat_history_enabled.setter
    def chat_history_enabled(self, v):
        self.memory.chat_history_enabled = v

    @property
    def chat_history_db_path(self) -> Path:
        return self.memory.chat_history_db_path

    @chat_history_db_path.setter
    def chat_history_db_path(self, v):
        self.memory.chat_history_db_path = v

    @property
    def max_history_messages(self) -> int:
        return self.memory.max_history_messages

    @max_history_messages.setter
    def max_history_messages(self, v):
        self.memory.max_history_messages = v

    @property
    def summarization_threshold(self) -> int:
        return self.memory.summarization_threshold

    @summarization_threshold.setter
    def summarization_threshold(self, v):
        self.memory.summarization_threshold = v

    @property
    def max_context_tokens(self) -> int:
        return self.memory.max_context_tokens

    @max_context_tokens.setter
    def max_context_tokens(self, v):
        self.memory.max_context_tokens = v

    @property
    def enable_auto_summarization(self) -> bool:
        return self.memory.enable_auto_summarization

    @enable_auto_summarization.setter
    def enable_auto_summarization(self, v):
        self.memory.enable_auto_summarization = v

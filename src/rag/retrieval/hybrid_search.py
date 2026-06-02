"""
Hybrid Search: Vector + Sparse Vector с Reciprocal Rank Fusion на стороне Qdrant.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from fastembed import SparseTextEmbedding
from qdrant_client import models

from src.core.clients import ClientManager
from src.core.config import Config
from src.rag.retrieval.fuzzy_name_matcher import FuzzyNameMatcher
from src.rag.retrieval.scorer import BusinessLogicScorer

logger = logging.getLogger(__name__)


class HybridSearchService:
    """Hybrid Search с нативным RRF в Qdrant (dense + sparse)."""

    def __init__(self, config: Config):
        self.config = config
        self.client_manager = ClientManager.get_instance(config)
        
        # Модель для разреженных векторов Qdrant/bm25
        logger.info("[+] Инициализация SparseTextEmbedding(Qdrant/bm25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Scorer для начисления бизнес-бонусов
        self.scorer = BusinessLogicScorer()
        
        # Потокобезопасность и статус инициализации
        self._bm25_lock = asyncio.Lock()
        self._initialized = False
        
        self.name_matcher = FuzzyNameMatcher()

    async def initialize(self):
        """Явная инициализация при старте сервиса (построение словаря FuzzyNameMatcher)."""
        async with self._bm25_lock:
            if self._initialized:
                return
            
            logger.info("[+] Инициализация HybridSearchService (FuzzyMatcher)...")
            client = self.client_manager.get_qdrant_client()
            loop = asyncio.get_running_loop()
            
            corpus_texts = []
            offset = None
            try:
                while True:
                    points, next_offset = await loop.run_in_executor(
                        None,
                        lambda off=offset: client.scroll(
                            collection_name=self.config.collection_name,
                            limit=256,
                            offset=off,
                            with_payload=True,
                            with_vectors=False,
                        ),
                    )
                    
                    for point in points:
                        payload = point.payload or {}
                        text = payload.get("original_text") or payload.get("text", "")
                        if text:
                            corpus_texts.append(text)
                            
                    if next_offset is None:
                        break
                    offset = next_offset
            except ValueError as e:
                if "not found" in str(e).lower():
                    logger.warning(f"Коллекция {self.config.collection_name} не найдена. FuzzyNameMatcher пропущен.")
                else:
                    raise
            
            if corpus_texts:
                await asyncio.to_thread(self.name_matcher.rebuild, corpus_texts)
                self._initialized = True
                logger.info("[+] HybridSearchService успешно инициализирован (FuzzyMatcher)")
            else:
                logger.warning("[!] Нет данных для FuzzyNameMatcher")

    async def search(
        self,
        query: str,
        limit: int = 10,
        company_id: Optional[str] = None,
        qdrant_filter: Optional[Any] = None,
        intent: Optional[str] = None,
    ) -> List[Dict]:
        """Hybrid search с нативным RRF и бизнес-бонусами."""
        if not self._initialized:
            await self.initialize()

        # ── Нечёткая коррекция имён ──
        corrected_query, was_corrected = self.name_matcher.correct_query(query)
        if was_corrected:
            logger.info("[F] Запрос скорректирован FuzzyNameMatcher: '%s' → '%s'", query, corrected_query)
            query = corrected_query

        # Векторный гибридный поиск (всегда уважает фильтр)
        fetch_limit = self.config.vector_fetch_limit
        raw_results = await self._vector_search(query, limit=fetch_limit, qdrant_filter=qdrant_filter)

        # Применение бонусов бизнес-логики и пересортировка
        scored_results = self.scorer.apply_bonuses(raw_results, query, company_id=company_id)

        return scored_results[:limit]

    async def _vector_search(self, query: str, limit: int, qdrant_filter: Optional[Any] = None) -> List[Dict]:
        """Векторный поиск по dense и sparse векторам с RRF-слиянием в Qdrant."""
        loop = asyncio.get_running_loop()
        akm = self.client_manager.api_key_manager
        
        max_retries = 10 if akm else 1
        last_error = None
        query_list = None
        
        # 1. Генерация плотного вектора
        for attempt in range(max_retries):
            current_key = akm.get_current_key() if akm else self.config.gemini_api_key
            embedder = self.client_manager.get_embedder(api_key=current_key)
            
            try:
                query_vector = await loop.run_in_executor(
                    None,
                    lambda: embedder.encode(query, task_type="RETRIEVAL_QUERY", normalize=True),
                )
                
                if query_vector.ndim > 1:
                    query_vector = query_vector[0]
                query_list = query_vector.tolist()
                break
                
            except Exception as e:
                last_error = e
                err_str = str(e).upper()
                is_rate_error = any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED"])
                
                if is_rate_error and akm:
                    logger.warning(
                        "Rate limit (429) для embeddings (попытка %d/%d). Ротация ключа... (Key: ...%s)",
                        attempt + 1, max_retries, current_key[-4:]
                    )
                    akm.mark_key_exhausted(current_key, f"embedding rate limit: {err_str}")
                    
                    if akm.is_all_exhausted():
                        logger.error("Все API ключи исчерпаны для эмбеддингов!")
                        from src.core.exceptions import SearchError
                        raise SearchError(f"Embedding quota exceeded for all keys: {e}") from e
                    continue
                
                logger.error("Ошибка при генерации вектора запроса: %s", e)
                from src.core.exceptions import SearchError
                raise SearchError(f"Vector generation failed: {e}") from e
        else:
            from src.core.exceptions import SearchError
            raise SearchError(f"Failed to generate vector after {max_retries} attempts.") from last_error

        # 2. Генерация разреженного вектора
        def get_sparse():
            raw = list(self.sparse_model.embed([query]))[0]
            return models.SparseVector(
                indices=raw.indices.tolist() if hasattr(raw.indices, "tolist") else list(raw.indices),
                values=raw.values.tolist() if hasattr(raw.values, "tolist") else list(raw.values)
            )
        
        sparse_vector = await loop.run_in_executor(None, get_sparse)

        # 3. Запрос к Qdrant
        client = self.client_manager.get_qdrant_client()
        try:
            prefetch = [
                models.Prefetch(
                    query=query_list,
                    using="",
                    limit=limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=limit,
                )
            ]
            
            search_result = await loop.run_in_executor(
                None,
                lambda: client.query_points(
                    collection_name=self.config.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                ),
            )
        except ValueError as e:
            if "not found" in str(e).lower():
                logger.warning(f"Коллекция {self.config.collection_name} не найдена. Пропуск поиска.")
                return []
            raise

        results = []
        for point in search_result.points:
            payload = point.payload or {}
            doc_id = str(point.id)
            results.append(
                {
                    "id": doc_id,
                    "score": point.score,
                    "text": payload.get("text", ""),
                    "original_text": payload.get("original_text", ""),
                    "source": payload.get("source", "Unknown"),
                    "parent_text": payload.get("parent_text"),
                    "parent_id": payload.get("parent_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "doc_type": payload.get("doc_type"),
                    "company_tag": payload.get("company_tag"),
                    "filename_clean": payload.get("filename_clean"),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return results

    async def clear_cache(self):
        """Очистка кэша (интерфейсная заглушка, так как локального индекса BM25 больше нет)."""
        logger.info("🧹 Кэш HybridSearchService очищен (локальный индекс BM25 отключен)")

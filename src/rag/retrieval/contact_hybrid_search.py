import asyncio
import logging
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from qdrant_client import models

from src.core.clients import ClientManager
from src.core.config import Config
from src.rag.ingestion.embeddings import AdaptiveRateLimiter

logger = logging.getLogger(__name__)


class ContactHybridSearch:
    """Класс для гибридного поиска контактов в коллекции Qdrant contacts_v1."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.client_manager = ClientManager.get_instance(self.config)
        self.collection_name = "contacts_v1"

        logger.info("[+] Получение SparseTextEmbedding для контактов из ClientManager...")
        self.sparse_model = self.client_manager.get_sparse_embedder()

        # Кэш для эмбеддингов запросов и rate limiter (100 RPM, 30k TPM)
        self._embedding_cache = TTLCache(maxsize=1000, ttl=3600)
        self._rate_limiter = AdaptiveRateLimiter(max_rpm=100, max_tpm=30000)

    async def search(
        self,
        semantic_query: str = "",
        company_filter: Optional[str] = None,
        exact_phone: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Выполняет гибридный поиск контактов в Qdrant."""
        if semantic_query.strip() and company_filter:
            if company_filter.lower() not in semantic_query.lower():
                semantic_query = f"{semantic_query} {company_filter}".strip()

        client = self.client_manager.get_qdrant_client()
        loop = asyncio.get_running_loop()

        if not client.collection_exists(self.collection_name):
            logger.warning(f"Коллекция {self.collection_name} не существует. Поиск невозможен.")
            return []

        # 1. Построение фильтров Qdrant
        strict_must = []
        global_must = []

        if company_filter:
            # Разделяем фильтр по компании на слова, чтобы MatchText гарантированно находил их при скролле
            for word in company_filter.split():
                strict_must.append(models.FieldCondition(key="company", match=models.MatchText(text=word)))

        if exact_phone:
            phone_digits = "".join(c for c in exact_phone if c.isdigit())
            if phone_digits:
                phone_filter_cond = models.Filter(
                    should=[
                        models.FieldCondition(key="phone", match=models.MatchText(text=phone_digits)),
                        models.FieldCondition(key="exact_phone", match=models.MatchValue(value=phone_digits)),
                    ]
                )
                strict_must.append(phone_filter_cond)
                global_must.append(phone_filter_cond)

        qdrant_filter_strict = models.Filter(must=strict_must) if strict_must else None
        qdrant_filter_global = models.Filter(must=global_must) if global_must else None

        # 2. Если семантический запрос пустой, ищем только по фильтрам (например, по номеру телефона)
        if not semantic_query.strip():
            if not qdrant_filter_strict:
                return []

            logger.info(
                f"[QDRANT CONTACTS SEARCH] Фильтрационный поиск (Pass 1 - strict) по: company_filter={company_filter}, exact_phone={exact_phone}"
            )
            try:
                scroll_result = await loop.run_in_executor(
                    None,
                    lambda: client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=qdrant_filter_strict,
                        limit=limit,
                        with_payload=True,
                    ),
                )
                points = scroll_result[0]
            except Exception as e:
                logger.error(f"Ошибка при строго-фильтрационном поиске в Qdrant: {e}")
                points = []

            # Если строго-фильтрационный поиск пуст и есть различие с глобальным фильтром
            # (то есть company_filter был задан, и мы можем попробовать без него)
            if not points and company_filter:
                if qdrant_filter_global:
                    logger.info(
                        f"[QDRANT CONTACTS SEARCH] Фильтрационный поиск (Pass 2 - global fallback) по: exact_phone={exact_phone}"
                    )
                    try:
                        scroll_result = await loop.run_in_executor(
                            None,
                            lambda: client.scroll(
                                collection_name=self.collection_name,
                                scroll_filter=qdrant_filter_global,
                                limit=limit,
                                with_payload=True,
                            ),
                        )
                        points = scroll_result[0]
                    except Exception as e:
                        logger.error(f"Ошибка при глобально-фильтрационном поиске в Qdrant: {e}")
                        points = []
                else:
                    logger.info("[QDRANT CONTACTS SEARCH] Глобальный фильтр пуст, скролл без фильтров отменен.")
                    points = []
            results = []
            for point in points:
                payload = point.payload or {}
                results.append(
                    {
                        "id": point.id,
                        "score": 1.0,
                        "company": payload.get("company", ""),
                        "department": payload.get("department", ""),
                        "full_name": payload.get("full_name", ""),
                        "position": payload.get("position", ""),
                        "phone": payload.get("phone", ""),
                        "email": payload.get("email", ""),
                    }
                )
            return results

        # 3. Гибридный семантический поиск
        logger.info(
            f"[QDRANT CONTACTS SEARCH] Гибридный поиск: query='{semantic_query}' | company: '{company_filter}' | phone: '{exact_phone}'"
        )

        # 3.1. Генерация плотного вектора (с кэшированием и ретраями)
        cache_key = semantic_query.strip().lower()
        query_list = None
        akm = self.client_manager.api_key_manager
        max_retries = 10 if akm else 1
        last_error = None

        if cache_key in self._embedding_cache:
            query_list = self._embedding_cache[cache_key]
        else:
            for attempt in range(max_retries):
                current_key = akm.get_current_key() if akm else self.config.gemini_api_key
                embedder = self.client_manager.get_embedder(api_key=current_key)

                estimated_tokens = len(semantic_query) // 2
                await self._rate_limiter.acquire(request_count=1, token_count=estimated_tokens)

                try:
                    query_vector = await loop.run_in_executor(
                        None,
                        lambda: embedder.encode(semantic_query, task_type="RETRIEVAL_QUERY", normalize=True),
                    )

                    if query_vector.ndim > 1:
                        query_vector = query_vector[0]
                    query_list = query_vector.tolist()
                    self._embedding_cache[cache_key] = query_list
                    break

                except Exception as e:
                    last_error = e
                    err_str = str(e).upper()
                    is_rate_error = any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "QUOTA"])

                    if is_rate_error:
                        if akm:
                            logger.warning(
                                "Rate limit (429) для эмбеддингов контактов (попытка %d/%d). Ротация ключа... (Key: ...%s)",
                                attempt + 1,
                                max_retries,
                                current_key[-4:],
                            )
                            akm.mark_key_exhausted(current_key, f"embedding rate limit: {err_str}")

                            if akm.is_all_exhausted():
                                logger.error("Все API ключи исчерпаны для эмбеддингов контактов!")
                                self._rate_limiter.force_wait(65.0)
                                raise RuntimeError(f"Embedding quota exceeded for all keys: {e}") from e
                        else:
                            self._rate_limiter.force_wait(65.0)
                        continue

                    logger.error("Ошибка при генерации вектора запроса контактов: %s", e)
                    raise RuntimeError(f"Vector generation failed: {e}") from e
            else:
                raise RuntimeError(f"Failed to generate vector after {max_retries} attempts.") from last_error

        # 3.2. Генерация разреженного вектора
        def get_sparse():
            raw = list(self.sparse_model.embed([semantic_query]))[0]
            return models.SparseVector(
                indices=raw.indices.tolist() if hasattr(raw.indices, "tolist") else list(raw.indices),
                values=raw.values.tolist() if hasattr(raw.values, "tolist") else list(raw.values),
            )

        sparse_vector = await loop.run_in_executor(None, get_sparse)

        # 3.3. Запрос к Qdrant (Pass 1 - strict)
        async def run_hybrid_search(q_filter: Optional[models.Filter]) -> List[Any]:
            try:
                prefetch = [
                    models.Prefetch(
                        query=query_list,
                        using="",
                        limit=limit * 3,  # берем с запасом для слияния
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=limit * 3,
                    ),
                ]

                search_result = await loop.run_in_executor(
                    None,
                    lambda: client.query_points(
                        collection_name=self.collection_name,
                        prefetch=prefetch,
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        query_filter=q_filter,
                        limit=limit,
                        with_payload=True,
                    ),
                )
                return search_result.points
            except Exception as e:
                logger.error(f"Ошибка при гибридном поиске в Qdrant с фильтром {q_filter}: {e}")
                return []

        logger.info(
            f"[QDRANT CONTACTS SEARCH] Pass 1 (Strict): query='{semantic_query}' | filter={qdrant_filter_strict}"
        )
        points = await run_hybrid_search(qdrant_filter_strict)

        # Pass 2 (Global Fallback), если Pass 1 ничего не вернул и был задан company_filter
        if not points and company_filter:
            logger.info(
                f"[QDRANT CONTACTS SEARCH] Pass 1 вернул 0 результатов. Pass 2 (Global Fallback): query='{semantic_query}' | filter={qdrant_filter_global}"
            )
            points = await run_hybrid_search(qdrant_filter_global)

        results = []
        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "company": payload.get("company", ""),
                    "department": payload.get("department", ""),
                    "full_name": payload.get("full_name", ""),
                    "position": payload.get("position", ""),
                    "phone": payload.get("phone", ""),
                    "email": payload.get("email", ""),
                }
            )
        return results

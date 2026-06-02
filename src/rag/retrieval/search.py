"""
Async поиск с современным RAG pipeline.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.clients import ClientManager
from src.core.config import Config
from src.core.exceptions import SearchError
from src.core.prompt_manager import PromptManager
from src.rag.retrieval.fallback import FallbackRetriever
from src.rag.retrieval.hybrid_search import HybridSearchService
from src.rag.retrieval.metrics import MetricsCollector
from src.rag.retrieval.rag_fusion import RAGFusion
from src.rag.retrieval.reranker import LLMReranker
from src.rag.retrieval.smart_links_retriever import SmartLinksRetriever

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Явный контракт возврата SearchService.search().

    Заменяет side-effect поля _last_documents / _last_retrieval_status.
    """

    chunks: List[str] = field(default_factory=list)
    """[Счёт X.XX] строки для передачи в LLM."""
    documents: List[Dict] = field(default_factory=list)
    """Нефильтрованный список документов для интерфейсов."""
    retrieval_status: Optional[str] = None
    """correct | incorrect | empty | refined | ambiguous."""


class SearchService:
    """Аsync поисковая служба с современным pipeline."""

    def __init__(self, config: Config, prompt_manager: Optional[PromptManager] = None):
        self.config = config
        self.prompt_manager = prompt_manager or PromptManager.get_instance()
        self.client_manager = ClientManager.get_instance(config)
        self.hybrid = HybridSearchService(config)
        self.fusion = RAGFusion(config, self.hybrid, self.prompt_manager)
        self.reranker = LLMReranker(config, self.prompt_manager)
        self.fallback = FallbackRetriever(config, self.hybrid, self.prompt_manager)
        self.metrics = MetricsCollector(config)
        self.smart_links = SmartLinksRetriever(config)

    async def initialize(self):
        """Явная инициализация всех компонентов поиска."""
        await self.hybrid.initialize()

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        company_id: Optional[str] = None,
        qdrant_filter: Optional[Any] = None,
        intent: Optional[str] = None,
    ) -> SearchResult:
        """
        Основной метод поиска.

        Returns:
            SearchResult: чанки (chunks), документы (documents), статус поиска
        """
        if limit is None:
            limit = self.config.search_limit

        start_time = time.time()
        retrieval_status: Optional[str] = None

        try:
            # 1. Проактивный поиск
            if self.config.use_rag_fusion:
                candidates = await self.fusion.search_with_fusion(
                    query, limit=self.config.rerank_max_docs, company_id=company_id
                )
            else:
                candidates = await self.hybrid.search(
                    query,
                    limit=self.config.rerank_max_docs,
                    company_id=company_id,
                    qdrant_filter=qdrant_filter,
                    intent=intent,
                )

            # 2. Дедупликация (до реранкера)
            unique_candidates = []
            seen_ids = set()
            for cand in candidates:
                p_id = cand.get("parent_id") or cand.get("source")
                if p_id not in seen_ids:
                    unique_candidates.append(cand)
                    seen_ids.add(p_id)
            candidates = unique_candidates

            # 3. Оценка и сортировка
            if self.config.use_llm_rerank:
                candidates, retrieval_status = await self.reranker.rerank_batch(
                    query, candidates, top_k=self.config.rerank_top_k
                )

            # 4. Реактивный фоллбэк
            is_empty = not candidates
            is_incorrect = retrieval_status is not None and retrieval_status.upper() == "INCORRECT"

            if self.config.use_fallback and (is_empty or is_incorrect):
                fallback_docs = await self.fallback.execute_fallback(
                    query, limit=self.config.rerank_max_docs, company_id=company_id
                )
                # Добавляем в конец списка candidates с дедупликацией
                for doc in fallback_docs:
                    p_id = doc.get("parent_id") or doc.get("source")
                    if p_id not in seen_ids:
                        candidates.append(doc)
                        seen_ids.add(p_id)
                # Если fallback вернул документы, обновляем статус на correct
                if fallback_docs:
                    retrieval_status = "correct"

            # 5. Форматирование результатов и обработка smart_links
            documents = candidates[:limit]

            # Форматируем результаты (превращаем в строки с заголовками)
            formatted = self._format_results(documents, limit)
            combined = "\n".join(formatted)
            if "[[Файл:" in combined or "[[Папка:" in combined:
                formatted = await self._process_smart_links(formatted)

            self.metrics.log_search(
                query=query,
                num_results=len(formatted),
                latency=time.time() - start_time,
                cache_hit=False,
                retrieval_status=retrieval_status,
            )

            return SearchResult(
                chunks=formatted,
                documents=documents,
                retrieval_status=retrieval_status,
            )
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

    async def _process_smart_links(self, results: List[str]) -> List[str]:
        """
        СОХРАНЕНА ЛОГИКА: Обработка [[Файл: ...]] и [[Папка: ...]]
        ОПТИМИЗИРОВАНО: Batch fetch для устранения N+1.
        """
        combined_text = "\n".join(results)

        file_links = set(re.findall(r"\[\[Файл:\s*(.*?)\]\]", combined_text, re.IGNORECASE))
        folder_links = set(re.findall(r"\[\[Папка:\s*(.*?)\]\]", combined_text, re.IGNORECASE))

        all_targets_with_type = []
        for f in file_links:
            all_targets_with_type.append((f.strip(), False))
        for d in folder_links:
            all_targets_with_type.append((d.strip(), True))

        if not all_targets_with_type:
            return results

        print(f"  -> Обнаружено {len(all_targets_with_type)} ссылок. Подгружаю контекст батчем...")

        # Получаем чанки для ВСЕХ ссылок через ретривер
        tasks = [self.smart_links.fetch_by_source(target, is_folder) for target, is_folder in all_targets_with_type]
        fetched_results = await asyncio.gather(*tasks)

        extra_chunks = []
        found_texts = set(self.clean_scores(results))

        for (target, _), chunks in zip(all_targets_with_type, fetched_results, strict=False):
            for ch in chunks:
                if ch not in found_texts:
                    extra_chunks.append(f"[Ссылка: {target}] {ch}")
                    found_texts.add(ch)

        return results + extra_chunks

    @staticmethod
    def clean_scores(results: List[str]) -> List[str]:
        """
        Убрать префиксы [Score: X.XX] из результатов.

        Используется перед передачей в LLM.
        """
        clean = []
        for r in results:
            if "] " in r:
                clean.append(r.split("] ", 1)[1])
            else:
                clean.append(r)
        return clean

    def _format_results(self, candidates: List[Dict], limit: int) -> List[str]:
        """Форматирование результатов в строковый вид с метаданными."""
        results = []
        for cand in candidates[:limit]:
            text = self._select_text(cand)
            title = cand.get("metadata", {}).get("title") or cand.get("title", "Документ без названия")

            # Используем более человекопонятный заголовок вместо пути к файлу
            header = f"=== СОДЕРЖИМОЕ ДОКУМЕНТА: {title} ==="

            results.append(f"{header}\n{text}\n")
        return results

    def _select_text(self, candidate: Dict) -> str:
        """Выбор текста для LLM (parent chunk при наличии)."""
        parent_text = candidate.get("parent_text")
        source = candidate.get("source", "Unknown")

        logger.info(
            f"DEBUG: Candidate from {source}. Has parent_text: {parent_text is not None}, len: {len(parent_text) if parent_text else 0}"
        )

        # Если parent_text есть (даже если это пустая строка, хотя это странно)
        # мы должны использовать его, если включен режим parent-child.
        # Но на практике мы проверяем наличие контента.
        if self.config.use_parent_child_chunks and parent_text and len(parent_text) > 0:
            return parent_text

        return candidate.get("text", "")

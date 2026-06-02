"""
RAG Fusion: Multi-query retrieval с RRF.
"""

import asyncio
from typing import Dict, List, Optional

from src.core.config import Config
from src.core.prompt_manager import PromptManager
from src.llm.text import TextLLMService
from src.rag.retrieval.hybrid_search import HybridSearchService


class RAGFusion:
    """Multi-query RAG с RRF fusion."""

    def __init__(self, config: Config, hybrid_search: HybridSearchService, prompt_manager: PromptManager):
        self.config = config
        self.search = hybrid_search
        self.prompt_manager = prompt_manager
        self.llm = TextLLMService(config)

    async def search_with_fusion(self, query: str, limit: int = 10, company_id: Optional[str] = None) -> List[Dict]:
        """Multi-query search с RRF fusion."""
        variations = await self._expand_query(query)
        all_queries = [query] + variations

        # Выполняем все поисковые запросы параллельно
        tasks = [self.search.search(q, limit=self.config.fusion_fetch_limit, company_id=company_id) for q in all_queries]
        all_results = await asyncio.gather(*tasks)

        all_rankings: List[Dict[str, int]] = []
        doc_by_id = {}
        for results in all_results:
            ranking = {}
            for rank, r in enumerate(results):
                doc_id = r["id"]
                ranking[doc_id] = rank
                doc_by_id[doc_id] = r
            all_rankings.append(ranking)

        rrf_scores = {}
        k = self.config.rrf_k
        all_ids = set()
        for ranking in all_rankings:
            all_ids.update(ranking.keys())

        for doc_id in all_ids:
            score = 0.0
            for ranking in all_rankings:
                if doc_id in ranking:
                    score += 1.0 / (k + ranking[doc_id])
            rrf_scores[doc_id] = score

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_results = []
        for doc_id in sorted_ids[:limit]:
            doc = doc_by_id.get(doc_id, {})
            fused_results.append(
                {
                    "id": doc_id,
                    "text": doc.get("text", ""),
                    "original_text": doc.get("original_text", ""),
                    "source": doc.get("source", "Unknown"),
                    "parent_text": doc.get("parent_text"),
                    "parent_id": doc.get("parent_id"),
                    "chunk_index": doc.get("chunk_index"),
                    "score": rrf_scores.get(doc_id, 0.0),
                }
            )
        return fused_results

    async def _expand_query(self, query: str) -> List[str]:
        if self.config.query_variations <= 0:
            return []
        prompt_template = self.prompt_manager.get_prompt("query_expansion")
        prompt = prompt_template.format(n=self.config.query_variations, query=query)
        response = await self.llm.generate(prompt, temperature=0.3)
        variations = [line.strip() for line in response.splitlines() if line.strip()]
        return variations[: self.config.query_variations]

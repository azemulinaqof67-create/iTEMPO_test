"""
AdaptiveFallback Retriever: combines CRAG and HyDE principles into a single LLM call.
"""

from typing import Dict, List, Optional

from src.core.config import Config
from src.core.prompt_manager import PromptManager
from src.llm.text import TextLLMService
from src.rag.retrieval.hybrid_search import HybridSearchService


class FallbackRetriever:
    """Fallback retriever generating an optimized search query combining query rewriting and hypothetical context."""

    def __init__(self, config: Config, search_service: HybridSearchService, prompt_manager: PromptManager):
        self.config = config
        self.search_service = search_service
        self.prompt_manager = prompt_manager
        self.llm = TextLLMService(config)

    async def execute_fallback(self, query: str, limit: int = 10, company_id: Optional[str] = None) -> List[Dict]:
        """
        Executes 1 LLM call to generate the fallback query string,
        then performs exactly 1 hybrid search vector call.
        """
        prompt_template = self.prompt_manager.get_prompt("fallback")
        fallback_query = await self.llm.generate(
            prompt_template.format(query=query),
            temperature=0.3
        )
        fallback_query = fallback_query.strip().strip('"\'')
        if not fallback_query:
            fallback_query = query
        return await self.search_service.search(fallback_query, limit=limit, company_id=company_id)

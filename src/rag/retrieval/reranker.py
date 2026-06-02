"""
LLM-based reranker (cross-encoder style).
"""

import logging
from typing import Dict, List, Literal, Tuple
from pydantic import BaseModel, Field

from src.core.config import Config
from src.core.prompt_manager import PromptManager
from src.llm.text import TextLLMService

logger = logging.getLogger(__name__)


class RerankerOutput(BaseModel):
    """Схема ответа реранкера."""
    order: List[int] = Field(
        description="Индексы наиболее релевантных документов в порядке убывания их полезности."
    )
    status: Literal["CORRECT", "INCORRECT"] = Field(
        description="'CORRECT' если в документах есть ответ или полезная информация, иначе 'INCORRECT'."
    )


class LLMReranker:
    """Cross-encoder reranking через LLM."""

    def __init__(self, config: Config, prompt_manager: PromptManager):
        self.config = config
        self.prompt_manager = prompt_manager
        self.llm = TextLLMService(config)

    async def rerank_batch(self, query: str, documents: List[Dict], top_k: int) -> Tuple[List[Dict], str]:
        """Batch reranking через один LLM вызов."""
        if not documents:
            return [], "incorrect"

        docs_list = []
        for i, doc in enumerate(documents[: self.config.rerank_max_docs]):
            metadata = doc.get("metadata") or {}
            system_hints = metadata.get("system_hints", [])
            hints_str = ""
            if system_hints and isinstance(system_hints, list):
                hints_str = f"(Подсказка системы: {', '.join(system_hints)}) "
            
            doc_text = doc.get('text', '')[: self.config.rerank_doc_chars]
            docs_list.append(f"[{i}] {hints_str}{doc_text}")

        docs_text = "\n\n".join(docs_list)

        try:
            prompt_template = self.prompt_manager.get_prompt("reranker")
            response = await self.llm.generate_structured(
                prompt=prompt_template.format(query=query, documents=docs_text),
                response_schema=RerankerOutput,
                temperature=0.1,
            )
            order = response.order
            status = response.status.strip().lower()
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            order = list(range(len(documents)))
            status = "correct"

        # Восстанавливаем порядок
        reranked: List[Dict] = []
        seen = set()
        for idx in order:
            if 0 <= idx < len(documents) and idx not in seen:
                reranked.append(documents[idx])
                seen.add(idx)

        # Добавляем упущенные документы
        for i, doc in enumerate(documents):
            if i not in seen:
                reranked.append(doc)

        return reranked[:top_k], status


"""
Contextual Retrieval: добавление контекста документа к чанкам.
"""

import asyncio
from pathlib import Path
from typing import Dict, List

from src.core.config import Config
from src.llm.text import TextLLMService


class ContextualChunker:
    """Добавляет контекст документа к чанкам перед индексацией."""

    CONTEXT_PROMPT = (
        "Документ: {doc_title}\n"
        "Путь: {doc_source}\n\n"
        "<полный_документ>\n"
        "{full_document}\n"
        "</полный_документ>\n\n"
        "Вот чанк из этого документа:\n"
        "<чанк>\n"
        "{chunk}\n"
        "</чанк>\n\n"
        "Напиши краткое описание контекста (1-2 предложения), "
        "которое поможет понять содержание чанка без полного документа. "
        "Включи: название документа, раздел, тему. "
        "Ответь ТОЛЬКО контекстом, без пояснений."
    )

    def __init__(self, config: Config):
        self.config = config
        self.llm = TextLLMService(config)
        self._semaphore = asyncio.Semaphore(self.config.contextual_parallelism)

    async def contextualize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Возвращает чанки с добавленным контекстом."""
        if not chunks:
            return chunks

        total = len(chunks)
        completed = 0
        lock = asyncio.Lock()

        async def _contextualize_with_progress(chunk: Dict) -> Dict:
            nonlocal completed
            res = await self._contextualize_one(chunk)
            async with lock:
                completed += 1
                if completed % 10 == 0 or completed == total:
                    percent = (completed / total) * 100
                    print(f"⏳ [Контекстуализация] Обработано {completed}/{total} чанков ({percent:.1f}%)")
            return res

        tasks = [_contextualize_with_progress(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    async def _contextualize_one(self, chunk: Dict) -> Dict:
        async with self._semaphore:
            doc_text = chunk.get("document_text", "")
            if not doc_text:
                return chunk

            source = chunk.get("source", "")
            doc_title = Path(source).stem if source else "Документ"
            full_document = doc_text[: self.config.contextual_max_doc_chars]

            prompt = self.CONTEXT_PROMPT.format(
                doc_title=doc_title,
                doc_source=source,
                full_document=full_document,
                chunk=chunk.get("text", ""),
            )

            context = await self.llm.generate(
                prompt,
                model_override=self.config.contextual_text_model,
                temperature=0.2,
            )

            combined = f"{context.strip()}\n\n{chunk.get('text', '')}"
            chunk["text"] = combined
            return chunk

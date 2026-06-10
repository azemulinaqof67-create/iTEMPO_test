"""
Алгоритмы разбиения текста на смысловые сегменты.
"""

import re
import uuid
from typing import Dict, List, Optional

from src.core.config import Config
from src.rag.ingestion.semantic_chunker import SemanticChunker


class TextChunker:
    """Слой Логики: Отвечает за разбиение текста на чанки по правилам."""

    def __init__(self, config: Config, semantic_chunker: Optional[SemanticChunker] = None):
        self.config = config
        self.semantic_chunker = semantic_chunker

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Разбиение списка документов на чанки."""
        chunked: List[Dict] = []

        for doc in documents:
            chunked.extend(self.chunk_single_document(doc))

        return chunked

    def chunk_single_document(self, doc: Dict) -> List[Dict]:
        """Разбиение одного документа."""
        source = doc["source"]
        text = doc["text"]
        metadata = doc.get("metadata", {})

        context_prefix = self._get_context_prefix(metadata)

        # Для справочников контактов ПРИНУДИТЕЛЬНО используем построчный чанкинг с МЕЛКИМ размером чанка
        is_contacts = "contacts" in source.lower()

        # Для контактов используем меньший размер чанка, чтобы избежать перемешивания людей в одном блоке
        active_chunk_size = 500 if is_contacts else self.config.chunk_size
        active_overlap = 50 if is_contacts else self.config.chunk_overlap

        if self.config.use_parent_child_chunks:
            return self._create_parent_child_chunks(text, source, metadata, context_prefix)

        # Выбор алгоритма: для контактов — всегда _chunk_text
        if is_contacts:
            initial_company = metadata.get("company", "")
            chunks = self._chunk_text(text, active_chunk_size, active_overlap, initial_company=initial_company)
        elif self.semantic_chunker:
            chunks = self.semantic_chunker.chunk(text=text, max_chunk_size=self.config.semantic_chunk_size)
        else:
            chunks = self._chunk_text(text, active_chunk_size, active_overlap)

        result = []
        for chunk_text in chunks:
            result.append(
                self._build_chunk_dict(
                    chunk_text=chunk_text,
                    source=source,
                    metadata=metadata,
                    context_prefix=context_prefix,
                    full_text=text,
                )
            )
        return result

    def _get_context_prefix(self, metadata: Dict) -> str:
        if not metadata:
            return ""
        title = metadata.get("title", "")
        desc = metadata.get("description", "")
        if title or desc:
            return f"[Документ: {title}] [Контекст: {desc}]\n"
        return ""

    def _create_parent_child_chunks(self, text: str, source: str, metadata: Dict, context_prefix: str) -> List[Dict]:
        result = []
        if self.semantic_chunker:
            parent_chunks = self.semantic_chunker.chunk(text, max_chunk_size=self.config.parent_chunk_size)
        else:
            parent_chunks = self._chunk_text(
                text, self.config.parent_chunk_size, self.config.chunk_overlap, split_by_headers=False
            )

        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            child_chunks = self._chunk_text(
                parent_text,
                self.config.child_chunk_size,
                min(self.config.chunk_overlap, max(0, self.config.child_chunk_size // 3)),
            )

            for idx, child_text in enumerate(child_chunks):
                chunk = self._build_chunk_dict(
                    chunk_text=child_text,
                    source=source,
                    metadata=metadata,
                    context_prefix=context_prefix,
                    full_text=text,
                )
                chunk.update(
                    {
                        "parent_text": parent_text,
                        "parent_id": parent_id,
                        "chunk_index": idx,
                    }
                )
                result.append(chunk)
        return result

    def _build_chunk_dict(
        self, chunk_text: str, source: str, metadata: Dict, context_prefix: str, full_text: str
    ) -> Dict:
        chunk = {
            "text": context_prefix + chunk_text,
            "original_text": chunk_text,
            "source": source,
            "metadata": metadata,
            "company": str(metadata.get("company", "")),
            "department": str(metadata.get("department", "")),
        }
        if self.config.use_contextual_retrieval:
            chunk["document_text"] = full_text
        return chunk

    def _chunk_text(
        self, text: str, chunk_size: int, overlap: int, initial_company: str = "", split_by_headers: bool = True
    ) -> List[str]:
        """Алгоритм разбивки текста на чанки с поддержкой Sticky Headers."""
        if not split_by_headers:
            return [text]

        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_len = 0

        # Состояние Sticky Headers
        current_company = initial_company
        current_section = ""

        # Паттерны заголовков
        company_num_pattern = re.compile(r"^\d+\.?\s+(\d+\.\d+)\s+(АО|ООО|ГК|ИП)\s+(.*)$", re.IGNORECASE)
        section_num_pattern = re.compile(r"^\d+\.?\s+(\d+\.\d+)\s+(.*)$")
        md_company_pattern = re.compile(
            r"^#\s+(?:Телефонный\s+справочник\s+)?(?:АО|ООО|ГК|ИП)?\s*[\"«](.*?)[\"»]", re.IGNORECASE
        )
        md_section_pattern = re.compile(r"^###?\s+(.*)$")

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                if current_chunk:
                    current_chunk.append(line)
                    current_len += len(line) + 1
                continue

            # Заголовки
            co_match = company_num_pattern.match(line_strip) or md_company_pattern.match(line_strip)
            sec_match = section_num_pattern.match(line_strip) or md_section_pattern.match(line_strip)

            is_header = False
            if co_match or sec_match:
                if sec_match and hasattr(sec_match, "group") and section_num_pattern.match(line_strip):
                    if sec_match.group(1).count(".") == 1:
                        is_header = True
                else:
                    is_header = True

            # ЛОГИКА РАЗБИЕНИЯ
            if is_header:
                if split_by_headers:
                    # Закрываем старый чанк
                    if current_chunk:
                        prefix = self._get_prefix(current_company, current_section)
                        chunks.append(self._finalize_chunk(current_chunk, prefix))
                        current_chunk = []
                        current_len = 0

                    # Обновляем контекст
                    if co_match:
                        current_company = co_match.group(co_match.lastindex).strip()
                        current_section = ""
                    elif sec_match:
                        current_section = sec_match.group(sec_match.lastindex).strip()

                    # Начинаем новый чанк
                    current_chunk = [line]
                    current_len = len(line)
                    continue
                else:
                    # Просто добавляем заголовок как текст
                    current_chunk.append(line)
                    current_len += len(line) + 1
                    continue

            # Обычный текст
            line_len = len(line) + 1
            if current_len + line_len > chunk_size and current_chunk:
                prefix = self._get_prefix(current_company, current_section)
                chunks.append(self._finalize_chunk(current_chunk, prefix))

                # Overlap logic
                current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_len = sum(len(l) + 1 for l in current_chunk)

            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            prefix = self._get_prefix(current_company, current_section)
            chunks.append(self._finalize_chunk(current_chunk, prefix))

        return chunks

    def _get_prefix(self, company: str, section: str) -> str:
        prefix = ""
        if company:
            prefix += f"[{company}]"
        if section:
            prefix += f" [{section}]"
        if prefix:
            prefix += ": "
        return prefix

    def _finalize_chunk(self, lines: List[str], prefix: str) -> str:
        """Сборка строк в чанк с проверкой наличия префикса."""
        chunk_str = "\n".join(lines)
        if prefix and prefix.strip() not in chunk_str:
            return prefix + "\n" + chunk_str
        return chunk_str

"""
Оркестрация обработки документов.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from src.core.clients import ClientManager
from src.core.config import Config
from src.rag.ingestion.document_loader import DocumentLoader
from src.rag.ingestion.semantic_chunker import SemanticChunker
from src.rag.ingestion.text_chunker import TextChunker


class DocumentProcessor:
    """Оркестратор: Координация загрузки и нарезки документов."""

    def __init__(self, config: Config):
        self.config = config
        self.loader = DocumentLoader(config)

        # Инициализация семантического чанкера отключена принудительно для стабильности Parent-Child
        semantic_chunker = None
        # if self.config.use_semantic_chunking:
        #     embedder = ClientManager.get_instance(self.config).get_embedder()
        #     semantic_chunker = SemanticChunker(embedder=embedder, threshold=self.config.semantic_similarity_threshold)

        self.chunker = TextChunker(config, semantic_chunker)

    def prepare_chunks(self, files: Optional[List[Path]] = None) -> List[Dict[str, str]]:
        """
        Основной метод: загрузка документов → chunking.
        """
        documents = self.loader.load_documents(files=files)
        chunks = self.chunk_documents(documents)

        print(f"Загружено {len(documents)} документов")
        print(f"Создано {len(chunks)} чанков")

        return chunks

    def list_document_files(self) -> List[Path]:
        """Прокси к лоадеру."""
        return self.loader.list_files()

    def load_documents(self, files: Optional[List[Path]] = None) -> List[Dict]:
        """Прокси к лоадеру."""
        return self.loader.load_documents(files=files)

    def _normalize_folder(self, folder: str) -> str:
        folder = folder.lower().strip()
        folder = re.sub(r'^\d+_', '', folder)
        if folder == "policies":
            return "policy"
        if folder == "locations":
            return "location"
        if folder == "benefits":
            return "benefit"
        if folder == "logistics":
            return "logistics"
        if folder.endswith("s") and not folder.endswith("ss"):
            return folder[:-1]
        return folder

    def extract_tags(self, source_path: str) -> dict:
        path_str = source_path.replace("\\", "/")
        if path_str.startswith("data/"):
            path_str = path_str[len("data/"):]
        
        parts = path_str.split("/")
        filename = parts[-1]
        folder_parts = parts[:-1]
        
        filename_clean = Path(filename).stem
        
        if not folder_parts:
            return {
                "doc_type": "general",
                "company_tag": "all",
                "filename_clean": filename_clean
            }
        
        first_folder = folder_parts[0]
        if re.match(r'^\d+_', first_folder):
            company_tag = "all"
            doc_type_parts = []
            for f in folder_parts:
                doc_type_parts.append(self._normalize_folder(f))
            doc_type = "_".join(doc_type_parts)
        else:
            company_tag = first_folder.lower()
            doc_type_parts = ["company"]
            for f in folder_parts[1:]:
                doc_type_parts.append(self._normalize_folder(f))
            doc_type = "_".join(doc_type_parts)
            
        return {
            "doc_type": doc_type,
            "company_tag": company_tag,
            "filename_clean": filename_clean
        }

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Прокси к чанкеру с обогащением метаданными из пути."""
        chunks = self.chunker.chunk_documents(documents)
        for chunk in chunks:
            source = chunk.get("source", "")
            if source:
                tags = self.extract_tags(source)
                chunk.update(tags)
        return chunks

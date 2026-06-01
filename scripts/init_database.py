"""
Скрипт для первичной инициализации векторной базы знаний.
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config  # noqa: E402
from src.rag.ingestion.chunks_cache import ChunksCache  # noqa: E402
from src.rag.ingestion.contextual_retrieval import ContextualChunker  # noqa: E402
from src.rag.ingestion.document_hasher import DocumentHasher  # noqa: E402
from src.rag.ingestion.document_processor import DocumentProcessor  # noqa: E402
from src.rag.ingestion.embeddings import EmbeddingService  # noqa: E402


async def main():
    try:
        config = Config.from_env()
    except Exception as e:
        print(f"ОШИБКА: {e}")
        print("Скопируйте env.example в .env и заполните токены.")
        raise SystemExit(1) from e

    print("--- Инициализация базы знаний ---")

    # 1. Загрузка чанков с кэшированием
    processor = DocumentProcessor(config)
    files = processor.list_document_files()

    if not files:
        print("ВНИМАНИЕ: Не найдено документов для индексации!")
        return

    hasher = DocumentHasher(config)
    if config.use_incremental_updates:
        changed_files = [f for f in files if hasher.has_changed(f)]
        if not changed_files:
            print("Изменений не найдено. База не обновлялась.")
            return
        files = changed_files

    cache_dir = Path(config.data_path) / ".chunks_cache"
    chunks_cache = ChunksCache(cache_dir)

    def make_chunker(proc: DocumentProcessor):
        """Создаёт функцию chunking без проблем с замыканием."""
        def chunker_fn(path: Path):
            doc = proc.load_documents(files=[path])
            return proc.chunk_documents(doc) if doc else []
        return chunker_fn

    chunker_fn = make_chunker(processor)

    all_chunks = []
    for idx, file_path in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] {file_path.name}")
        file_chunks = chunks_cache.get_or_create(file_path, chunker_fn)
        # Обогащаем чанки тегами из пути на случай, если они загружены из старого кэша
        for chunk in file_chunks:
            source = chunk.get("source", "")
            if source and "doc_type" not in chunk:
                tags = processor.extract_tags(source)
                chunk.update(tags)
        all_chunks.extend(file_chunks)

    chunks = all_chunks

    if not chunks and not config.use_incremental_updates:
        print("ВНИМАНИЕ: Не удалось создать чанки!")
        return

    if config.use_contextual_retrieval and chunks:
        contextualizer = ContextualChunker(config)
        chunks = await contextualizer.contextualize_chunks(chunks)

    # 2. Создание векторов
    embedding_service = EmbeddingService(config)
    
    if config.use_incremental_updates:
        data_dir = Path(config.data_path)
        await embedding_service.incremental_update(
            chunks, 
            target_sources=[str(f.relative_to(data_dir)).replace("\\", "/") for f in files]
        )
        for f in files:
            hasher.update_hash(f)
        hasher.save()
    else:
        await embedding_service.update_database(chunks)
        # Сохраняем хеши после полной инициализации, чтобы в следующий раз сработал инкремент
        for f in processor.list_document_files():
            hasher.update_hash(f)
        hasher.save()

    # Закрытие клиентов
    from src.core.clients import ClientManager

    ClientManager.get_instance().close_all()

    print("--- Инициализация завершена ---")


if __name__ == "__main__":
    asyncio.run(main())

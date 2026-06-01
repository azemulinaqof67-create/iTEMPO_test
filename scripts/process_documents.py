"""
Полный цикл обработки документов: подготовка чанков + векторизация.

Единая команда для всего процесса.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Автоматическая настройка прокси из .env
load_dotenv()
proxy_url = os.getenv("BOT_HTTPS_PROXY") or os.getenv("HTTPS_PROXY")
force_proxy = os.getenv("BOT_FORCE_PROXY") or os.getenv("FORCE_PROXY")

if proxy_url and force_proxy == "1":
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    print(f"✅ Прокси применен для скрипта обработки: {proxy_url}")

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

    print("=" * 60)
    print("ПОЛНЫЙ ЦИКЛ ОБРАБОТКИ ДОКУМЕНТОВ")
    print(f"Storage Path: {config.storage_path}")
    print(f"Collection: {config.collection_name}")
    print("=" * 60)

    # 1. Определение файлов для обработки
    processor = DocumentProcessor(config)
    files = processor.list_document_files()

    if not files:
        print("\n⚠ ВНИМАНИЕ: Не найдено документов для обработки!")
        print(f"Проверьте папку: {config.data_path}")
        return

    print(f"\nНайдено документов: {len(files)}")

    # Инкрементальное обновление
    hasher = DocumentHasher(config)
    if config.use_incremental_updates:
        changed_files = [f for f in files if hasher.has_changed(f)]
        if not changed_files:
            print("\n[OK] Изменений не найдено. База актуальна.")
            return
        print(f"Изменённых документов: {len(changed_files)}")
        files = changed_files

    # ========================================
    # ЭТАП 1: ПОДГОТОВКА ЧАНКОВ
    # ========================================
    print("\n" + "=" * 60)
    print("ЭТАП 1: ПОДГОТОВКА ЧАНКОВ")
    print("=" * 60)

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

    if not all_chunks:
        print("\n⚠ ОШИБКА: Не удалось создать чанки!")
        return

    # Статистика чанков
    stats = chunks_cache.get_stats()
    print("\n--- Статистика подготовки ---")
    print(f"Документов в кэше: {stats['cached_documents']}")
    print(f"Всего чанков: {stats['total_chunks']}")
    print(f"Примерно токенов: {stats['total_tokens']:,}")
    print(f"Кэш: {cache_dir}")

    chunks = all_chunks

    # Контекстуализация
    if config.use_contextual_retrieval:
        print("\n--- Контекстуализация чанков ---")
        contextualizer = ContextualChunker(config)
        chunks = await contextualizer.contextualize_chunks(chunks)

    # ========================================
    # ЭТАП 2: ВЕКТОРИЗАЦИЯ
    # ========================================
    print("\n" + "=" * 60)
    print("ЭТАП 2: ВЕКТОРИЗАЦИЯ")
    print("=" * 60)

    embedding_service = EmbeddingService(config)

    if config.use_incremental_updates:
        print("\nИнкрементальное обновление...")
        data_dir = Path(config.data_path)
        await embedding_service.incremental_update(
            chunks, 
            target_sources=[str(f.relative_to(data_dir)).replace("\\", "/") for f in files]
        )
        for f in files:
            hasher.update_hash(f)
        hasher.save()
    else:
        print("\nПолное обновление базы...")
        await embedding_service.update_database(chunks)

    # Закрытие клиентов
    from src.core.clients import ClientManager

    ClientManager.get_instance().close_all()

    print("\n" + "=" * 60)
    print("[OK] ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

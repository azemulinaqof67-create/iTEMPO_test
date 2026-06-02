"""
Полный цикл обработки документов: подготовка чанков + векторизация.

Единая команда для всего процесса.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Автоматическая настройка прокси из .env
load_dotenv()

# Настройка кодировки для корректного вывода кириллицы в Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Настройка уровня логирования из переменной окружения LOG_LEVEL
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=log_level
)

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
    parser = argparse.ArgumentParser(description="Полный цикл обработки документов: подготовка чанков + векторизация.")
    parser.add_argument("--chunk-only", "-c", action="store_true", help="Только подготовка чанков (без векторизации и загрузки в базу)")
    parser.add_argument("--force", "-f", action="store_true", help="Принудительное полное обновление базы (игнорировать инкрементальный режим)")
    parser.add_argument("--incremental", "-i", action="store_true", help="Принудительное инкрементальное обновление")
    parser.add_argument("--clear-cache", action="store_true", help="Очистить кэш чанков перед запуском")
    args = parser.parse_args()

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

    print(f"\nНайдено документов всего: {len(files)}")

    # Настройка инкрементального режима
    use_incremental = config.use_incremental_updates
    if args.force:
        use_incremental = False
    elif args.incremental:
        use_incremental = True

    # Инкрементальное обновление
    hasher = DocumentHasher(config)
    if use_incremental:
        changed_files = [f for f in files if hasher.has_changed(f)]
        if not changed_files:
            print("\n[OK] Изменений не найдено. База актуальна.")
            return
        print(f"Изменённых документов для обработки: {len(changed_files)}")
        files = changed_files

    # ========================================
    # ЭТАП 1: ПОДГОТОВКА ЧАНКОВ
    # ========================================
    print("\n" + "=" * 60)
    print("ЭТАП 1: ПОДГОТОВКА ЧАНКОВ")
    print("=" * 60)

    cache_dir = Path(config.data_path) / ".chunks_cache"
    chunks_cache = ChunksCache(cache_dir)

    if args.clear_cache:
        print("Очистка кэша чанков...")
        chunks_cache.clear_cache()

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

    if args.chunk_only:
        print("\n" + "=" * 60)
        print("[OK] ПОДГОТОВКА ЧАНКОВ ЗАВЕРШЕНА (--chunk-only)")
        print("=" * 60)
        
        # Закрытие клиентов
        from src.core.clients import ClientManager
        ClientManager.get_instance(config).close_all()
        return

    # ========================================
    # ЭТАП 2: ВЕКТОРИЗАЦИЯ
    # ========================================
    print("\n" + "=" * 60)
    print("ЭТАП 2: ВЕКТОРИЗАЦИЯ")
    print("=" * 60)

    embedding_service = EmbeddingService(config)

    if use_incremental:
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
        # Сохраняем хеши после полной инициализации, чтобы в следующий раз сработал инкремент
        for f in processor.list_document_files():
            hasher.update_hash(f)
        hasher.save()

    # Закрытие клиентов
    from src.core.clients import ClientManager

    ClientManager.get_instance(config).close_all()

    print("\n" + "=" * 60)
    print("[OK] ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

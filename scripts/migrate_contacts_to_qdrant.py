import asyncio
import sqlite3
import logging
from pathlib import Path
import uuid

from qdrant_client import models
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, SparseIndexParams, TokenizerType

from src.core.config import Config
from src.core.clients import ClientManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def migrate(config=None):
    # 1. Загрузка конфигурации проекта
    if config is None:
        logger.info("Загрузка конфигурации из .env...")
        try:
            config = Config.from_env()
        except Exception as e:
            logger.error(f"Не удалось загрузить конфигурацию: {e}")
            return

    # 2. Подключение к SQLite data/contacts.db
    sqlite_db_path = config.data_path / "contacts.db"
    if not sqlite_db_path.exists():
        logger.error(f"Файл SQLite базы данных не найден по пути {sqlite_db_path.absolute()}")
        return
        
    logger.info(f"Чтение данных из SQLite ({sqlite_db_path})...")
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()
    try:
        cursor.execute("SELECT id, company, department, full_name, position, phone FROM contacts")
        rows = cursor.fetchall()
    except Exception as e:
        logger.error(f"Ошибка при чтении из SQLite: {e}")
        sqlite_conn.close()
        return
    sqlite_conn.close()
    
    if not rows:
        logger.warning("Таблица contacts пуста.")
        return
        
    logger.info(f"Найдено {len(rows)} записей в SQLite. Инициализация клиентов Qdrant...")

    # 3. Инициализация ClientManager и EmbeddingService (с полноценным пулом ключей и Rate Limiter)
    from src.rag.ingestion.embeddings import EmbeddingService
    client_manager = ClientManager.get_instance(config)
    qdrant_client = client_manager.get_qdrant_client()
    emb_service = EmbeddingService(config)

    collection_name = "contacts_v1"

    # 4. Проверяем существование коллекции и получаем уже импортированные ID
    existing_ids = set()
    if qdrant_client.collection_exists(collection_name):
        logger.info(f"Коллекция {collection_name} уже существует. Получаем список существующих контактов...")
        offset = None
        while True:
            records, offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=["id"],
                with_vectors=False,
                offset=offset,
            )
            for r in records:
                if r.payload and "id" in r.payload:
                    existing_ids.add(r.payload["id"])
            if offset is None:
                break
        logger.info(f"В Qdrant уже найдено {len(existing_ids)} векторизованных контактов.")
    else:
        logger.info(f"Создание коллекции {collection_name} в Qdrant...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=config.vector_size, distance=Distance.COSINE),
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))},
        )
        
        # 5. Создаем payload-индексы
        logger.info("Создание payload-индексов...")
        # Текстовый индекс с TokenizerType.WORD на поле phone для поиска по подстрокам телефона
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="phone",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
            )
        )
        # Keyword индексы для компании и телефона (для точного поиска)
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="company",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="exact_phone",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    # Фильтруем контакты, оставляя только новые
    rows_to_migrate = [r for r in rows if r["id"] not in existing_ids]
    if not rows_to_migrate:
        logger.info("Все контакты уже векторизованы.")
        return

    # 6. Векторизация и импорт
    # Увеличиваем размер батча до 100 записей (максимально допустимый предел для одного запроса в Gemini API)
    batch_size = 100
    total = len(rows_to_migrate)
    loop = asyncio.get_running_loop()

    logger.info(f"Начало импорта {total} новых/оставшихся контактов из {len(rows)}...")
    
    try:
        for idx in range(0, total, batch_size):
            batch = rows_to_migrate[idx : idx + batch_size]
            
            # Строим тексты для векторизации
            texts = []
            payloads = []
            for r in batch:
                full_name = r["full_name"] or ""
                position = r["position"] or ""
                department = r["department"] or ""
                company = r["company"] or ""
                phone = r["phone"] or ""
                
                # Строка для эмбеддингов
                texts.append(f"{full_name} {position} {department} {company}".strip())
                
                payloads.append({
                    "id": r["id"],
                    "company": company,
                    "department": department,
                    "full_name": full_name,
                    "position": position,
                    "phone": phone,
                    "exact_phone": phone # для точного поиска
                })
                
            # Генерация dense векторов через EmbeddingService (с AdaptiveRateLimiter и ротацией ключей)
            dense_embs = await emb_service._encode_with_retry(loop, texts)
            
            # Генерация sparse векторов через EmbeddingService
            sparse_embs = await emb_service._encode_sparse(loop, texts)
            
            # Запись в Qdrant
            points = []
            for p, d_emb, s_emb in zip(payloads, dense_embs, sparse_embs):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "": d_emb.tolist() if hasattr(d_emb, "tolist") else list(d_emb),
                            "sparse": s_emb
                        },
                        payload=p
                    )
                )
                
            await loop.run_in_executor(
                None,
                lambda: qdrant_client.upsert(collection_name=collection_name, points=points)
            )
            logger.info(f"Импортировано {min(idx + batch_size, total)}/{total} контактов...")
            # Пауза 1 секунда для стабильности API
            await asyncio.sleep(1.0)
            
        logger.info("Миграция контактов в Qdrant успешно завершена!")
        
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.warning("\n⚠️ Векторизация контактов прервана пользователем. Сохраняем уже записанный прогресс...")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при импорте контактов: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(migrate())

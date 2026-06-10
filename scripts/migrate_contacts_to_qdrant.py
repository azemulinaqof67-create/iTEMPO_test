import asyncio
import sqlite3
import logging
import sys
from pathlib import Path
import uuid

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import models
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, SparseIndexParams, TokenizerType

from src.core.config import Config
from src.core.clients import ClientManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def migrate(config=None, force=False):
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
    sqlite_conn = sqlite3.connect(sqlite_db_path, timeout=30.0)
    try:
        sqlite_conn.execute("PRAGMA journal_mode=WAL;")
        sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception as e:
        logger.warning(f"Не удалось установить PRAGMA для SQLite в скрипте миграции: {e}")
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()
    try:
        cursor.execute("SELECT id, company, department, full_name, position, phone, email FROM contacts")
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
    existing_points = {} # id -> (qdrant_id, hash)
    if force and qdrant_client.collection_exists(collection_name):
        logger.info(f"Флаг --force активен. Удаление существующей коллекции {collection_name}...")
        qdrant_client.delete_collection(collection_name)

    if qdrant_client.collection_exists(collection_name):
        logger.info(f"Коллекция {collection_name} уже существует. Получаем список существующих контактов...")
        offset = None
        while True:
            records, offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=["id", "content_hash"],
                with_vectors=False,
                offset=offset,
            )
            for r in records:
                if r.payload and "id" in r.payload:
                    existing_points[r.payload["id"]] = (r.id, r.payload.get("content_hash", ""))
            if offset is None:
                break
        logger.info(f"В Qdrant уже найдено {len(existing_points)} векторизованных контактов.")
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
            field_name="email",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="exact_phone",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    # Фильтруем контакты, оставляя только измененные и новые
    import hashlib
    def compute_contact_hash(r):
        content = f"{r['company']}|{r['department']}|{r['full_name']}|{r['position']}|{r['phone']}|{r['email']}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    rows_to_migrate = []
    sqlite_ids = set()
    
    for r in rows:
        cid = r["id"]
        sqlite_ids.add(cid)
        chash = compute_contact_hash(r)
        
        if cid not in existing_points or existing_points[cid][1] != chash:
            rows_to_migrate.append((r, chash))

    # Удаляем устаревшие контакты (которых больше нет в SQLite или у которых старый формат UUID)
    ids_to_delete = []
    for cid, (q_id, chash) in existing_points.items():
        if cid not in sqlite_ids:
            ids_to_delete.append(q_id)
        elif not isinstance(q_id, int):
            ids_to_delete.append(q_id)

    if ids_to_delete:
        logger.info(f"Удаление {len(ids_to_delete)} устаревших записей из Qdrant...")
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=ids_to_delete)
        )

    if not rows_to_migrate:
        logger.info("Все контакты актуальны. Синхронизация не требуется.")
        return

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
            for item in batch:
                r, chash = item
                full_name = r["full_name"] or ""
                position = r["position"] or ""
                department = r["department"] or ""
                company = r["company"] or ""
                phone = r["phone"] or ""
                email = r["email"] or ""
                
                # Строка для эмбеддингов
                texts.append(f"{full_name} {position} {department} {company} {email}".strip())
                
                payloads.append({
                    "id": r["id"],
                    "company": company,
                    "department": department,
                    "full_name": full_name,
                    "position": position,
                    "phone": phone,
                    "email": email,
                    "exact_phone": phone, # для точного поиска
                    "content_hash": chash
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
                        id=int(p["id"]),
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
    import argparse
    parser = argparse.ArgumentParser(description="Миграция контактов из SQLite в Qdrant.")
    parser.add_argument("--force", "-f", action="store_true", help="Принудительная перевекторизация всех контактов (пересоздание коллекции)")
    args = parser.parse_args()

    asyncio.run(migrate(force=args.force))

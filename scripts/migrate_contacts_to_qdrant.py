import asyncio
import sqlite3
import logging
from pathlib import Path
import uuid
import time

from qdrant_client import models
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, SparseIndexParams, TokenizerType

from src.core.config import Config
from src.core.clients import ClientManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def encode_with_retry_and_rotation(client_manager, texts, loop, max_retries=15):
    akm = client_manager.api_key_manager
    last_error = None
    
    for attempt in range(max_retries):
        current_key = akm.get_current_key() if akm else client_manager.config.gemini_api_key
        # Получаем (или создаем) embedder для конкретного ключа
        current_embedder = client_manager.get_embedder(api_key=current_key)
        
        try:
            # Делаем вызов API
            dense_embs = await loop.run_in_executor(
                None,
                lambda: current_embedder.encode(texts, task_type="RETRIEVAL_DOCUMENT", normalize=True)
            )
            return dense_embs
        except Exception as e:
            last_error = e
            err_str = str(e).upper()
            is_rate_error = any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "QUOTA"])
            
            if is_rate_error:
                if akm:
                    masked_key = akm.get_masked_key(current_key)
                    logger.warning(
                        f"Rate limit (429) для эмбеддингов (попытка {attempt+1}/{max_retries}). Ротация ключа... (Key: {masked_key})"
                    )
                    akm.mark_key_exhausted(current_key, f"migration rate limit: {err_str}")
                    
                    if akm.is_all_exhausted():
                        logger.warning("Все API-ключи временно исчерпаны. Ожидание 30 секунд...")
                        await asyncio.sleep(30.0)
                        akm.reset_exhausted_keys()
                else:
                    logger.warning("Получен лимит запросов (429). Ожидание 60 секунд...")
                    await asyncio.sleep(60.0)
                continue
            
            logger.error(f"Неизвестная ошибка при получении эмбеддингов: {e}")
            raise e
            
    raise RuntimeError(f"Не удалось получить эмбеддинги после {max_retries} попыток.") from last_error

async def migrate():
    # 1. Загрузка конфигурации проекта
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

    # 3. Инициализация ClientManager
    client_manager = ClientManager.get_instance(config)
    qdrant_client = client_manager.get_qdrant_client()
    sparse_embedder = client_manager.get_sparse_embedder()

    collection_name = "contacts_v1"

    # 4. Пересоздаем коллекцию contacts_v1
    logger.info(f"Пересоздание коллекции {collection_name} в Qdrant...")
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        
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

    # 6. Векторизация и импорт
    # Обрабатываем батчами, например, по 20 записей (из-за ограничений Gemini)
    batch_size = 20
    total = len(rows)
    loop = asyncio.get_running_loop()

    logger.info(f"Начало импорта {total} контактов...")
    for idx in range(0, total, batch_size):
        batch = rows[idx : idx + batch_size]
        
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
            
        # Генерация dense векторов
        dense_embs = await encode_with_retry_and_rotation(client_manager, texts, loop)
        
        # Генерация sparse векторов
        def _get_sparse():
            raw_embs = sparse_embedder.embed(texts)
            res = []
            for emb in raw_embs:
                res.append(
                    models.SparseVector(
                        indices=emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices),
                        values=emb.values.tolist() if hasattr(emb.values, "tolist") else list(emb.values)
                    )
                )
            return res
            
        sparse_embs = await loop.run_in_executor(None, _get_sparse)
        
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
        # Небольшая пауза для снижения вероятности 429
        await asyncio.sleep(0.5)

    logger.info("Миграция контактов в Qdrant успешно завершена!")

if __name__ == "__main__":
    asyncio.run(migrate())

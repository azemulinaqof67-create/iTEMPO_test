# DEPRECATED: контакты больше не хранятся в Postgres.
# Используйте: uv run python -m scripts.migrate_contacts_to_qdrant
import asyncio
import sqlite3
import logging
from pathlib import Path
import asyncpg
from src.core.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def migrate():
    # 1. Загрузка конфигурации проекта
    logger.info("Загрузка конфигурации из .env...")
    try:
        config = Config.from_env()
        db_url = config.database_url
    except Exception as e:
        logger.error(f"Не удалось загрузить конфигурацию: {e}")
        return

    if not db_url:
        logger.error("DATABASE_URL не настроена в .env!")
        return

    logger.info("Подключение к PostgreSQL...")
    conn = await asyncpg.connect(db_url)
    
    try:
        # 2. Включаем pg_trgm для нечеткого поиска
        logger.info("Включение расширения pg_trgm в PostgreSQL...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        # 3. Создаем таблицу contacts
        logger.info("Создание таблицы contacts в PostgreSQL...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id SERIAL PRIMARY KEY,
                company TEXT,
                department TEXT,
                full_name TEXT,
                position TEXT,
                phone TEXT
            );
        """)
        
        # 4. Создаем GIN триграммный индекс на full_name
        logger.info("Создание GIN trgm индекса на поле full_name...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_contacts_full_name_trgm 
            ON contacts USING gin (full_name gin_trgm_ops);
        """)
        
        # Дополнительно: добавим индексы для точного поиска по телефону и компании
        logger.info("Создание индексов на phone и company...")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_contacts_phone ON contacts (phone);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts (company);")
        
        # 5. Подключение к SQLite data/contacts.db
        sqlite_db_path = Path("data/contacts.db")
        if not sqlite_db_path.exists():
            logger.error(f"Файл SQLite базы данных не найден по пути {sqlite_db_path.absolute()}")
            return
            
        logger.info(f"Чтение данных из SQLite ({sqlite_db_path})...")
        sqlite_conn = sqlite3.connect(sqlite_db_path)
        sqlite_conn.row_factory = sqlite3.Row
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT id, company, department, full_name, position, phone FROM contacts")
        rows = cursor.fetchall()
        sqlite_conn.close()
        
        logger.info(f"Найдено {len(rows)} записей в SQLite. Подготовка к переносу...")
        
        # Преобразуем строки в кортежи
        contacts_data = [
            (
                row["id"],
                row["company"],
                row["department"],
                row["full_name"],
                row["position"],
                row["phone"]
            )
            for row in rows
        ]
        
        # 6. Очищаем существующие контакты в PG перед импортом
        logger.info("Очистка существующей таблицы contacts в PostgreSQL...")
        await conn.execute("TRUNCATE TABLE contacts RESTART IDENTITY CASCADE;")
        
        # 7. Запись данных в PG
        logger.info("Запись контактов в PostgreSQL...")
        await conn.executemany("""
            INSERT INTO contacts (id, company, department, full_name, position, phone)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, contacts_data)
        
        logger.info(f"Перенос успешно завершен. Мигрировано {len(contacts_data)} записей.")
        
    except Exception as e:
        logger.exception(f"Произошла ошибка при миграции: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate())

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg


class AsyncHistoryManager:
    """Асинхронный менеджер истории для клиента с использованием asyncpg."""

    def __init__(self, db_url: str = None):
        if db_url is None:
            # Fallback to loading config if not explicitly provided
            try:
                from src.core.config import Config

                db_url = Config.from_env().database_url
            except Exception:
                db_url = "postgresql://postgres:123456@127.0.0.1:5432/itempo"  # fallback
        self.db_url = db_url
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool and self._initialized:
            return self._pool

        async with self._init_lock:
            if self._pool and self._initialized:
                return self._pool
            self._pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=5)
            await self._init_db()
            self._initialized = True
            return self._pool

    async def _init_db(self):
        """Инициализация таблиц базы данных PostgreSQL"""
        async with self._pool.acquire() as conn:
            # Таблица чатов
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS client_chats (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # Таблица сообщений
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS client_messages (
                    id SERIAL PRIMARY KEY,
                    chat_id TEXT,
                    role TEXT,
                    text TEXT,
                    timestamp TEXT,
                    FOREIGN KEY(chat_id) REFERENCES client_chats(id) ON DELETE CASCADE
                )
            """)

            # Таблица черновиков заявок
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS client_ticket_drafts (
                    id SERIAL PRIMARY KEY,
                    chat_id TEXT,
                    ad_user TEXT,
                    query TEXT,
                    assistant_answer TEXT,
                    extra_info_1 TEXT,
                    extra_info_2 TEXT,
                    status TEXT,
                    created_at TEXT,
                    FOREIGN KEY(chat_id) REFERENCES client_chats(id) ON DELETE CASCADE
                )
            """)

            # Таблица профиля пользователя
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS client_user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    async def save_profile_data(self, profile: Dict[str, str]):
        """Сохраняет данные профиля"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for key, value in profile.items():
                    await conn.execute(
                        """
                        INSERT INTO client_user_profile (key, value) VALUES ($1, $2)
                        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                        """,
                        key,
                        value,
                    )

    async def get_profile_data(self) -> Dict[str, str]:
        """Загружает все данные профиля"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT key, value FROM client_user_profile")
            return {row["key"]: row["value"] for row in rows}

    async def get_chats(self) -> List[Dict]:
        """Возвращает список чатов (без сообщений, но с количеством сообщений)"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.*, (SELECT COUNT(*) FROM client_messages m WHERE m.chat_id = c.id) as message_count
                FROM client_chats c
                ORDER BY c.updated_at DESC
            """)

            chats = []
            for row in rows:
                chats.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "message_count": row["message_count"],
                    }
                )
            return chats

    async def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Возвращает полный объект чата с сообщениями"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            chat_row = await conn.fetchrow("SELECT * FROM client_chats WHERE id = $1", chat_id)
            if not chat_row:
                return None

            msg_rows = await conn.fetch("SELECT * FROM client_messages WHERE chat_id = $1 ORDER BY id ASC", chat_id)

            messages = []
            for row in msg_rows:
                messages.append(
                    {
                        "role": row["role"],
                        "text": row["text"],
                        "timestamp": row["timestamp"],
                    }
                )

            return {
                "id": chat_row["id"],
                "title": chat_row["title"],
                "created_at": chat_row["created_at"],
                "updated_at": chat_row["updated_at"],
                "messages": messages,
            }

    async def create_chat(self, title: str = "Новый чат") -> str:
        """Создает новый чат и возвращает его ID"""
        pool = await self.get_pool()
        chat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO client_chats (id, title, created_at, updated_at) VALUES ($1, $2, $3, $4)",
                chat_id,
                title,
                now,
                now,
            )
        return chat_id

    async def add_message(self, chat_id: str, role: str, text: str):
        """Добавляет сообщение в чат"""
        pool = await self.get_pool()
        now = datetime.now().isoformat()

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Добавляем сообщение
                await conn.execute(
                    "INSERT INTO client_messages (chat_id, role, text, timestamp) VALUES ($1, $2, $3, $4)",
                    chat_id,
                    role,
                    text,
                    now,
                )

                # Обновляем время чата
                await conn.execute("UPDATE client_chats SET updated_at = $1 WHERE id = $2", now, chat_id)

                if role == "user":
                    count = await conn.fetchval("SELECT count(*) FROM client_messages WHERE chat_id = $1", chat_id)
                    if count <= 2:
                        current_title = await conn.fetchval("SELECT title FROM client_chats WHERE id = $1", chat_id)
                        if current_title == "Новый чат":
                            new_title = (text[:30] + "...") if len(text) > 30 else text
                            await conn.execute("UPDATE client_chats SET title = $1 WHERE id = $2", new_title, chat_id)

    async def delete_chat(self, chat_id: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM client_chats WHERE id = $1", chat_id)

    async def clear_chat_messages(self, chat_id: str):
        """Удаляет все сообщения из конкретного чата"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM client_messages WHERE chat_id = $1", chat_id)

    async def clear_history(self):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM client_messages")
                await conn.execute("DELETE FROM client_chats")
                await conn.execute("DELETE FROM client_ticket_drafts")

    async def save_ticket_draft(
        self,
        chat_id: str,
        ad_user: str,
        query: str,
        assistant_answer: str,
        extra_info_1: str = None,
        extra_info_2: str = None,
        status: str = "pending",
    ):
        """Сохраняет черновик заявки в базу данных"""
        pool = await self.get_pool()
        now = datetime.now().isoformat()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO client_ticket_drafts
                (chat_id, ad_user, query, assistant_answer, extra_info_1, extra_info_2, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                chat_id,
                ad_user,
                query,
                assistant_answer,
                extra_info_1,
                extra_info_2,
                status,
                now,
            )


class HistoryManager:
    """Синхронная обертка для обратной совместимости с десктопным клиентом."""

    def __init__(self, db_url=None):
        self._async_manager = AsyncHistoryManager(db_url)

    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(coro, loop).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def save_profile_data(self, profile):
        return self._run(self._async_manager.save_profile_data(profile))

    def get_profile_data(self):
        return self._run(self._async_manager.get_profile_data())

    def get_chats(self):
        return self._run(self._async_manager.get_chats())

    def get_chat(self, chat_id):
        return self._run(self._async_manager.get_chat(chat_id))

    def create_chat(self, title="Новый чат"):
        return self._run(self._async_manager.create_chat(title))

    def add_message(self, chat_id, role, text):
        return self._run(self._async_manager.add_message(chat_id, role, text))

    def delete_chat(self, chat_id):
        return self._run(self._async_manager.delete_chat(chat_id))

    def clear_chat_messages(self, chat_id):
        return self._run(self._async_manager.clear_chat_messages(chat_id))

    def clear_history(self):
        return self._run(self._async_manager.clear_history())

    def save_ticket_draft(self, *args, **kwargs):
        return self._run(self._async_manager.save_ticket_draft(*args, **kwargs))

"""
Менеджер истории чата с поддержкой суммаризации.

Хранит историю сообщений в PostgreSQL (через asyncpg) и автоматически создает резюме
для длинных разговоров.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

import asyncpg

from src.core.config import Config

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """
    Управление историями чата с автоматической суммаризацией (PostgreSQL).
    """

    def __init__(self, config: Config):
        self.config = config
        self.db_url = config.database_url
        self._pools: Dict[asyncio.AbstractEventLoop, asyncpg.Pool] = {}
        self._locks: Dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self._initialized_loops: Set[asyncio.AbstractEventLoop] = set()

    async def get_pool(self) -> asyncpg.Pool:
        """Получить пул соединений (ленивая инициализация для текущего event loop)."""
        loop = asyncio.get_running_loop()
        
        if loop in self._pools and loop in self._initialized_loops:
            return self._pools[loop]
            
        if loop not in self._locks:
            self._locks[loop] = asyncio.Lock()
            
        async with self._locks[loop]:
            # Двойная проверка
            if loop in self._pools and loop in self._initialized_loops:
                return self._pools[loop]
                
            pool = await asyncpg.create_pool(
                self.db_url,
                min_size=self.config.db_pool_min_size,
                max_size=self.config.db_pool_max_size
            )
            self._pools[loop] = pool
            # Инициализируем БД только один раз на весь запуск (но пул создаем для каждого loop)
            # Чтобы не делать это конкурентно, метод _init_database внутри себя использует этот же пул
            # Но get_pool возвращает пул. Мы временно помечаем как инициализированный для текущего loop,
            # чтобы _init_database мог делать запросы без бесконечной рекурсии.
            self._initialized_loops.add(loop)
            
            try:
                # Инициализация схемы и таблиц
                async with pool.acquire() as conn:
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            platform TEXT NOT NULL,
                            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                            message TEXT NOT NULL,
                            timestamp DOUBLE PRECISION NOT NULL,
                            metadata TEXT DEFAULT '{}'
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_session_time
                        ON chat_messages(session_id, timestamp DESC);
                        
                        CREATE TABLE IF NOT EXISTS chat_summaries (
                            session_id TEXT PRIMARY KEY,
                            summary TEXT NOT NULL,
                            messages_count INTEGER NOT NULL,
                            last_updated DOUBLE PRECISION NOT NULL
                        );
                        
                        CREATE TABLE IF NOT EXISTS users (
                            user_id TEXT PRIMARY KEY,
                            company_id TEXT,
                            voice_mode BOOLEAN DEFAULT TRUE
                        );
                    """)
                    await conn.execute("""
                        ALTER TABLE users ADD COLUMN IF NOT EXISTS voice_mode BOOLEAN DEFAULT TRUE;
                    """)
                # Дополнительные колонки для админки
                # Чтобы не вызывать бесконечную рекурсию, временно отключаем вызов get_pool() внутри _ensure_admin_schema,
                # или передаем туда pool напрямую. Но так как self._initialized_loops уже содержит loop,
                # вызов self.get_pool() внутри _ensure_admin_schema вернет pool мгновенно без блокировок.
                await self._ensure_admin_schema()
            except Exception as e:
                logger.error(f"Error during db init: {e}")
                # Если произошла ошибка, сбросим флаг
                self._initialized_loops.discard(loop)
                raise e
                
            return pool

    async def close(self):
        """Закрытие всех пулов соединений."""
        for pool in list(self._pools.values()):
            try:
                await pool.close()
            except Exception:
                pass
        self._pools.clear()
        self._initialized_loops.clear()

    async def save_message(
        self,
        session_id: str,
        platform: str,
        role: str,
        message: str,
        metadata: Optional[Dict] = None,
    ):
        pool = await self.get_pool()
        metadata_str = json.dumps(metadata or {})
        timestamp = time.time()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_messages (session_id, platform, role, message, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                session_id, platform, role, message, timestamp, metadata_str
            )
        logger.debug(f"Message saved for session {session_id}, role: {role}")

    async def get_history(self, session_id: str, max_messages: Optional[int] = None) -> List[Dict[str, any]]:
        if max_messages is None:
            max_messages = self.config.max_history_messages

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, message, timestamp, metadata
                FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                session_id, max_messages
            )

        messages = []
        for row in reversed(rows):
            metadata_str = row['metadata']
            metadata = json.loads(metadata_str) if metadata_str else {}
            messages.append(
                {
                    "role": row['role'],
                    "content": row['message'],
                    "timestamp": row['timestamp'],
                    "metadata": metadata,
                }
            )

        logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
        return messages

    async def get_message_count(self, session_id: str) -> int:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = $1",
                session_id
            )
            return count or 0

    async def get_user_company(self, user_id: str) -> Optional[str]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            company_id = await conn.fetchval(
                "SELECT company_id FROM users WHERE user_id = $1",
                str(user_id)
            )
            logger.debug(f"DB: Чтение компании для {user_id} -> {company_id}")
            return company_id

    async def set_user_company(self, user_id: str, company_id: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, company_id)
                VALUES ($1, $2)
                ON CONFLICT (user_id) 
                DO UPDATE SET company_id = EXCLUDED.company_id
                """,
                str(user_id), company_id
            )
        logger.info(f"DB: Компания для {user_id} установлена в {company_id}")

    async def get_voice_mode(self, user_id: str) -> bool:
        """True = голосовые ответы включены (по умолчанию), False = отвечает текстом."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT voice_mode FROM users WHERE user_id = $1",
                str(user_id)
            )
        # None = записи нет → используем значение по умолчанию (True)
        return bool(value) if value is not None else True

    async def set_voice_mode(self, user_id: str, enabled: bool):
        """True = голосом, False = текстом."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (user_id, voice_mode)
                VALUES ($1, $2)
                ON CONFLICT (user_id)
                DO UPDATE SET voice_mode = EXCLUDED.voice_mode
                """,
                str(user_id), enabled
            )
        logger.info(f"DB: voice_mode для {user_id} установлен в {enabled}")

    async def get_summary(self, session_id: str) -> Optional[str]:
        stats = await self.get_summary_stats(session_id)
        return stats["summary"] if stats else None

    async def get_summary_stats(self, session_id: str) -> Optional[Dict[str, any]]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT summary, messages_count FROM chat_summaries WHERE session_id = $1",
                session_id
            )
            if row:
                return {"summary": row['summary'], "messages_count": row['messages_count']}
            return None

    async def save_summary(self, session_id: str, summary: str, messages_count: int):
        pool = await self.get_pool()
        timestamp = time.time()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_summaries (session_id, summary, messages_count, last_updated)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (session_id) 
                DO UPDATE SET 
                    summary = EXCLUDED.summary, 
                    messages_count = EXCLUDED.messages_count, 
                    last_updated = EXCLUDED.last_updated
                """,
                session_id, summary, messages_count, timestamp
            )
        logger.info(f"Summary saved for session {session_id}, {messages_count} messages summarized")

    async def check_summarization_needed(self, session_id: str) -> bool:
        if not self.config.enable_auto_summarization:
            return False

        stats = await self.get_summary_stats(session_id)
        summarized_count = stats["messages_count"] if stats else 0
        
        message_count = await self.get_message_count(session_id)
        new_messages = message_count - summarized_count
        
        return new_messages > self.config.summarization_threshold

    async def get_old_messages_for_summarization(
        self, session_id: str, keep_recent: int, offset: int = 0
    ) -> List[Dict[str, any]]:
        total_count = await self.get_message_count(session_id)
        if total_count <= keep_recent or total_count <= offset:
            return []

        limit = total_count - keep_recent - offset
        if limit <= 0:
            return []

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, message, timestamp, metadata
                FROM chat_messages
                WHERE session_id = $1
                ORDER BY timestamp ASC
                LIMIT $2 OFFSET $3
                """,
                session_id, limit, offset
            )

        messages = []
        for row in rows:
            metadata_str = row['metadata']
            metadata = json.loads(metadata_str) if metadata_str else {}
            messages.append(
                {
                    "role": row['role'],
                    "content": row['message'],
                    "timestamp": row['timestamp'],
                    "metadata": metadata,
                }
            )

        return messages

    def format_history_for_llm(self, messages: List[Dict], summary: Optional[str] = None) -> List[Dict[str, str]]:
        history = []

        if summary:
            history.append({"role": "user", "content": f"РЕЗЮМЕ ПРЕДЫДУЩЕГО РАЗГОВОРА:\n{summary}"})
            history.append(
                {
                    "role": "model",
                    "content": "Понял, продолжаю разговор с учетом резюме.",
                }
            )

        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            history.append({"role": role, "content": msg["content"]})

        return history

    async def clear_history(self, session_id: str, clear_summary: bool = True):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chat_messages WHERE session_id = $1", session_id)
                if clear_summary:
                    await conn.execute("DELETE FROM chat_summaries WHERE session_id = $1", session_id)
        logger.info(f"History cleared for session {session_id}")

    # ─── Методы для панели администратора ───────────────────────────────────

    async def _ensure_admin_schema(self):
        """Создание дополнительных колонок и таблиц для функций администратора."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # 1. Дополнительные колонки в users
            await conn.execute("""
                ALTER TABLE users ADD COLUMN IF NOT EXISTS is_blocked BOOLEAN DEFAULT FALSE;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS last_activity DOUBLE PRECISION;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS platform TEXT;
                ALTER TABLE users ADD COLUMN IF NOT EXISTS first_seen DOUBLE PRECISION;
            """)
            
            # 2. Таблица admin_users
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    company_id TEXT,
                    permissions TEXT NOT NULL DEFAULT '[]'
                );
            """)

            # 3. Инициализация суперадмина по умолчанию
            admin_exists = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM admin_users WHERE username = 'admin')")
            if not admin_exists:
                default_password = self.config.admin_password
                pwd_hash = hash_password(default_password)
                all_permissions = json.dumps([
                    "view_stats", "view_logs", "manage_bot_users", "view_documents", 
                    "add_documents", "edit_documents", "delete_documents", 
                    "apply_changes", "send_broadcast", "manage_api_keys"
                ])
                await conn.execute("""
                    INSERT INTO admin_users (username, password_hash, role, company_id, permissions)
                    VALUES ($1, $2, $3, $4, $5)
                """, 'admin', pwd_hash, 'superadmin', 'all', all_permissions)
                logger.info("Default superadmin 'admin' created in database.")

    async def get_admin_user_by_username(self, username: str) -> Optional[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, username, password_hash, role, company_id, permissions
                FROM admin_users
                WHERE username = $1
            """, username)
            if row:
                def parse_company_ids(val: Optional[str]) -> List[str]:
                    if not val:
                        return []
                    if val == "all":
                        return ["all"]
                    if val.startswith("[") and val.endswith("]"):
                        try:
                            return json.loads(val)
                        except Exception:
                            return [val]
                    return [val]
                
                c_ids = parse_company_ids(row["company_id"])
                return {
                    "id": row["id"],
                    "username": row["username"],
                    "password_hash": row["password_hash"],
                    "role": row["role"],
                    "company_id": row["company_id"],
                    "company_ids": c_ids,
                    "permissions": json.loads(row["permissions"]),
                }
            return None

    async def get_all_admin_users(self) -> List[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, username, role, company_id, permissions
                FROM admin_users
                ORDER BY username ASC
            """)
        
        def parse_company_ids(val: Optional[str]) -> List[str]:
            if not val:
                return []
            if val == "all":
                return ["all"]
            if val.startswith("[") and val.endswith("]"):
                try:
                    return json.loads(val)
                except Exception:
                    return [val]
            return [val]

        result = []
        for row in rows:
            c_ids = parse_company_ids(row["company_id"])
            result.append({
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "company_id": row["company_id"],
                "company_ids": c_ids,
                "permissions": json.loads(row["permissions"]),
            })
        return result

    async def create_admin_user(self, username: str, password_hash: str, role: str, company_id: Optional[str], permissions: List[str]) -> int:
        pool = await self.get_pool()
        permissions_str = json.dumps(permissions)
        async with pool.acquire() as conn:
            val = await conn.fetchval("""
                INSERT INTO admin_users (username, password_hash, role, company_id, permissions)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, username, password_hash, role, company_id, permissions_str)
            return val

    async def update_admin_user(self, user_id: int, username: str, password_hash: Optional[str], role: str, company_id: Optional[str], permissions: List[str]):
        pool = await self.get_pool()
        permissions_str = json.dumps(permissions)
        async with pool.acquire() as conn:
            if password_hash:
                await conn.execute("""
                    UPDATE admin_users
                    SET username = $2, password_hash = $3, role = $4, company_id = $5, permissions = $6
                    WHERE id = $1
                """, user_id, username, password_hash, role, company_id, permissions_str)
            else:
                await conn.execute("""
                    UPDATE admin_users
                    SET username = $2, role = $3, company_id = $4, permissions = $5
                    WHERE id = $1
                """, user_id, username, role, company_id, permissions_str)

    async def delete_admin_user(self, user_id: int):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM admin_users WHERE id = $1", user_id)

    async def update_last_activity(self, user_id: str, platform: str):
        """Обновить время последней активности пользователя."""
        pool = await self.get_pool()
        now = time.time()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (user_id, last_activity, platform, first_seen)
                VALUES ($1, $2, $3, $2)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    last_activity = EXCLUDED.last_activity,
                    platform = COALESCE(EXCLUDED.platform, users.platform),
                    first_seen = COALESCE(users.first_seen, EXCLUDED.first_seen)
            """, str(user_id), now, platform)

    async def get_all_users(self, limit: int = 500, offset: int = 0, company_ids: Optional[List[str]] = None) -> List[Dict]:
        """Список всех пользователей для панели администратора."""
        pool = await self.get_pool()
        if company_ids and 'all' in company_ids:
            company_ids = None
        async with pool.acquire() as conn:
            if company_ids:
                rows = await conn.fetch("""
                    SELECT
                        user_id,
                        company_id,
                        voice_mode,
                        COALESCE(is_blocked, FALSE) as is_blocked,
                        last_activity,
                        platform,
                        first_seen
                    FROM users
                    WHERE company_id = ANY($3::text[])
                    ORDER BY COALESCE(last_activity, 0) DESC
                    LIMIT $1 OFFSET $2
                """, limit, offset, company_ids)
            else:
                rows = await conn.fetch("""
                    SELECT
                        user_id,
                        company_id,
                        voice_mode,
                        COALESCE(is_blocked, FALSE) as is_blocked,
                        last_activity,
                        platform,
                        first_seen
                    FROM users
                    ORDER BY COALESCE(last_activity, 0) DESC
                    LIMIT $1 OFFSET $2
                """, limit, offset)
        result = []
        for row in rows:
            result.append({
                "user_id": row["user_id"],
                "company_id": row["company_id"],
                "voice_mode": row["voice_mode"],
                "is_blocked": row["is_blocked"],
                "last_activity": row["last_activity"],
                "platform": row["platform"],
                "first_seen": row["first_seen"],
            })
        return result

    async def get_users_count(self, company_ids: Optional[List[str]] = None) -> int:
        """Общее количество пользователей."""
        pool = await self.get_pool()
        if company_ids and 'all' in company_ids:
            company_ids = None
        async with pool.acquire() as conn:
            if company_ids:
                return await conn.fetchval("SELECT COUNT(*) FROM users WHERE company_id = ANY($1::text[])", company_ids) or 0
            return await conn.fetchval("SELECT COUNT(*) FROM users") or 0

    async def block_user(self, user_id: str):
        """Заблокировать пользователя."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (user_id, is_blocked)
                VALUES ($1, TRUE)
                ON CONFLICT (user_id)
                DO UPDATE SET is_blocked = TRUE
            """, str(user_id))
        logger.info(f"User {user_id} blocked")

    async def unblock_user(self, user_id: str):
        """Разблокировать пользователя."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET is_blocked = FALSE WHERE user_id = $1
            """, str(user_id))
        logger.info(f"User {user_id} unblocked")

    async def is_user_blocked(self, user_id: str) -> bool:
        """Проверить, заблокирован ли пользователь."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT is_blocked FROM users WHERE user_id = $1", str(user_id)
            )
        return bool(val) if val is not None else False

    async def get_stats(self, company_ids: Optional[List[str]] = None) -> Dict:
        """Статистика для дашборда с опциональной фильтрацией по компаниям."""
        pool = await self.get_pool()
        now = time.time()
        day_ago = now - 86400
        week_ago = now - 86400 * 7

        if company_ids and 'all' in company_ids:
            company_ids = None

        async with pool.acquire() as conn:
            if company_ids:
                total_messages = await conn.fetchval("""
                    SELECT COUNT(*) FROM chat_messages m
                    LEFT JOIN users u ON m.session_id = u.user_id
                    WHERE u.company_id = ANY($1::text[])
                """, company_ids) or 0
                
                today_messages = await conn.fetchval("""
                    SELECT COUNT(*) FROM chat_messages m
                    LEFT JOIN users u ON m.session_id = u.user_id
                    WHERE m.timestamp > $1 AND u.company_id = ANY($2::text[])
                """, day_ago, company_ids) or 0
                
                week_messages = await conn.fetchval("""
                    SELECT COUNT(*) FROM chat_messages m
                    LEFT JOIN users u ON m.session_id = u.user_id
                    WHERE m.timestamp > $1 AND u.company_id = ANY($2::text[])
                """, week_ago, company_ids) or 0
                
                total_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE company_id = ANY($1::text[])", company_ids) or 0
                
                active_today = await conn.fetchval("""
                    SELECT COUNT(DISTINCT m.session_id) FROM chat_messages m
                    LEFT JOIN users u ON m.session_id = u.user_id
                    WHERE m.timestamp > $1 AND u.company_id = ANY($2::text[])
                """, day_ago, company_ids) or 0

                hourly_rows = await conn.fetch("""
                    SELECT
                        EXTRACT(EPOCH FROM to_timestamp(m.timestamp) AT TIME ZONE 'UTC'
                            - INTERVAL '1 second' * (EXTRACT(EPOCH FROM to_timestamp(m.timestamp) AT TIME ZONE 'UTC')::bigint % 3600)) AS hour_ts,
                        COUNT(*) as cnt
                    FROM chat_messages m
                    LEFT JOIN users u ON m.session_id = u.user_id
                    WHERE m.timestamp > $1 AND u.company_id = ANY($2::text[])
                    GROUP BY hour_ts
                    ORDER BY hour_ts
                """, day_ago, company_ids)

                daily_rows = await conn.fetch("""
                    SELECT
                        DATE(to_timestamp(m.timestamp)) as day,
                        COUNT(*) as cnt
                    FROM chat_messages m
                    LEFT JOIN users u ON m.session_id = u.user_id
                    WHERE m.timestamp > $1 AND u.company_id = ANY($2::text[])
                    GROUP BY day
                    ORDER BY day
                """, week_ago, company_ids)
            else:
                total_messages = await conn.fetchval("SELECT COUNT(*) FROM chat_messages") or 0
                today_messages = await conn.fetchval(
                    "SELECT COUNT(*) FROM chat_messages WHERE timestamp > $1", day_ago
                ) or 0
                week_messages = await conn.fetchval(
                    "SELECT COUNT(*) FROM chat_messages WHERE timestamp > $1", week_ago
                ) or 0
                total_users = await conn.fetchval("SELECT COUNT(*) FROM users") or 0
                active_today = await conn.fetchval(
                    "SELECT COUNT(DISTINCT session_id) FROM chat_messages WHERE timestamp > $1",
                    day_ago
                ) or 0

                hourly_rows = await conn.fetch("""
                    SELECT
                        EXTRACT(EPOCH FROM to_timestamp(timestamp) AT TIME ZONE 'UTC'
                            - INTERVAL '1 second' * (EXTRACT(EPOCH FROM to_timestamp(timestamp) AT TIME ZONE 'UTC')::bigint % 3600)) AS hour_ts,
                        COUNT(*) as cnt
                    FROM chat_messages
                    WHERE timestamp > $1
                    GROUP BY hour_ts
                    ORDER BY hour_ts
                """, day_ago)

                daily_rows = await conn.fetch("""
                    SELECT
                        DATE(to_timestamp(timestamp)) as day,
                        COUNT(*) as cnt
                    FROM chat_messages
                    WHERE timestamp > $1
                    GROUP BY day
                    ORDER BY day
                """, week_ago)

        return {
            "total_messages": total_messages,
            "today_messages": today_messages,
            "week_messages": week_messages,
            "total_users": total_users,
            "active_today": active_today,
            "hourly": [{"ts": float(r["hour_ts"]), "count": r["cnt"]} for r in hourly_rows],
            "daily": [{"day": str(r["day"]), "count": r["cnt"]} for r in daily_rows],
        }

    async def get_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        search: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        company_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Получение истории сообщений для лога в панели администратора."""
        pool = await self.get_pool()
        conditions = []
        params = []
        idx = 1

        if user_id:
            conditions.append(f"session_id = ${idx}")
            params.append(user_id)
            idx += 1
        if platform:
            conditions.append(f"platform = ${idx}")
            params.append(platform)
            idx += 1
        if search:
            conditions.append(f"message ILIKE ${idx}")
            params.append(f"%{search}%")
            idx += 1
        if date_from:
            conditions.append(f"timestamp >= ${idx}")
            params.append(date_from)
            idx += 1
        if date_to:
            conditions.append(f"timestamp <= ${idx}")
            params.append(date_to)
            idx += 1
        if company_ids and 'all' not in company_ids:
            conditions.append(f"session_id IN (SELECT user_id FROM users WHERE company_id = ANY(${idx}::text[]))")
            params.append(company_ids)
            idx += 1

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.extend([limit, offset])

        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT session_id, platform, role, message, timestamp, metadata
                FROM chat_messages
                {where}
                ORDER BY timestamp DESC
                LIMIT ${idx} OFFSET ${idx + 1}
            """, *params)

        result = []
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            result.append({
                "session_id": row["session_id"],
                "platform": row["platform"],
                "role": row["role"],
                "message": row["message"],
                "timestamp": row["timestamp"],
                "metadata": meta,
            })
        return result

    async def get_logs_count(
        self,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        search: Optional[str] = None,
        company_ids: Optional[List[str]] = None,
    ) -> int:
        """Общее количество сообщений для пагинации."""
        pool = await self.get_pool()
        conditions = []
        params = []
        idx = 1

        if user_id:
            conditions.append(f"session_id = ${idx}")
            params.append(user_id)
            idx += 1
        if platform:
            conditions.append(f"platform = ${idx}")
            params.append(platform)
            idx += 1
        if search:
            conditions.append(f"message ILIKE ${idx}")
            params.append(f"%{search}%")
            idx += 1
        if company_ids and 'all' not in company_ids:
            conditions.append(f"session_id IN (SELECT user_id FROM users WHERE company_id = ANY(${idx}::text[]))")
            params.append(company_ids)
            idx += 1

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        async with pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT COUNT(*) FROM chat_messages {where}", *params
            ) or 0


# ─── Хелперы авторизации ───────────────────────────────────────────────────

def hash_password(password: str) -> str:
    import hashlib
    import os
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + ":" + key.hex()


def verify_password(stored_password_hash: str, provided_password: str) -> bool:
    import hashlib
    try:
        salt_hex, key_hex = stored_password_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return new_key == key
    except Exception:
        return False


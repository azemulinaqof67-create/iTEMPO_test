import asyncio
from src.core.config import Config
from src.storage.chat_history import ChatHistoryManager

async def main():
    config = Config.from_env()
    db = ChatHistoryManager(config)
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        # Проверяем существование пользователя 43105022
        user_exists = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM users WHERE user_id = '43105022')")
        print(f"Пользователь 43105022 существует: {user_exists}")
        
        # Обновим никнейм для 43105022
        await conn.execute("UPDATE users SET username = 'Иван (@ivan_test)' WHERE user_id = '43105022'")
        print("Никнейм для 43105022 обновлен.")
        
        # Обновим никнейм для max_29412356
        await conn.execute("UPDATE users SET username = 'Максим (MAX)' WHERE user_id = 'max_29412356'")
        print("Никнейм для max_29412356 обновлен.")

if __name__ == "__main__":
    asyncio.run(main())

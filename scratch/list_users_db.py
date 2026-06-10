import asyncio
from src.core.config import Config
from src.storage.chat_history import ChatHistoryManager

async def main():
    config = Config.from_env()
    db = ChatHistoryManager(config)
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        users = await conn.fetch("SELECT * FROM users LIMIT 10")
        print("Users in DB:")
        for u in users:
            print(dict(u))
            
        messages = await conn.fetch("SELECT DISTINCT session_id FROM chat_messages LIMIT 10")
        print("\nSessions in chat_messages:")
        for m in messages:
            print(dict(m))

if __name__ == "__main__":
    asyncio.run(main())

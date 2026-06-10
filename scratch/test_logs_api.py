import asyncio
from src.core.config import Config
from src.storage.chat_history import ChatHistoryManager

async def main():
    config = Config.from_env()
    db = ChatHistoryManager(config)
    logs = await db.get_logs(limit=10)
    print("Logs from DB:")
    for log in logs:
        print(f"session_id: {log.get('session_id')}, username: {log.get('username')}, platform: {log.get('platform')}")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import sys
from pathlib import Path

# selector loop для Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.clients import ClientManager
from src.core.config import Config

async def count_points():
    config = Config.from_env()
    client_manager = ClientManager.get_instance(config)
    client = client_manager.get_qdrant_client()
    
    for coll in ["documents_v2", "contacts_v1"]:
        if client.collection_exists(coll):
            info = client.get_collection(coll)
            print(f"Коллекция {coll}: {info.points_count} точек")
        else:
            print(f"Коллекция {coll} НЕ существует")
            
    client_manager.close_all()

if __name__ == "__main__":
    asyncio.run(count_points())

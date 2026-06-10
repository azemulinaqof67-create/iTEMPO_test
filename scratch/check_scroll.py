import asyncio
from qdrant_client import models
from src.core.clients import ClientManager
from src.core.config import Config

async def test_scroll():
    config = Config.from_env()
    client_manager = ClientManager.get_instance(config)
    client = client_manager.get_qdrant_client()
    
    words = "АЙТИ ТЭМПО".split()
    must_conds = [
        models.FieldCondition(key="company", match=models.MatchText(text=word)) for word in words
    ]
    q_filter_words = models.Filter(must=must_conds)
    
    print("Scroll filter words:", q_filter_words)
    result = client.scroll(collection_name="contacts_v1", scroll_filter=q_filter_words, limit=10, with_payload=True)
    points = result[0]
    print(f"Найдено {len(points)} точек.")
    for p in points:
        print(f"{p.payload.get('full_name')} | {p.payload.get('company')}")

if __name__ == "__main__":
    asyncio.run(test_scroll())

import asyncio
import sys
from pathlib import Path
from qdrant_client import models

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.clients import ClientManager
from src.core.config import Config

async def test_filters():
    config = Config.from_env()
    client_manager = ClientManager.get_instance(config)
    client = client_manager.get_qdrant_client()
    
    collection_name = config.collection_name
    print(f"Collection: {collection_name}")
    
    # Вариант 1: Только фильтр по компании (technotron OR shared) в must
    filter1 = models.Filter(
        must=[
            models.Filter(
                should=[
                    models.FieldCondition(key="company_tag", match=models.MatchValue(value="technotron")),
                    models.FieldCondition(key="company_tag", match=models.MatchValue(value="shared")),
                ]
            )
        ]
    )
    res1 = client.scroll(collection_name=collection_name, scroll_filter=filter1, limit=5)
    print(f"Фильтр 1 (must company_tag) вернул: {len(res1[0])} точек")
    for p in res1[0]:
        print(f"  - {p.payload.get('source')} (company_tag: {p.payload.get('company_tag')})")
        
    # Вариант 2: Фильтр по компании в must + should по department
    filter2 = models.Filter(
        must=[
            models.Filter(
                should=[
                    models.FieldCondition(key="company_tag", match=models.MatchValue(value="technotron")),
                    models.FieldCondition(key="company_tag", match=models.MatchValue(value="shared")),
                ]
            )
        ],
        should=[
            models.FieldCondition(key="department", match=models.MatchAny(any=["HR", "Routine", "General"]))
        ]
    )
    res2 = client.scroll(collection_name=collection_name, scroll_filter=filter2, limit=5)
    print(f"Фильтр 2 (must company_tag + should department) вернул: {len(res2[0])} точек")
    
    # Вариант 3: Просто scroll без фильтров вообще
    res3 = client.scroll(collection_name=collection_name, limit=5)
    print(f"Фильтр 3 (без фильтров) вернул: {len(res3[0])} точек")
    for p in res3[0]:
        print(f"  - {p.payload.get('source')} (company_tag: {p.payload.get('company_tag')})")

    client_manager.close_all()

if __name__ == "__main__":
    asyncio.run(test_filters())

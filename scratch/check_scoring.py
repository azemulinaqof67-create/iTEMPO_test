import asyncio
from src.rag.retrieval.contact_hybrid_search import ContactHybridSearch
from src.core.config import Config

async def test_search():
    config = Config.from_env()
    search = ContactHybridSearch(config)
    
    print("--- Тестирование запроса 'сотрудники' + company 'АЙТИ ТЭМПО' ---")
    results = await search.search(semantic_query="сотрудники", company_filter="АЙТИ ТЭМПО", limit=30)
    for i, r in enumerate(results):
        print(f"{i+1}. Score: {r['score']:.4f} | {r['full_name']} | {r['company']} | {r['position']}")

    print("\n--- Тестирование запроса '' + company 'АЙТИ ТЭМПО' ---")
    results = await search.search(semantic_query="", company_filter="АЙТИ ТЭМПО", limit=30)
    for i, r in enumerate(results):
        print(f"{i+1}. Score: {r['score']:.4f} | {r['full_name']} | {r['company']} | {r['position']}")

if __name__ == "__main__":
    asyncio.run(test_search())

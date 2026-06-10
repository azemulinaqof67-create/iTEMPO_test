import asyncio
import logging
from src.core.config import Config
from src.rag.retrieval.contact_hybrid_search import ContactHybridSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_search():
    config = Config.from_env()
    search_service = ContactHybridSearch(config)

    # Тест 1: Семантический поиск по ФИО
    logger.info("=== Тест 1: Поиск 'Крылов Олег' ===")
    results = await search_service.search(semantic_query="Крылов Олег")
    for r in results:
        print(f"ФИО: {r['full_name']} | Компания: {r['company']} | Должность: {r['position']} | Тел: {r['phone']} | Score: {r['score']:.4f}")

    # Тест 2: Поиск по частичному номеру телефона
    logger.info("\n=== Тест 2: Поиск по части телефона '912' ===")
    results = await search_service.search(exact_phone="912")
    for r in results:
        print(f"ФИО: {r['full_name']} | Компания: {r['company']} | Тел: {r['phone']}")

    # Тест 3: Семантический поиск + Фильтр по компании
    logger.info("\n=== Тест 3: Поиск 'директор' с фильтром по компании 'ЗТЭО' ===")
    results = await search_service.search(semantic_query="директор", company_filter="ЗТЭО")
    for r in results:
        print(f"ФИО: {r['full_name']} | Компания: {r['company']} | Должность: {r['position']} | Score: {r['score']:.4f}")

if __name__ == "__main__":
    asyncio.run(test_search())

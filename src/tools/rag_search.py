import logging

from qdrant_client import models

from src.core.config import Config
from src.models.state import QueryIntent
from src.rag.retrieval.search import SearchService
from src.utils.company_mapper import normalize_company_id

logger = logging.getLogger(__name__)


class FilteredRAGTool:
    def __init__(self):
        self.config = Config.from_env()
        self.search_service = SearchService(self.config)

    async def initialize(self):
        """Прогрев поисковых индексов."""
        await self.search_service.initialize()

    async def search(self, query: str, intent_data: QueryIntent) -> str:
        """
        Выполняет поиск с трансляцией интентов в фильтры Qdrant.
        Применяет строгий (must) фильтр по company_tag, чтобы гарантировать
        изоляцию данных между предприятиями.
        """
        try:
            must_conditions = []
            should_conditions = []

            # 1. Обязательная фильтрация по компании (MUST)
            company_id = intent_data.target_company

            if company_id:
                # Нормализуем ID компании к короткому ключу (например, "КМК" → "kmk")
                # для фильтрации по полю company_tag, которое хранит папочный ключ
                company_tag_id = normalize_company_id(company_id)

                # Строгий фильтр: документ ОБЯЗАН принадлежать компании пользователя
                # ИЛИ быть в общей папке (shared). Никаких исключений для других компаний.
                company_access_filter = models.Filter(
                    should=[
                        # Документы выбранной компании (по папочному тегу data/<company_tag>/)
                        models.FieldCondition(key="company_tag", match=models.MatchValue(value=company_tag_id)),
                        # Общие документы холдинга (папка data/shared/)
                        models.FieldCondition(key="company_tag", match=models.MatchValue(value="shared")),
                    ]
                )
                must_conditions.append(company_access_filter)
                logger.info(f"--- RAG COMPANY FILTER (STRICT): tag='{company_tag_id}' OR 'shared' ---")

            # Создание объекта фильтра Qdrant (только обязательные must-условия для изоляции компаний)
            qdrant_filter = models.Filter(must=must_conditions if must_conditions else None)

            logger.info("--- RAG SEARCH WITH FILTERS ---")
            logger.info(f"Query: {query}")
            logger.info(f"Intent: {intent_data.intent}")
            logger.info(f"Company ID: {company_id}")
            if must_conditions or should_conditions:
                logger.info(f"Qdrant Filters: {len(must_conditions)} must, {len(should_conditions)} should")

            # Выполнение гибридного поиска
            search_result = await self.search_service.search(
                query=query, limit=10, company_id=company_id, qdrant_filter=qdrant_filter, intent=intent_data.intent
            )

            if not search_result.chunks:
                logger.warning(f"⚠️ No results found for query: {query}")
                if intent_data.intent == "emergency":
                    return (
                        "В базе знаний не найдена конкретная инструкция, но ПРИ ЧРЕЗВЫЧАЙНОЙ СИТУАЦИИ:\n"
                        "1. Немедленно сообщите руководителю.\n"
                        "2. Вызовите скорую помощь (103/112) или обратитесь в ближайший медпункт (АБК-3).\n"
                        "3. Свяжитесь со службой безопасности."
                    )
                return "В базе знаний ничего не найдено по вашему запросу."

            # Логируем что именно нашли
            logger.info(f"✅ Found {len(search_result.chunks)} chunks")
            for i, chunk in enumerate(search_result.chunks[:3]):
                source = (
                    getattr(search_result.documents[i], "source", "Unknown")
                    if i < len(search_result.documents)
                    else "Unknown"
                )
                logger.info(f"Chunk {i + 1} from {source} (len: {len(chunk)}): {chunk[:150]}...")

            # Возвращаем очищенные чанки (без скоров) для LLM
            clean_chunks = SearchService.clean_scores(search_result.chunks)
            return "\n\n".join(clean_chunks)

        except Exception as e:
            logger.error(f"FilteredRAGTool error: {e}")
            return "Извините, произошла техническая ошибка при поиске в базе знаний."

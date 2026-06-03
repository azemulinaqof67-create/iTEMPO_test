import logging
from typing import Dict, Any, Optional
from qdrant_client import models
from src.models.state import QueryIntent
from src.rag.retrieval.search import SearchService
from src.core.config import Config
from src.utils.company_mapper import get_full_company_names

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
        """
        try:
            must_conditions = []
            should_conditions = []
            
            # 1. Обязательная фильтрация по компании (MUST)
            company_id = intent_data.target_company
            is_topic_shift = intent_data.is_topic_shift

            if company_id and not is_topic_shift:
                # Получаем официальные названия из единого маппера
                full_names = get_full_company_names(company_id)
                if not full_names:
                    full_names = [company_id]
                
                # Ищем документы конкретной компании ИЛИ общие документы холдинга
                should_matches = [
                    models.FieldCondition(key="company", match=models.MatchText(text=company_id)),
                    models.FieldCondition(key="company", match=models.MatchValue(value="ГК «ТЭМПО»")),
                    models.FieldCondition(key="department", match=models.MatchValue(value="General")),
                    models.FieldCondition(key="company_tag", match=models.MatchValue(value="shared")),
                    models.FieldCondition(key="metadata.organization", match=models.MatchValue(value="shared"))
                ]
                
                for name in full_names:
                    should_matches.append(
                        models.FieldCondition(key="company", match=models.MatchValue(value=name))
                    )

                company_filter = models.Filter(should=should_matches)
                should_conditions.append(company_filter)
            elif is_topic_shift:
                should_conditions.append(
                    models.FieldCondition(key="department", match=models.MatchValue(value="General"))
                )
            
            # 2. Рекомендательная фильтрация (SHOULD) - не блокирует, а помогает ранжированию
            if intent_data.intent == "emergency":
                should_conditions.extend([
                    models.FieldCondition(
                        key="tags",
                        match=models.MatchAny(any=["инцидент", "травма", "ЧС", "скорая", "помощь", "пожар", "безопасность"])
                    ),
                    models.FieldCondition(key="department", match=models.MatchValue(value="Security")),
                    models.FieldCondition(key="department", match=models.MatchValue(value="Safety"))
                ])
            elif intent_data.intent == "hr_policy":
                should_conditions.append(
                    models.FieldCondition(key="department", match=models.MatchAny(any=["HR", "Routine", "General"]))
                )

            # Создание объекта фильтра Qdrant
            qdrant_filter = models.Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None
            )
            
            logger.info(f"--- RAG SEARCH WITH FILTERS ---")
            logger.info(f"Query: {query}")
            logger.info(f"Intent: {intent_data.intent}")
            if must_conditions or should_conditions:
                logger.info(f"Qdrant Filters: {len(must_conditions)} must, {len(should_conditions)} should")

            # Выполнение гибридного поиска
            search_result = await self.search_service.search(
                query=query, 
                limit=10,
                company_id=company_id,
                qdrant_filter=qdrant_filter,
                intent=intent_data.intent
            )
            
            if not search_result.chunks:
                logger.warning(f"⚠️ No results found for query: {query}")
                if intent_data.intent == "emergency":
                    return ("В базе знаний не найдена конкретная инструкция, но ПРИ ЧРЕЗВЫЧАЙНОЙ СИТУАЦИИ:\n"
                            "1. Немедленно сообщите руководителю.\n"
                            "2. Вызовите скорую помощь (103/112) или обратитесь в ближайший медпункт (АБК-3).\n"
                            "3. Свяжитесь со службой безопасности.")
                return "В базе знаний ничего не найдено по вашему запросу."

            # Логируем что именно нашли
            logger.info(f"✅ Found {len(search_result.chunks)} chunks")
            for i, chunk in enumerate(search_result.chunks[:3]):
                source = getattr(search_result.documents[i], 'source', 'Unknown') if i < len(search_result.documents) else 'Unknown'
                logger.info(f"Chunk {i+1} from {source} (len: {len(chunk)}): {chunk[:150]}...")

            # Возвращаем очищенные чанки (без скоров) для LLM
            clean_chunks = SearchService.clean_scores(search_result.chunks)
            return "\n\n".join(clean_chunks)

        except Exception as e:
            logger.error(f"FilteredRAGTool error: {e}")
            return "Извините, произошла техническая ошибка при поиске в базе знаний."

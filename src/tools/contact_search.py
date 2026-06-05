import logging
from typing import Optional, Any
from pydantic import BaseModel, Field
from src.core.config import Config
from src.rag.retrieval.contact_hybrid_search import ContactHybridSearch

logger = logging.getLogger(__name__)

class ContactSearchInput(BaseModel):
    """Схема входных аргументов для инструмента поиска контактов."""
    semantic_query: str = Field(
        default="",
        description="Имя, фамилия, должность или отдел искомого лица. ВНИМАНИЕ: Оставьте это поле пустым (\"\"), если пользователь просит вывести просто список или всех сотрудников компании/отдела, чтобы избежать искажения сортировки."
    )
    company_filter: Optional[str] = Field(
        default=None,
        description="Название компании для фильтрации контактов (например, 'КМК', 'ЗТЭО', 'ИТЗ')"
    )
    exact_phone: Optional[str] = Field(
        default=None,
        description="Точный или частичный номер телефона для поиска владельца контакта"
    )
    limit: int = Field(
        default=10,
        description="Максимальное количество возвращаемых контактов. Увеличьте это значение (вплоть до 50), если пользователь запрашивает список сотрудников или ищет 'других'."
    )

class ContactSearchTool:
    """Инструмент для поиска контактов в Qdrant с использованием гибридного поиска (Dense + Sparse/BM25)."""
    
    def __init__(self, config: Optional[Config] = None, db_path: Optional[str] = None):
        self.config = config or Config.from_env()
        self.search_service = ContactHybridSearch(self.config)

    async def search(
        self, 
        semantic_query: str = "", 
        company_filter: Optional[str] = None, 
        exact_phone: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Выполняет поиск контактов в Qdrant.
        """
        logger.info(
            f"[QDRANT SEARCH TOOL] query: '{semantic_query}' | company: '{company_filter}' | phone: '{exact_phone}'"
        )

        try:
            params_validated = ContactSearchInput(
                semantic_query=semantic_query,
                company_filter=company_filter,
                exact_phone=exact_phone,
                limit=kwargs.get("limit", 10)
            )
        except Exception as e:
            logger.error(f"Validation error in ContactSearchTool: {e}")
            return f"Ошибка валидации параметров поиска: {e}"

        # Если поисковый запрос состоит только из цифр, а exact_phone не задан,
        # перенаправляем запрос в exact_phone для поиска по номеру
        q_digits = "".join(c for c in params_validated.semantic_query if c.isdigit())
        if q_digits and len(q_digits) >= 4 and not params_validated.exact_phone:
            params_validated.exact_phone = params_validated.semantic_query
            params_validated.semantic_query = ""

        if not params_validated.semantic_query and not params_validated.exact_phone:
            return "Не указано имя, должность или телефон для поиска."

        try:
            rows = await self.search_service.search(
                semantic_query=params_validated.semantic_query,
                company_filter=params_validated.company_filter,
                exact_phone=params_validated.exact_phone,
                limit=params_validated.limit
            )

            if not rows:
                search_term = params_validated.semantic_query or params_validated.exact_phone
                return f"По запросу '{search_term}' ничего не найдено."

            formatted_results = []
            for i, row in enumerate(rows, 1):
                logger.info(f"[QDRANT SEARCH TOOL] Match found: {row['full_name']} | Score: {row.get('score', 0.0):.3f}")
                formatted_results.append(
                    f"{i}. {row['full_name'] or '—'} — {row['position'] or '—'}\n"
                    f"   Отдел: {row['department'] or '—'}, Компания: {row['company'] or '—'}\n"
                    f"   Тел: {row['phone'] or '—'}"
                )

            return "Найдены контакты:\n\n" + "\n\n".join(formatted_results)

        except Exception as e:
            logger.exception(f"ContactSearchTool error: {e}")
            return "Произошла ошибка при поиске в базе контактов."

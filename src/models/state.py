import operator
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Annotated, TypedDict
from langgraph.graph.message import add_messages

class QueryIntent(BaseModel):
    """Схема классификации намерения пользователя."""
    intent: Literal["contact_search", "location_search", "hr_policy", "general_info", "emergency", "weather"] = Field(
        description="Категория запроса пользователя. 'emergency' — для ЧС, травм и медпунктов. 'weather' — для прогноза погоды."
    )
    is_topic_shift: bool = Field(
        default=False,
        description="Установи в True, если пользователь резко сменил тему разговора и предыдущий контекст больше не применим."
    )
    target_company: Optional[str] = Field(
        default=None, 
        description="Название компании или завода (например, 'КМК', 'ЗТЭО', 'ИТЗ', 'АЙТИ')."
    )
    target_person: Optional[str] = Field(
        default=None, 
        description="ФИО, должность или название отдела, если это поиск контактов."
    )
    target_location: Optional[str] = Field(
        default=None,
        description="Название города для поиска погоды (например, 'Набережные Челны')."
    )
    requires_rag: bool = Field(
        default=False,
        description="Установи в True, если запрос пользователя требует обращения к текстовым документам/базе знаний (например, поиск руководства, графиков работы, адресов, регламентов, отпусков, обязанностей сотрудников или любой дополнительной текстовой информации, выходящей за рамки простого поиска контакта/телефона/почты)."
    )

def clearable_add(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """Умный редьюсер: склеивает результаты, но если видит сигнал __CLEAR__, полностью очищает список."""
    if right is None:
        return left if left is not None else []
    if right and right[0] == "__CLEAR__":
        return []
    
    res = left.copy() if left is not None else []
    res.extend([r for r in right if r != "__CLEAR__"])
    return res

class AgentState(TypedDict):
    """Состояние графа LangGraph с поддержкой памяти."""
    messages: Annotated[list, add_messages]
    query: str
    user_company: Optional[str] # Выбранное пользователем предприятие
    intent: Optional[QueryIntent]
    extracted_context: Optional[str] # Извлеченный контекст из промежуточных шагов (например, отдел сотрудника)
    search_results: Annotated[List[str], clearable_add]
    answer: Optional[str]

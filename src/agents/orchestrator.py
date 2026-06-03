import logging
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.core.config import Config
from src.models.state import AgentState, QueryIntent
from src.agents.router import IntentRouter
from src.tools.contact_search import ContactSearchTool
from src.tools.rag_search import FilteredRAGTool
from src.tools.weather_tool import WeatherSearchTool
from src.llm.text import TextLLMService

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.router = IntentRouter(config)
        self.contact_tool = ContactSearchTool(db_path=str(config.data_path / "contacts.db"))
        self.rag_tool = FilteredRAGTool()
        self.weather_tool = WeatherSearchTool(config.default_city)
        self.llm_service = TextLLMService(config)

    async def initialize(self):
        """Инициализация инструментов."""
        await self.rag_tool.initialize()
        
        # 1. Инициализация памяти (Checkpoint)
        self.memory = MemorySaver()
        
        # Инициализация графа
        workflow = StateGraph(AgentState)
        
        # Добавление узлов
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("search_contacts", self.search_contacts)
        workflow.add_node("search_documents", self.search_documents)
        workflow.add_node("search_weather", self.search_weather)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Настройка ребер
        workflow.add_edge(START, "analyze_query")
        
        # Маршрутизация после анализа
        workflow.add_conditional_edges(
            "analyze_query",
            self.route_after_analysis,
            {
                "search_contacts": "search_contacts",
                "search_documents": "search_documents",
                "search_weather": "search_weather"
            }
        )
        
        # Умная маршрутизация после поиска контактов
        workflow.add_conditional_edges(
            "search_contacts",
            self.route_after_contacts,
            {
                "search_documents": "search_documents",
                "generate_answer": "generate_answer"
            }
        )
        
        # Все пути ведут к генерации ответа (кроме контактов, они решают сами)
        workflow.add_edge("search_documents", "generate_answer")
        workflow.add_edge("search_weather", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # 2. Компиляция с чекпоинтером
        self.app = workflow.compile(checkpointer=self.memory)

    async def _decontextualize_query(self, state: AgentState) -> str:
        """
        Превращает зависимый от контекста запрос в независимый.
        Пример: "А в каком отделе он работает?" -> "В каком отделе работает Зиннатуллин Ильнар Ильгизович?"
        """
        messages = state.get("messages", [])
        if not messages:
            return state["query"]

        # Если в истории только одно сообщение (текущее), деконтекстуализация не нужна
        if len(messages) <= 1:
            return state["query"]

        # Быстрая эвристика: если запрос не содержит контекстно-зависимых слов,
        # пропускаем дорогостоящий LLM-вызов и возвращаем запрос без изменений.
        CONTEXT_DEPENDENT_WORDS = {
            "он", "она", "они", "оно", "его", "её", "ее", "их",
            "ему", "ей", "им", "него", "неё", "нее", "них",
            "там", "туда", "оттуда", "тогда", "тот", "та", "те", "то",
            "этот", "эта", "эти", "это", "этого", "этой", "этих",
            "такой", "такая", "такие", "такое",
            "а в", "а у", "а на", "а он", "а она", "а что", "а где",
            "и он", "и она", "и они", "и там",
        }
        query_lower = state["query"].lower().strip()
        # Проверяем наличие хотя бы одного контекстного слова
        needs_decontextualization = any(
            f" {w} " in f" {query_lower} " or query_lower.startswith(f"{w} ") or query_lower == w
            for w in CONTEXT_DEPENDENT_WORDS
        )
        # Дополнительно: короткие запросы, начинающиеся с "а " или "и " — почти всегда контекстные
        if not needs_decontextualization:
            if query_lower.startswith(("а ", "и ", "ну а ")):
                needs_decontextualization = True

        if not needs_decontextualization:
            logger.info("--- DECONTEXTUALIZING QUERY: skipped (no context-dependent words) ---")
            return state["query"]

        logger.info("--- DECONTEXTUALIZING QUERY ---")
        
        history_text = "\n".join([f"{'User' if i%2==0 else 'Assistant'}: {m.content if hasattr(m, 'content') else m}" 
                                  for i, m in enumerate(messages[:-1])])
        
        prompt = f"""Ты — AI-редактор контекста. Твоя задача — переформулировать текущий запрос пользователя так, чтобы он был понятен без истории диалога.

ПРАВИЛА (СТРОГО):
1. Если запрос УЖЕ понятен (например, "кто директор в Технотроне?", "где столовая?"), верни его БЕЗ ИЗМЕНЕНИЙ.
2. Если запрос начинается с "а...", "и...", "а в...", "а у..." (например, "а в Технотроне?"), ты ОБЯЗАН дополнить его темой из истории (например, "кто главный в Технотроне?").
3. Если в новом вопросе упомянута ДРУГАЯ компания или человек в виде полного предложения (например, "Расскажи про ИТЗ"), не смешивай это с прошлой темой.
5. ПРИОРИТЕТ ПОСЛЕДНЕГО СООБЩЕНИЯ (КРИТИЧЕСКИ ВАЖНО): Если в вопросе есть местоимения (он, она, его, её, у неё), они ВСЕГДА относятся к тому человеку (ФИО) или той должности, о которой Ассистент писал в своем САМОМ ПОСЛЕДНЕМ ответе. 
   - Не перескакивай вглубь истории! Если в последнем ответе Ассистента обсуждалась Шириева, значит "она" — это Шириева. Если обсуждался Помощник, значит "она" — это Помощник.
   - Если ФИО не названо, заменяй местоимение на ДОЛЖНОСТЬ (например: "где находится Помощник Управляющего Технотрона").
6. Возвращай только текст вопроса.
7. ЗАПРЕЩЕНО добавлять отсебятину или отвечать на вопрос. Ты только переформулируешь его.

ИСТОРИЯ ДИАЛОГА:
{history_text}

НОВЫЙ ВОПРОС:
{state['query']}

ПЕРЕФОРМУЛИРОВАННЫЙ ВОПРОС:"""

        try:
            resolved = await self.llm_service.generate(prompt, temperature=0.0)
            return resolved.strip()
        except Exception as e:
            logger.error(f"Decontextualization failed: {e}")
            return state["query"]

    async def analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Узел анализа интента с учетом контекста."""
        logger.info(f"--- ANALYZE QUERY (Stateless): {state['query']} ---")
        
        # 1. Резолвим контекст (кто такой "он", "там" и т.д.)
        resolved_query = await self._decontextualize_query(state)
        if resolved_query != state["query"]:
            logger.info(f"--- RESOLVED QUERY: {resolved_query} ---")
        
        # 2. Классифицируем уже "чистый" запрос
        intent = await self.router.classify_query(resolved_query)
        
        # 3. Если компания не определена в запросе, берем из настроек пользователя
        if not intent.target_company and state.get("user_company"):
            mapping = {
                "it": "АЙТИ",
                "itz": "ИТЗ",
                "technotron": "ПТФК Технотрон",
                "metiz": "Метиз",
                "kmk": "КМК",
                "ntz": "НТЗ",
                "kzmk": "КЗМК",
                "zteo": "ЗТЭО",
                "td": "ТД",
                "sks": "СКС",
                "port": "Порт"
            }
            mapped_company = mapping.get(state["user_company"].lower())
            if mapped_company:
                intent.target_company = mapped_company
                logger.info(f"Using user-selected company fallback: {mapped_company}")
        
        # Сохраняем в историю само сообщение
        return {
            "intent": intent, 
            "query": resolved_query,
            "messages": [("user", state["query"])],
            "search_results": ["__CLEAR__"], # Очищаем корзину прошлого поиска через умный редьюсер
            "extracted_context": None # Очищаем промежуточный контекст
        }

    async def search_contacts(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- SEARCH CONTACTS (SQL) ---")
        person = state["intent"].target_person or ""
        company = state["intent"].target_company
        
        # Расширяем поиск для руководителей (но НЕ если ищут помощника или секретаря)
        boss_keywords = ["управляющий", "директор", "руководитель", "начальник", "главный", "босс"]
        exclude_keywords = ["помощник", "секретарь", "приемная", "референт"]
        search_query = person
        
        person_lower = person.lower()
        if any(k in person_lower for k in boss_keywords) and not any(e in person_lower for e in exclude_keywords):
            search_query = f"{person} директор управляющий руководитель помощник секретарь приемная"
            logger.info(f"Expanded search query for boss: {search_query}")
            
        results = await self.contact_tool.search(
            target_person=search_query,
            target_company=company
        )
        
        # Извлекаем Отдел и Компанию из текстового результата (берем первую запись)
        import re
        extracted_context = None
        dept_match = re.search(r"Отдел:\s*([^\n]+)", results)
        comp_match = re.search(r"Компания:\s*([^|\n]+)", results)
        
        if dept_match and comp_match:
            dept = dept_match.group(1).strip()
            comp = comp_match.group(1).strip()
            # Добавляем синонимы для отдела кадров и бухгалтерии, так как в расписаниях они могут называться иначе
            if "служба финансового директора" in dept.lower():
                dept = f"{dept} Бухгалтерия"
            extracted_context = f"{dept} {comp}"
        elif dept_match:
            dept = dept_match.group(1).strip()
            if "служба финансового директора" in dept.lower():
                dept = f"{dept} Бухгалтерия"
            extracted_context = dept
            
        if extracted_context:
            logger.info(f"Extracted context for multi-hop: {extracted_context}")
            
        return {
            "search_results": [f"КОНТАКТЫ:\n{results}"],
            "extracted_context": extracted_context
        }

    def route_after_contacts(self, state: AgentState) -> List[str]:
        """Решает, нужно ли идти в базу знаний после контактов."""
        if state.get("intent") and state["intent"].requires_rag:
            logger.info("--- MULTI-HOP: ROUTING TO RAG FOR EXTRA INFO (requires_rag is True) ---")
            return ["search_documents"]
            
        return ["generate_answer"]

    async def search_documents(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"--- SEARCH DOCUMENTS (RAG) ---")
        query_to_search = state["query"]
        
        # Обогащаем запрос, если пришли из контактов
        if state.get("extracted_context"):
            query_to_search = f"{query_to_search} {state['extracted_context']}"
            logger.info(f"RAG search enriched with extracted context: {query_to_search}")
            
        results = await self.rag_tool.search(query_to_search, state["intent"])
        
        # Передаем LLM подсказку о синонимах и отделе
        system_note = ""
        if state.get("extracted_context"):
            system_note = f"\n[СИСТЕМНОЕ СООБЩЕНИЕ: Обрати внимание, что отдел сотрудника ({state['extracted_context']}) может называться в документах иначе (например, 'Бухгалтерия', 'ИТР' и т.д.). Ищи соответствующие строки.]\n"
            
        return {"search_results": [f"{system_note}ДОКУМЕНТЫ ИЗ БАЗЫ ЗНАНИЙ:\n{results}"]}


    async def search_weather(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"--- SEARCH WEATHER ---")
        # Передаем извлеченный город из интента
        results = await self.weather_tool.search(state["intent"].target_location)
        return {"search_results": [f"ТЕКУЩАЯ ПОГОДА:\n{results}"]}

    async def generate_answer(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- GENERATE FINAL ANSWER ---")
        
        # Склеиваем все результаты поиска в один текстовый блок
        context_block = "\n\n===\n\n".join(state.get("search_results", []))
        
        prompt = f"""Ты — интеллектуальный корпоративный ассистент ГК «ТЭМПО».
Твоя задача: ответить на вопрос пользователя, опираясь ТОЛЬКО на предоставленный контекст.

КОНТЕКСТ ДЛЯ ОТВЕТА:
{context_block}

ВОПРОС ПОЛЬЗОВАТЕЛЯ (с учетом контекста):
{state['query']}

ПРАВИЛА:
1. Твой ответ должен быть САМОДОСТАТОЧНЫМ. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать "подробнее в документе", "см. раздел", "информация в базе знаний", упоминать названия разделов или любые названия документов из заголовков.
2. ТАБЛИЦЫ И ГРАФИКИ: Если вопрос касается РАСПИСАНИЙ, ВРЕМЕНИ ИЛИ ЧИСЕЛ, и в контексте найдена ТАБЛИЦА — ты ОБЯЗАН вывести её ПОДРОБНО (или нужную строку из неё). НО если пользователь спрашивает про МЕСТОПОЛОЖЕНИЕ кабинета или отдела (где находится, как пройти), КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать адреса из "расписаний столовой" или "графиков обедов". Ищи фактическое место работы (АБК, этаж, крыло), а не место, где люди едят!
3. ЗАПРЕЩЕНО ссылаться на любые внешние или внутренние разделы/базы данных. Твой ответ — это истина, которую ты сообщаешь напрямую.
4. КОНТАКТЫ: Если пользователь спрашивает контакты руководителя или конкретного лица, а их нет напрямую - ты ОБЯЗАН найти контакты "Помощника", "Секретаря", "Отдела кадров" или "Приемной" в этом же подразделении и предложить их как способ связи. ВНИМАНИЕ: Если пользователь УЖЕ спрашивает контакты Помощника или Секретаря, и их телефона нет в базе — просто честно скажи, что телефон не указан, НЕ НУЖНО предлагать связаться с ними же по кругу.
5. Если пользователь спрашивает "как устроиться", "что делать", "какие условия" - ты ОБЯЗАН выдать пошаговый план, список документов, контакты отдела кадров и конкретные требования из контекста.
6. Вытаскивай ВСЕ детали (условия, списки документов, шаги, сроки) из предоставленного контекста и пиши их прямо в чат.
7. Если в контексте есть ответ — выдай его максимально ПОЛНО. 
8. Если информации нет — честно скажи об этом.
9. Оформляй контакты красиво.
10. ЗАПРЕТ ГАЛЛЮЦИНАЦИЙ: Никогда не пиши фразы "по ссылке", "см. ссылку", если в твоем КОНТЕКСТЕ нет реального URL (http://...). Если информация есть только в виде текста — пиши её текстом. 
11. НИКОГДА не выводи внутренние ссылки на файлы вида [Текст](slug) или [[slug]].
12. СТРОЖАЙШИЙ ЗАПРЕТ: Тебе ЗАПРЕЩЕНО писать "информация доступна по ссылке" или выводить пути к файлам (например, technotron/graphics/...). Если пользователь спросил график — ты ОБЯЗАН найти время в предоставленном тексте и написать его.
13. УМНАЯ РАБОТА С КОНТЕКСТОМ: Отвечай СТРОГО по сути вопроса. Если ищут расписание — дай строку из таблицы. Если ищут карту или кабинет — выдай ссылку на карту, проигнорировав любые графики, случайно попавшие в контекст.
14. ПРАВИЛО ФОРМАТИРОВАНИЯ (ТОЛЬКО HTML): Используй <b>текст</b> для жирного шрифта, <i>текст</i> для курсива. ЗАПРЕЩЕНО использовать Markdown (** или *).
15. ТОЧНОСТЬ ФАМИЛИЙ И КОНТАКТОВ: Если в результатах поиска контактов выдано несколько людей, КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ игнорировать первый контакт (под номером 1) и выбирать людей с 2 или 3 места только из-за того, что их фамилия кажется тебе "правильнее". Алгоритм уже отсортировал их: контакт №1 — самый релевантный вашему запросу. Выдавай информацию именно о нем.
16. ТЕКСТ ВМЕСТО НОМЕРА: Если в предоставленном КОНТЕКСТЕ в поле "Тел:" вместо цифр указан текст (например, "Помощник Управляющего..."), ты ОБЯЗАН вывести этот текст пользователю как инструкцию по связи. ЗАПРЕЩЕНО скрывать этот текст или пытаться искать по нему новый контакт. Просто напиши: "Телефон: Помощник Управляющего...".

ПРАВИЛО ФОРМАТИРОВАНИЯ:
Если в предоставленном КОНТЕКСТЕ есть ВНЕШНИЕ ссылки (URL на сайты, карты), ты ОБЯЗАН сохранить их в своем ответе в формате Markdown: [Текст ссылки](URL). 
ВАЖНОЕ ИСКЛЮЧЕНИЕ ДЛЯ ИЗОБРАЖЕНИЙ: Если в тексте есть картинка вида ![Текст](ссылка.png) - ты ОБЯЗАН сохранить её в ответе точно в таком же виде! 
"""
        try:
            answer = await self.llm_service.generate(prompt)
            return {
                "answer": answer,
                "messages": [("assistant", answer)] # Добавляем ответ ассистента в историю
            }
        except Exception as e:
            logger.exception(f"Synthesis error: {e}")
            return {"answer": "Извините, произошла ошибка."}

    def route_after_analysis(self, state: AgentState) -> List[str]:
        intent = state["intent"].intent
        if intent == "contact_search":
            return ["search_contacts"]
        elif intent == "emergency":
            return ["search_contacts", "search_documents"]
        elif intent == "weather":
            return ["search_weather"]
        else:
            return ["search_documents"]

    async def process_query(self, query: str, thread_id: str = "default_user", user_company: Optional[str] = None) -> Dict[str, Any]:
        """Точка входа с поддержкой thread_id для памяти. Возвращает полное состояние."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Начальное состояние
        initial_input = {
            "query": query, 
            "search_results": [],
            "user_company": user_company
        }
        
        # Запуск графа
        result = await self.app.ainvoke(initial_input, config=config)
        return result

    async def clear_memory(self, thread_id: str):
        """Принудительно очищает оперативную память (LangGraph) для конкретного пользователя."""
        try:
            # Всегда используем прямую очистку словарей, так как штатный delete_thread 
            # в некоторых версиях падает с ошибкой 'unhashable type: dict'
            if hasattr(self.memory, 'storage') and isinstance(self.memory.storage, dict):
                keys_to_delete = []
                for k in list(self.memory.storage.keys()):
                    try:
                        if isinstance(k, tuple) and len(k) > 0 and str(k[0]) == str(thread_id):
                            keys_to_delete.append(k)
                        elif isinstance(k, str) and str(thread_id) in k:
                            keys_to_delete.append(k)
                    except Exception:
                        pass
                for k in keys_to_delete:
                    del self.memory.storage[k]
                    
            if hasattr(self.memory, 'writes') and isinstance(self.memory.writes, dict):
                keys_to_delete = []
                for k in list(self.memory.writes.keys()):
                    try:
                        if isinstance(k, tuple) and len(k) > 0 and str(k[0]) == str(thread_id):
                            keys_to_delete.append(k)
                        elif isinstance(k, str) and str(thread_id) in k:
                            keys_to_delete.append(k)
                    except Exception:
                        pass
                for k in keys_to_delete:
                    del self.memory.writes[k]
                    
            logger.info(f"LangGraph memory safely cleared for thread_id: {thread_id}")
        except Exception as e:
            logger.error(f"Error safely clearing LangGraph memory for thread_id {thread_id}: {e}")

if __name__ == "__main__":
    import asyncio
    async def test():
        cfg = Config.from_env()
        orch = AgentOrchestrator(cfg)
        # Тест на контекст
        print(await orch.process_query("Дай номер Зиннатуллина из АЙТИ", "user_1"))
        print(await orch.process_query("А в каком отделе он работает?", "user_1"))

    asyncio.run(test())

import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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
        self.database_url = config.database_url
        self.router = IntentRouter(config)
        self.contact_tool = ContactSearchTool(config=config)
        self.rag_tool = FilteredRAGTool()
        self.weather_tool = WeatherSearchTool(config.default_city)
        self.llm_service = TextLLMService(config)

    async def initialize(self):
        """Инициализация инструментов."""
        await self.rag_tool.initialize()
        
        # 1. Инициализация памяти (PostgreSQL Checkpointer)
        self._memory_ctx = AsyncPostgresSaver.from_conn_string(self.database_url)
        self.memory = await self._memory_ctx.__aenter__()
        await self.memory.setup()   # создаёт таблицы checkpoints, writes, migrations
        
        # Инициализация графа
        workflow = StateGraph(AgentState)
        
        # Добавление узлов
        workflow.add_node("summarize_if_needed", self.summarize_if_needed)
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("search_contacts", self.search_contacts)
        workflow.add_node("search_documents", self.search_documents)
        workflow.add_node("search_weather", self.search_weather)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Настройка ребер
        workflow.add_edge(START, "summarize_if_needed")
        workflow.add_edge("summarize_if_needed", "analyze_query")
        
        # Маршрутизация после анализа
        workflow.add_conditional_edges(
            "analyze_query",
            self.route_after_analysis,
            {
                "search_contacts": "search_contacts",
                "search_documents": "search_documents",
                "search_weather": "search_weather",
                "generate_answer": "generate_answer",  # для самопредставления и pre-computed ответов
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

    async def summarize_if_needed(self, state: AgentState) -> Dict[str, Any]:
        """
        Узел LangGraph, который сжимает историю диалога, если она превышает порог.
        """
        messages = state.get("messages", [])
        
        # Извлекаем лимиты из конфига
        threshold = self.config.memory.messages_summarize_threshold
        keep_recent = self.config.memory.messages_keep_recent
        max_chars = self.config.memory.max_chars_per_history_message
        
        if len(messages) <= threshold:
            return {}

        logger.info(f"--- ROLLING SUMMARY: history size {len(messages)} exceeds threshold {threshold} ---")
        
        # Оставляем keep_recent последних сообщений в графе.
        # Предыдущие сообщения суммаризируем.
        summarize_count = len(messages) - keep_recent
        to_summarize = messages[:summarize_count]
        
        # Формируем текст для суммаризации
        history_parts = []
        for m in to_summarize:
            msg_type = getattr(m, 'type', '')
            role = "Пользователь" if msg_type in ('human', 'user') else "Ассистент"
            content = m.content if hasattr(m, 'content') else str(m)
            # Обрезаем очень длинные сообщения
            if len(content) > max_chars:
                content = content[:max_chars] + "... [сообщение обрезано]"
            history_parts.append(f"{role}: {content}")
            
        history_text = "\n".join(history_parts)
        
        existing_summary = state.get("conversation_summary", "")
        summary_prompt = f"""Ниже представлена история диалога и, возможно, предыдущее резюме.
Создай обновленное краткое резюме диалога на русском языке. Оно должно содержать ключевые факты, упомянутые имена, компании, контакты, интересы пользователя и обсуждаемые темы.
Сделай резюме максимально информативным и компактным.

Предыдущее резюме (если есть):
{existing_summary}

Новые сообщения для добавления в резюме:
{history_text}

Обновленное краткое резюме (на русском):"""

        try:
            new_summary = await self.llm_service.generate(summary_prompt, temperature=0.3)
            new_summary = new_summary.strip()
            logger.info(f"--- ROLLING SUMMARY COMPLETED. New summary length: {len(new_summary)} ---")
            
            # Удаляем старые сообщения
            delete_actions = []
            for m in to_summarize:
                msg_id = getattr(m, 'id', None)
                if msg_id:
                    delete_actions.append(RemoveMessage(id=msg_id))
            
            # Добавим системное сообщение с новым резюме
            summary_msg = SystemMessage(content=f"КОНТЕКСТ ДИАЛОГА (РЕЗЮМЕ):\n{new_summary}")
            
            return {
                "conversation_summary": new_summary,
                "messages": delete_actions + [summary_msg]
            }
        except Exception as e:
            logger.error(f"Rolling summary failed: {e}")
            return {}

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

        # Ранний выход: если запрос содержит явно самодостаточные ключевые слова
        # (новая тема, не требующая привязки к старому контексту), пропускаем LLM
        TOPIC_SHIFT_SIGNALS = {
            "погода", "прогноз", "температура", "дождь", "снег",
            "травму", "травма", "скорую", "пожар", "медпункт",
            "как устроиться", "как попасть на работу", "вакансия",
        }
        if needs_decontextualization:
            if any(signal in query_lower for signal in TOPIC_SHIFT_SIGNALS):
                logger.info("--- DECONTEXTUALIZING QUERY: skipped (topic-shift signal detected) ---")
                return state["query"]

        if not needs_decontextualization:
            logger.info("--- DECONTEXTUALIZING QUERY: skipped (no context-dependent words) ---")
            return state["query"]

        logger.info("--- DECONTEXTUALIZING QUERY ---")
        
        summary = state.get("conversation_summary")
        summary_prefix = f"Резюме предыдущего разговора: {summary}\n" if summary else ""

        history_parts = []
        for m in messages[-10:]:
            # LangChain message types: 'human'/'user' for user, 'ai'/'assistant' for bot
            msg_type = getattr(m, 'type', '')
            if msg_type == 'system':
                continue
            role = "User" if msg_type in ('human', 'user') else "Assistant"
            content = m.content if hasattr(m, 'content') else str(m)
            history_parts.append(f"{role}: {content}")
        history_text = summary_prefix + "\n".join(history_parts)
        
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
            resolved = resolved.strip()
            # Guard: если LLM вернул пустую строку, используем оригинальный запрос
            return resolved if resolved else state["query"]
        except Exception as e:
            logger.error(f"Decontextualization failed: {e}")
            return state["query"]

    def _extract_self_introduced_name(self, query: str) -> Optional[str]:
        """
        Детектирует самопредставление пользователя.
        Возвращает имя если найдено, иначе None.
        """
        import re
        q = query.strip()
        # Patterns: 'Я [Name]', 'Меня зовут [Name]', 'Моё имя [Name]'
        patterns = [
            r'^\u044f\s+([\u0400-\u04ff][\u0400-\u04ff-]{1,30})$',          # Я Имя
            r'^\u043cеня\s+зовут\s+([\u0400-\u04ff][\u0400-\u04ff-]{1,30})',  # Меня зовут Имя
            r'^\u043cоё\s+имя\s+([\u0400-\u04ff][\u0400-\u04ff-]{1,30})',    # Моё имя Имя
            r'^\u044f\s+([\u0400-\u04ff][\u0400-\u04ff-]{1,30}\s+[\u0400-\u04ff][\u0400-\u04ff-]{1,30})$',  # Я Имя Фамилия
        ]
        for pattern in patterns:
            m = re.match(pattern, q.lower())
            if m:
                # Return original-case slice from query
                name_lower = m.group(1)
                idx = q.lower().find(name_lower)
                return q[idx: idx + len(name_lower)].strip() if idx != -1 else m.group(1).capitalize()
        return None

    async def analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Узел анализа интента с учетом контекста."""
        logger.info(f"--- ANALYZE QUERY (Stateless): {state['query']} ---")

        # 0. Перехват самопредставления — до деконтекстуализации и роутинга
        introduced_name = self._extract_self_introduced_name(state["query"])
        if introduced_name:
            logger.info(f"--- SELF-INTRODUCTION DETECTED: '{introduced_name}' ---")
            greeting = f"Здравствуйте, {introduced_name}! Чем могу помочь?"
            return {
                "intent": QueryIntent(intent="general_info"),
                "query": state["query"],
                "user_name": introduced_name,
                "messages": [("user", state["query"]), ("assistant", greeting)],
                "search_results": ["__CLEAR__"],
                "extracted_context": None,
                "answer": greeting,
            }

        
        # 1. Резолвим контекст (кто такой "он", "там" и т.д.)
        resolved_query = await self._decontextualize_query(state)
        if resolved_query != state["query"]:
            logger.info(f"--- RESOLVED QUERY: {resolved_query} ---")
        
        # 2. Классифицируем уже "чистый" запрос
        intent = await self.router.classify_query(resolved_query)
        
        # 3. Мы больше не применяем fallback компанию здесь. Контакты ищутся глобально (Shared).
        # Fallback компания для документов (базы знаний) будет применена непосредственно в search_documents.

        # Сохраняем в историю само сообщение; сбрасываем answer чтобы не short-circuit-нуть следующий запрос
        return {
            "intent": intent, 
            "query": resolved_query,
            "messages": [("user", state["query"])],
            "search_results": ["__CLEAR__"], # Очищаем корзину прошлого поиска через умный редьюсер
            "extracted_context": None,       # Очищаем промежуточный контекст
            "answer": None,                  # Сбрасываем pre-computed ответ
        }

    async def search_contacts(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- SEARCH CONTACTS (SQL) ---")
        intent = state["intent"]
        person = intent.target_person or ""
        company = intent.target_company
        phone = getattr(intent, "exact_phone", None)
        
        # Расширяем поиск для руководителей (но НЕ если ищут помощника или секретаря)
        boss_keywords = ["управляющий", "директор", "руководитель", "начальник", "главный", "босс"]
        exclude_keywords = ["помощник", "секретарь", "приемная", "референт"]
        search_query = person
        
        if person:
            person_lower = person.lower()
            if any(k in person_lower for k in boss_keywords) and not any(e in person_lower for e in exclude_keywords):
                search_query = f"{person} директор управляющий руководитель помощник секретарь приемная"
                logger.info(f"Expanded search query for boss: {search_query}")
            
        kwargs = {
            "semantic_query": search_query,
            "company_filter": company,
            "exact_phone": phone
        }
        
        results = await self.contact_tool.search(**kwargs)
        
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
        intent = state["intent"]
        
        # Применяем fallback компанию только для поиска документов
        if not intent.target_company and state.get("user_company"):
            mapping = {
                "it": "АЙТИ", "itz": "ИТЗ", "technotron": "ПТФК Технотрон",
                "metiz": "Метиз", "kmk": "КМК", "ntz": "НТЗ",
                "kzmk": "КЗМК", "zteo": "ЗТЭО", "td": "ТД",
                "sks": "СКС", "port": "Порт"
            }
            mapped_company = mapping.get(state["user_company"].lower())
            if mapped_company:
                intent.target_company = mapped_company
                logger.info(f"Using user-selected company fallback for documents: {mapped_company}")
        
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

        # Если ответ уже сформирован на предыдущем шаге (например, самопредставление), возвращаем его
        if state.get("answer"):
            logger.info("--- GENERATE FINAL ANSWER: using pre-computed answer ---")
            return {"answer": state["answer"]}

        # Склеиваем все результаты поиска в один текстовый блок
        context_block = "\n\n===\n\n".join(state.get("search_results", []))

        # Формируем блок истории диалога для LLM (10 последних сообщений)
        history_block = ""
        messages = state.get("messages", [])
        if messages:
            history_parts = []
            for m in messages[-10:]:
                msg_type = getattr(m, 'type', '')
                if msg_type == 'system':
                    continue
                role = "Пользователь" if msg_type in ('human', 'user') else "Ассистент"
                content = m.content if hasattr(m, 'content') else str(m)
                history_parts.append(f"{role}: {content}")
            if history_parts:
                history_block = "\n\nИСТОРИЯ ДИАЛОГА (последние сообщения):\n" + "\n".join(history_parts)

        # Имя пользователя, если он представился
        user_name = state.get("user_name")
        user_name_block = f"\nИМЯ ПОЛЬЗОВАТЕЛЯ: {user_name}" if user_name else ""
        
        summary_block = ""
        summary = state.get("conversation_summary")
        if summary:
            summary_block = f"\nКРАТКОЕ РЕЗЮМЕ ПРЕДЫДУЩЕЙ БЕСЕДЫ:\n{summary}\n"
        
        prompt = f"""Ты — интеллектуальный корпоративный ассистент ГК «ТЭМПО».
Твоя задача: ответить на вопрос пользователя, опираясь на предоставленный контекст, историю диалога и резюме беседы.
{user_name_block}
{summary_block}
КОНТЕКСТ ДЛЯ ОТВЕТА:
{context_block}
{history_block}

ВОПРОС ПОЛЬЗОВАТЕЛЯ (с учетом контекста):
{state['query']}

ПРАВИЛА:
RULE 1: Always prioritize checking the {{chat_history}} first. If the user's question can be answered strictly using the Chat History (e.g., their name, previous topics), answer it immediately WITHOUT saying 'information not found'. Если пользователь спрашивает 'как меня зовут' и его имя уже названо в ИСТОРИИ ДИАЛОГА — отвечай из истории, НЕ вызывай search tools.
RULE 2: Use retrieved database context ONLY for external facts.

1. Твой ответ должен быть САМОДОСТАТОЧНЫМ. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать "подробнее в документе", "см. раздел", "информация в базе знаний", упоминать названия разделов или любые названия документов из заголовков.
2. ТАБЛИЦЫ И ГРАФИКИ: Если вопрос касается РАСПИСАНИЙ, ВРЕМЕНИ ИЛИ ЧИСЕЛ, и в контексте найдена ТАБЛИЦА — ты ОБЯЗАН вывести её ПОДРОБНО (или нужную строку из неё). НО если пользователь спрашивает про МЕСТОПОЛОЖЕНИЕ кабинета или отдела (где находится, как пройти), КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать адреса из "расписаний столовой" или "графиков обедов". Ищи фактическое место работы (АБК, этаж, крыло), а не место, где люди едят!
3. ЗАПРЕЩЕНО ссылаться на любые внешние или внутренние разделы/базы данных. Твой ответ — это истина, которую ты сообщаешь напрямую.
4. КОНТАКТЫ: Если пользователь спрашивает контакты руководителя или конкретного лица, а их нет напрямую - ты ОБЯЗАН найти контакты "Помощника", "Секретаря", "Отдела кадров" или "Приемной" в этом же подразделении и предложить их как способ связи. ВНИМАНИЕ: Если пользователь УЖЕ спрашивает контакты Помощника или Секретаря, и их телефона нет в базе — просто честно скажи, что телефон не указан, НЕ НУЖНО предлагать связаться с ними же по кругу.
5. Если пользователь спрашивает "как устроиться", "что делать", "какие условия" - ты ОБЯЗАН выдать пошаговый план, список документов, контакты отдела кадров и конкретные требования из контекста.
6. Вытаскивай ВСЕ детали (условия, списки документов, шаги, сроки) из предоставленного контекста и пиши их прямо в чат.
7. Если в контексте есть ответ — выдай его максимально ПОЛНО. 
8. Если информации нет — честно скажи об этом.
9. СТРОГОЕ ОФОРМЛЕНИЕ КОНТАКТОВ: Если пользователь запрашивает контакт, ВСЕГДА выводи его в виде четкого списка, где каждый пункт с новой строки. Не используй разговорный стиль. Формат должен быть строго таким:
<b>ФИО:</b> [ФИО]
<b>Должность:</b> [Должность]
<b>Отдел:</b> [Отдел]
<b>Компания:</b> [Компания]
<b>Телефон:</b> [Телефон]
<b>Email:</b> [Email]
10. ЗАПРЕТ ГАЛЛЮЦИНАЦИЙ: Никогда не пиши фразы "по ссылке", "см. ссылку", если в твоем КОНТЕКСТЕ нет реального URL (http://...). Если информация есть только в виде текста — пиши её текстом. 
11. НИКОГДА не выводи внутренние ссылки на файлы вида [Текст](slug) или [[slug]].
12. СТРОЖАЙШИЙ ЗАПРЕТ: Тебе ЗАПРЕЩЕНО писать "информация доступна по ссылке" или выводить пути к файлам (например, technotron/graphics/...). Если пользователь спросил график — ты ОБЯЗАН найти время в предоставленном тексте и написать его.
13. УМНАЯ РАБОТА С КОНТЕКСТОМ: Отвечай СТРОГО по сути вопроса. Если ищут расписание — дай строку из таблицы. Если ищут карту или кабинет — выдай ссылку на карту, проигнорировав любые графики, случайно попавшие в контекст.
14. ПРАВИЛО ФОРМАТИРОВАНИЯ (ТОЛЬКО HTML): Используй <b>текст</b> для жирного шрифта, <i>текст</i> для курсива. ЗАПРЕЩЕНО использовать Markdown (** или *).
15. ТОЧНОСТЬ ФАМИЛИЙ И КОНТАКТОВ: Если в результатах поиска контактов выдано несколько людей, КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ игнорировать первый контакт (под номером 1) и выбирать людей с 2 или 3 места только из-за того, что их фамилия кажется тебе "правильнее". Алгоритм уже отсортировал их: контакт №1 — самый релевантный вашему запросу. Выдавай информацию именно о нем.
16. ТЕКСТ ВМЕСТО НОМЕРА: Если в предоставленном КОНТЕКСТЕ в поле "Тел:" вместо цифр указан текст (например, "Помощник Управляющего..."), ты ОБЯЗАН вывести этот текст пользователю как инструкцию по связи. ЗАПРЕЩЕНО скрывать этот текст или пытаться искать по нему новый контакт. Просто напиши: "Телефон: Помощник Управляющего...".
17. ЛИЧНЫЕ ВОПРОСЫ: Если пользователь спрашивает о себе ("как меня зовут", "кто я", "что я спрашивал"), ищи ответ ИСКЛЮЧИТЕЛЬНО в ИСТОРИИ ДИАЛОГА выше — НЕ в базе знаний. Если пользователь назвал своё имя в диалоге, используй его.

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
            err_msg = "Извините, произошла ошибка."
            return {
                "answer": err_msg,
                "messages": [("assistant", err_msg)]
            }

    def route_after_analysis(self, state: AgentState) -> List[str]:
        # Если ответ уже сформирован (например, самопредставление пользователя), пропускаем поиск
        if state.get("answer"):
            return ["generate_answer"]
            
        # Если запрос разговорный (например, "как меня зовут?"), касается истории чата
        # и не запрашивает корпоративные данные, направляем напрямую к генерации ответа
        query_lower = state["query"].lower()
        conversational_phrases = [
            "как меня зовут", "кто я", "мое имя", "моё имя", 
            "что я спрашивал", "о чем мы", "о чём мы", 
            "ты меня помнишь", "помнишь меня", "скажи моё имя", "скажи мое имя",
            "как тебя зовут", "что ты умеешь", "спасибо", "привет", "здравствуй",
            "как дела", "помоги мне", "что можешь", "ты кто"
        ]
        is_conversational = any(phrase in query_lower for phrase in conversational_phrases)
        has_corporate_keywords = any(word in query_lower for word in ["телефон", "номер", "контакт", "почта", "завод", "тэмпо", "кабинет", "схема", "документ", "инструкция"])
        
        if is_conversational and not has_corporate_keywords:
            logger.info("Conversational query detected. Bypassing tool execution.")
            return ["generate_answer"]

        intent = state["intent"].intent
        if intent == "personal":
            return ["generate_answer"]
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

    async def inject_voice_turn(self, thread_id: str, user_text: str, assistant_text: str):
        """
        Записывает голосовое взаимодействие в LangGraph MemorySaver,
        чтобы текстовый канал мог видеть его в истории диалога.
        
        Это решает проблему: голос → текст ассистент не помнит, что было сказано голосом.
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            voice_messages = [
                HumanMessage(content=f"[Голос] {user_text}"),
                AIMessage(content=f"[Голос] {assistant_text}"),
            ]
            await self.app.aupdate_state(
                config,
                {"messages": voice_messages},
            )
            logger.info(f"Voice turn injected into LangGraph memory for thread_id={thread_id}: user={user_text[:60]!r}")
        except Exception as e:
            # Некритичная ошибка — голос уже отправлен, просто не запишем в MemorySaver
            logger.warning(f"Failed to inject voice turn into LangGraph memory: {e}")

    async def clear_memory(self, thread_id: str):
        """Принудительно очищает историю диалога в PostgreSQL для конкретного пользователя."""
        try:
            if hasattr(self, 'memory') and hasattr(self.memory, 'adelete_thread'):
                await self.memory.adelete_thread(thread_id)
                logger.info(f"LangGraph PG memory safely cleared via adelete_thread for thread_id: {thread_id}")
            else:
                logger.warning("memory.adelete_thread is not available")
        except Exception as e:
            logger.error(f"Error clearing LangGraph PG memory for thread_id {thread_id}: {e}")

    async def close(self):
        """Закрыть соединение с PostgreSQL (при остановке сервера)."""
        try:
            if hasattr(self, '_memory_ctx'):
                await self._memory_ctx.__aexit__(None, None, None)
                logger.info("AsyncPostgresSaver connection closed successfully.")
        except Exception as e:
            logger.warning(f"Error closing AsyncPostgresSaver connection: {e}")

if __name__ == "__main__":
    import asyncio
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def test():
        cfg = Config.from_env()
        orch = AgentOrchestrator(cfg)
        await orch.initialize()
        try:
            # Тест на контекст
            print(await orch.process_query("Дай номер Зиннатуллина из АЙТИ", "user_1"))
            print(await orch.process_query("А в каком отделе он работает?", "user_1"))
        finally:
            await orch.close()

    asyncio.run(test())

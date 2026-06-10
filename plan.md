# Переход на гибридный поиск Qdrant для контактов

## Описание

Текущая система поиска контактов использует PostgreSQL (`pg_trgm`) в Docker-контейнере, что создаёт риск потери данных при `docker compose down -v`. Цель — перевести поиск контактов на Qdrant (уже используемый в проекте для RAG-документов), сохранив `data/contacts.db` (SQLite) как единственный источник правды (bind mount на хосте).

## Архитектурные принципы

```
data/contacts.db      (SQLite, bind mount хоста)  →  ИСТОЧНИК ПРАВДЫ
qdrant_storage_v2/    (bind mount хоста)           →  ПОИСКОВЫЙ ИНДЕКС
postgres_data         (Docker volume)              →  ЛОГИ, ПОЛЬЗОВАТЕЛИ (не контакты)
```

> [!IMPORTANT]
> Qdrant в проекте работает **in-process** (локально, через `QdrantClient(path=...)`) — отдельного Docker-сервиса Qdrant **нет**. `qdrant_storage_v2/` — это bind mount на хост. Всё это уже реализовано в `ClientManager.get_qdrant_client()`.

> [!WARNING]
> `HybridSearchService.initialize()` сейчас загружает имена из **PostgreSQL** для `FuzzyNameMatcher`. Это нужно исправить — переключить на чтение из SQLite (или из Qdrant payload после миграции).

---

## Исправления плана относительно исходного `plan.md`

1. **Этап 1 (скрипт миграции):** Нужно **не создавать новый `QdrantClient`**, а использовать существующий `ClientManager.get_qdrant_client()` с правильным `path` из конфига (`config.storage_path`). Иначе Qdrant откроет новый процесс и заблокирует storage.
2. **Этап 1 (имена коллекции):** Имя `"contacts"` не должно совпадать с именем основной RAG-коллекции (`config.collection_name`). Нужно завести отдельное имя, например `"contacts_v1"`.
3. **Этап 3 (FuzzyNameMatcher):** `HybridSearchService.initialize()` читает имена из Postgres — это нужно исправить одновременно с миграцией на Qdrant, переключив на SQLite.
4. **Этап 3 (не переиспользовать `HybridSearchService`):** Он жёстко привязан к `config.collection_name` и логике RAG-документов (поля `text`, `source`, `chunk_index` и т.д.). Для контактов нужен **отдельный легковесный класс** `ContactHybridSearch`.

---

## Proposed Changes

### Этап 0 — Пометить устаревший скрипт

#### [MODIFY] [migrate_contacts_to_pg.py](file:///e:/Old/bots/Worker/iTEMPO/iTEMPO_test/scripts/migrate_contacts_to_pg.py)
Добавить в начало файла:
```python
# DEPRECATED: контакты больше не хранятся в Postgres.
# Используйте: uv run python -m scripts.migrate_contacts_to_qdrant
```

---

### Этап 1 — Скрипт миграции SQLite → Qdrant

#### [NEW] [migrate_contacts_to_qdrant.py](file:///e:/Old/bots/Worker/iTEMPO/iTEMPO_test/scripts/migrate_contacts_to_qdrant.py)

**Логика:**
1. Читать из `data/contacts.db` (SQLite) — единственный источник правды.
2. Получать `QdrantClient` через `ClientManager.get_qdrant_client()` с правильным `path=config.storage_path` — **не создавать новый клиент**.
3. Создать/пересоздать коллекцию `"contacts_v1"` с конфигурацией:
   - Dense: `VectorParams(size=config.vector_size, distance=Distance.COSINE)`
   - Sparse: `SparseVectorParams()` с именем `"sparse"`
   - Создать payload-индекс типа Text (`TextIndexParams(type="text", tokenizer=TokenizerType.WORD)`) для поля `phone`, чтобы поддерживать фильтрацию по частичным номерам.
4. Строка для векторизации каждого контакта: `f"{full_name} {position} {department} {company}"`
5. Dense-вектор: через `ClientManager.get_embedder()` (GeminiEmbedder, `task_type="RETRIEVAL_DOCUMENT"`)
6. Sparse-вектор: через `ClientManager.get_sparse_embedder()` (fastembed BM25)
7. Payload: все поля контакта (`id`, `full_name`, `position`, `department`, `company`, `phone`)
8. Upsert батчами по 50 записей (из-за Gemini API лимитов)
9. Запуск: `uv run python -m scripts.migrate_contacts_to_qdrant`

---

### Этап 2 — Исправление FuzzyNameMatcher в HybridSearchService

#### [MODIFY] [hybrid_search.py](file:///e:/Old/bots/Worker/iTEMPO/iTEMPO_test/src/rag/retrieval/hybrid_search.py)

**Проблема:** `initialize()` читает имена из Postgres (падает при `UndefinedTableError: contacts`).

**Исправление:** Заменить чтение из Postgres на чтение из SQLite:
```python
# Было (asyncpg + Postgres):
conn = await asyncpg.connect(db_url)
rows = await conn.fetch("SELECT full_name FROM contacts ...")

# Стало (использование aiosqlite для безопасного неблокирующего доступа):
import aiosqlite
from pathlib import Path

db_path = Path(self.config.data_path) / "contacts.db"
async with aiosqlite.connect(str(db_path)) as db:
    async with db.execute("SELECT full_name FROM contacts WHERE full_name IS NOT NULL AND full_name != ''") as cursor:
        rows = await cursor.fetchall()
        contact_names = [r[0] for r in rows]
```

---

### Этап 3 — Новый класс ContactHybridSearch

#### [NEW] [contact_hybrid_search.py](file:///e:/Old/bots/Worker/iTEMPO/iTEMPO_test/src/rag/retrieval/contact_hybrid_search.py)

Отдельный легковесный класс, специализированный для поиска в коллекции `contacts_v1`:

- Принимает `semantic_query: str`, опциональные `company_filter: str`, `exact_phone: str`
- Строит `qdrant_filter` из `company_filter` (payload filter по полю `company` через `MatchText`) и `exact_phone` (точный `MatchText` по `phone`)
- Выполняет hybrid search (dense + sparse + RRF) аналогично `HybridSearchService._vector_search()`
- Возвращает список словарей с полями контакта (не RAG-поля)

#### [MODIFY] [contact_search.py](file:///e:/Old/bots/Worker/iTEMPO/iTEMPO_test/src/tools/contact_search.py)

1. Переименовать `search_query` → `semantic_query` в `ContactSearchInput`
2. Обновить описания полей для LLM
3. Заменить тело `ContactSearchTool.search()`: убрать `asyncpg`/SQL, вызвать `ContactHybridSearch`
4. Форматирование результата оставить **идентичным** текущему (чтобы не ломать `orchestrator.py` и промпты)

---

### Этап 4 — Оркестрация и промпты

#### [MODIFY] [orchestrator.py](file:///e:/Old/bots/Worker/iTEMPO/iTEMPO_test/src/agents/orchestrator.py)

1. В `__init__`: убрать `db_path=str(config.data_path / "contacts.db")` из `ContactSearchTool(...)` — он больше не нужен
2. В `search_contacts`: убедиться, что передаётся `semantic_query` (а не `search_query`) в `contact_tool.search()`
3. В `route_after_analysis`: расширить `conversational_phrases`:
   ```python
   "как тебя зовут", "что ты умеешь", "спасибо", "привет", "здравствуй",
   "как дела", "помоги мне", "что можешь", "ты кто"
   ```
4. В `generate_answer`: в `RULE 1` добавить пример: *"Если пользователь спрашивает 'как меня зовут' и его имя уже названо в ИСТОРИИ ДИАЛОГА — отвечай из истории, НЕ вызывай search tools."*

---

## Verification Plan

### Automated
```bash
uv run python -m scripts.migrate_contacts_to_qdrant   # должен завершиться без ошибок
uv run pytest src/tests/ -x -q                        # существующие тесты
```

### Manual
1. Запустить `uv run run_all_bots.py` — убедиться, что нет ошибок `UndefinedTableError: contacts` при старте
2. Запрос в боте: *"номер телефона Крылов Олег"* → должен вернуть контакт из Qdrant
3. Запрос: *"меня зовут Иван"*, затем *"как меня зовут"* → бот должен ответить из истории, не вызывая поиск
4. `docker compose down -v && docker compose up -d && uv run python -m scripts.migrate_contacts_to_qdrant` → убедиться, что поиск работает после полного сброса Docker volumes

# Чанкер документов

Модуль подготовки документов к индексации. Реализация — `src/rag/document_processor.py`, `src/rag/chunks_cache.py`, `src/rag/embeddings.py`, `src/rag/document_hasher.py`.

---

## Возможности

- **Форматы:** txt, pdf, doc, docx, xlsx
- **JSON-кэш:** автоматическая инвалидация при изменении файла (по SHA256-хэшу)
- **Rate Limiting:** соблюдение лимитов Gemini Embedding API (RPM, TPM)
- **Token-aware chunking:** разбиение с учётом фактического числа токенов
- **Инкрементальные обновления:** только изменённые файлы переиндексируются (`USE_INCREMENTAL_UPDATES=1`)
- **Parent-Child чанки:** двухуровневое разбиение — мелкие для поиска, крупные для LLM-контекста

---

## Структура данных

```
data/
  ├── .chunks_cache/           # JSON-кэш готовых чанков
  │   ├── document1.json
  │   └── document2.json
  ├── document1.txt
  └── document2.xlsx

qdrant_storage/
  └── document_hashes.json     # SHA256-хэши для инкрементальных обновлений
```

### Формат кэша

```json
{
  "source": "document.txt",
  "source_hash": "a1b2c3...",
  "created": "2026-01-27T10:30:00",
  "total_chunks": 32,
  "total_tokens": 15000,
  "chunks": [
    {
      "text": "...",
      "tokens": 450,
      "source": "document.txt"
    }
  ]
}
```

---

## xlsx обработка

Строки таблицы конвертируются в текстовый формат, первая строка — заголовки:

```
Номер: 4288 | ФИО: Забродин Никита Игоревич | Должность: Инженер | Отдел: Департамент ИТ
Номер: 7502 | ФИО: Попов Данил Николаевич | Должность: Дозиметрист | Отдел: Специальные
```

---

## Rate Limiting и KeyPool

Используется синергия индивидуальных лимитеров и общего пула ключей:

1. **Dual Token Bucket:** Каждому ключу назначен `AdaptiveRateLimiter` для контроля RPM и TPM. Коэффициент оценки токенов для русского языка — **1 токен ≈ 3.2 символа**.
2. **Proactive Rotation:** `KeyPool` мониторит утилизацию всех ключей. Если текущий ключ загружен на >75%, система автоматически переключается на свободный ключ **до** возникновения ошибки.
3. **Smart Retry:** При возникновении ошибки 429 система анализирует ответ. Если это минутный лимит (RPM), выполняется ожидание. Если суточный (RPD) — ключ изымается из пула.

```
[12:00:00] [Key1] Батч 1 (30 чанков) ✓ Utilization: 30%
[12:00:05] [Key1] Батч 2 (30 чанков) ✓ Utilization: 60%
[12:00:10] 🔄 Ротация: Key1 (78%) → Key2 (0%)
[12:00:11] [Key2] Батч 3 (30 чанков) ✓
```

---

## Инкрементальные обновления

`DocumentHasher` отслеживает SHA256 каждого файла. При запуске `process_documents.py` (или `process_documents.py --incremental`) заново индексируются только изменившиеся документы:

```python
from src.rag.document_hasher import DocumentHasher

hasher = DocumentHasher(config)
if hasher.has_changed(Path("data/doc.txt")):
    # переиндексировать
    hasher.update_hash(Path("data/doc.txt"))
hasher.save()  # сохранить в qdrant_storage/document_hashes.json
```

Управляется через `.env`:
```env
USE_INCREMENTAL_UPDATES=1
DOCUMENT_HASHES_PATH=qdrant_storage/document_hashes.json
```

---

## API

### ChunksCache

```python
from pathlib import Path
from src.rag.chunks_cache import ChunksCache

cache = ChunksCache(Path("data/.chunks_cache"))

# Получить из кэша или создать
chunks = cache.get_or_create(
    source_path=Path("data/document.txt"),
    chunker_fn=lambda p: create_chunks(p),
    force_refresh=False
)

# Статистика
cache.get_stats()
# {"cached_documents": 5, "total_chunks": 120, "total_tokens": 45000}

# Очистка
cache.clear_cache()                          # Весь кэш
cache.clear_cache(Path("data/document.txt")) # Один файл
```

### AdaptiveRateLimiter

```python
from src.rag.embeddings import AdaptiveRateLimiter

limiter = AdaptiveRateLimiter(max_rpm=90, max_tpm=28000)
await limiter.acquire(request_count=30, token_count=2400)

print(limiter.stats())
# "150 requests, 12000 tokens | Available: 60/90 RPM, 16000/28000 TPM"
```

---

## Скрипты управления базой знаний

| Скрипт | Назначение |
|--------|-----------|
| `scripts/process_documents.py` | Полный цикл: чанкинг + контекстуализация + векторизация. Поддерживает параметры `--force` (полная инициализация базы), `--chunk-only` (только чанкинг без векторизации), `--incremental` (принудительное инкрементальное обновление) и `--clear-cache` (очистка кэша чанков). |
| `scripts/evaluate_rag.py` | Оценка качества RAG (Recall@5, Recall@10, MRR, latency) |

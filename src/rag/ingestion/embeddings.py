"""
Управление векторной базой

Async обновление базы векторов с пулом API-ключей и проактивной ротацией.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from fastembed import SparseTextEmbedding
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, SparseIndexParams, SparseVector

from src.core.clients import ClientManager, GeminiEmbedder
from src.core.config import Config

logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """
    Dual token bucket rate limiter для одного API-ключа.

    Контролирует RPM и TPM независимо.
    """

    def __init__(self, max_rpm: int = 100, max_tpm: int = 30000, window_seconds: float = 60.0):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.window_seconds = window_seconds

        self._rpm_tokens = float(max_rpm) / 2
        self._rpm_per_second = max_rpm / window_seconds

        self._tpm_tokens = float(max_tpm) / 2
        self._tpm_per_second = max_tpm / window_seconds

        self._last_update = time.time()
        self._total_requests = 0
        self._total_tokens = 0

    def force_wait(self, seconds: float = 65.0):
        """Принудительно установить ожидание на `seconds` секунд (при внешней ошибке 429).

        Опустошает оба бакета (RPM и TPM), чтобы wait_seconds() вернул >= seconds.
        """
        self._rpm_tokens = 1.0 - (seconds * self._rpm_per_second)
        # FIX #1: опустошаем и TPM-бакет, иначе TPM-ошибки не блокируют TPM-проверку
        self._tpm_tokens = -(seconds * self._tpm_per_second)
        self._last_update = time.time()
        logger.info("[!] Лимитер принудительно заблокирован на %.0fs (RPM+TPM)", seconds)

    def _replenish(self) -> None:
        """Пополнить оба бакета исходя из прошедшего времени."""
        now = time.time()
        elapsed = now - self._last_update
        self._rpm_tokens = min(self.max_rpm, self._rpm_tokens + elapsed * self._rpm_per_second)
        self._tpm_tokens = min(self.max_tpm, self._tpm_tokens + elapsed * self._tpm_per_second)
        self._last_update = now

    def can_acquire(self, request_count: int, token_count: int) -> bool:
        """Проверить, можно ли выполнить запрос прямо сейчас (без ожидания)."""
        self._replenish()
        effective_req = min(request_count, self.max_rpm)
        effective_tok = min(token_count, self.max_tpm)
        return self._rpm_tokens >= effective_req and (token_count == 0 or self._tpm_tokens >= effective_tok)

    def utilization(self) -> float:
        """
        Уровень утilization бакета (0.0 — пустой, 1.0 — полный).

        Возвращает максимум из RPM и TPM utilization, чтобы отражать
        наиболее загруженный ресурс.
        """
        self._replenish()
        rpm_used = 1.0 - (self._rpm_tokens / self.max_rpm)
        tpm_used = 1.0 - (self._tpm_tokens / self.max_tpm)
        return max(rpm_used, tpm_used)

    async def acquire(self, request_count: int = 1, token_count: int = 0) -> None:
        """Ждать пока не будет доступности, затем списать токены."""
        while True:
            self._replenish()
            effective_req = min(request_count, self.max_rpm)
            effective_tok = min(token_count, self.max_tpm)

            rpm_ok = self._rpm_tokens >= effective_req
            tpm_ok = token_count == 0 or self._tpm_tokens >= effective_tok

            if rpm_ok and tpm_ok:
                self._rpm_tokens -= effective_req
                self._tpm_tokens -= effective_tok
                self._total_requests += effective_req
                self._total_tokens += effective_tok
                return

            wait_times = []
            if not rpm_ok:
                wait_times.append((effective_req - self._rpm_tokens) / self._rpm_per_second)
            if not tpm_ok:
                wait_times.append((effective_tok - self._tpm_tokens) / self._tpm_per_second)

            # min 0.05s — защита от busy-loop
            wait_time = max(max(wait_times), 0.05)
            logger.debug(
                "Rate limit: RPM %.0f/%d, TPM %.0f/%d, ожидание %.1fs...",
                self._rpm_tokens,
                self.max_rpm,
                max(0.0, self._tpm_tokens),
                self.max_tpm,
                wait_time,
            )
            await asyncio.sleep(wait_time)

    def debit(self, request_count: int, token_count: int) -> None:
        """Немедленно списать токены (без проверки доступности)."""
        self._replenish()
        self._rpm_tokens -= min(request_count, self.max_rpm)
        self._tpm_tokens -= min(token_count, self.max_tpm)
        self._total_requests += request_count
        self._total_tokens += token_count

    def wait_seconds(self, request_count: int, token_count: int) -> float:
        """Сколько секунд надо ждать для данного батча (0 если немедленно)."""
        self._replenish()
        effective_req = min(request_count, self.max_rpm)
        effective_tok = min(token_count, self.max_tpm)
        waits = []
        if self._rpm_tokens < effective_req:
            waits.append((effective_req - self._rpm_tokens) / self._rpm_per_second)
        if token_count > 0 and self._tpm_tokens < effective_tok:
            waits.append((effective_tok - self._tpm_tokens) / self._tpm_per_second)
        return max(waits) if waits else 0.0

    def stats(self) -> str:
        return (
            f"{self._total_requests} req, {self._total_tokens} tok | "
            f"RPM {max(0.0, self._rpm_tokens):.0f}/{self.max_rpm}, "
            f"TPM {max(0.0, self._tpm_tokens):.0f}/{self.max_tpm}"
        )


@dataclass
class _KeySlot:
    """Слот пула: ключ + лимитер + эмбеддер."""

    api_key: str
    limiter: AdaptiveRateLimiter
    embedder: GeminiEmbedder
    exhausted: bool = False  # суточный лимит исчерпан
    _exhausted_at: float = field(default=0.0, repr=False)

    def mark_exhausted(self) -> None:
        self.exhausted = True
        self._exhausted_at = time.time()

    def try_restore(self, reset_after: float = 3600.0) -> bool:
        """Восстановить ключ если прошло достаточно времени."""
        if self.exhausted and time.time() - self._exhausted_at >= reset_after:
            self.exhausted = False
            logger.info("[+] Ключ восстановлен: %s...%s", self.api_key[:4], self.api_key[-4:])
            return True
        return False


class KeyPool:
    """
    Пул API-ключей с индивидуальными лимитерами и равномерным round-robin распределением.

    Стратегия выбора:
    - Round-robin по всем доступным ключам (равномерная нагрузка).
    - При перегрузке текущего (wait > 0) — переключение на любой свободный.
    - Проактивная ротация при утилизации > ROTATION_THRESHOLD.
    """

    ROTATION_THRESHOLD = 0.75  # переключаемся при > 75% утилизации

    def __init__(self, slots: List[_KeySlot]):
        if not slots:
            raise ValueError("KeyPool требует хотя бы один ключ")
        self._slots = slots
        self._current_idx = 0
        # Round-robin счётчик для равномерного распределения при равной нагрузке
        self._rr_counter = 0

    @property
    def current(self) -> _KeySlot:
        return self._slots[self._current_idx]

    def best_slot(self, request_count: int, token_count: int) -> Optional[_KeySlot]:
        """
        Выбрать лучший слот для батча.

        Приоритеты:
        1. Слоты с нулевым ожиданием → round-robin среди них (равномерная нагрузка).
        2. Нет свободных → слот с минимальным временем ожидания.
        """
        for slot in self._slots:
            slot.try_restore()

        available = [s for s in self._slots if not s.exhausted]
        if not available:
            return None

        # Слоты готовые прямо сейчас (wait == 0)
        instant = [s for s in available if s.limiter.wait_seconds(request_count, token_count) == 0.0]
        if instant:
            # Round-robin среди готовых слотов вместо всегда первого по utilization.
            # Это гарантирует равномерное распределение нагрузки по всем ключам.
            rr_idx = self._rr_counter % len(instant)
            self._rr_counter += 1
            return instant[rr_idx]

        # Нет свободных — минимальное ожидание
        return min(available, key=lambda s: s.limiter.wait_seconds(request_count, token_count))

    def select(self, request_count: int, token_count: int) -> _KeySlot:
        """
        Выбор слота с проактивной ротацией.

        Всегда делегирует в best_slot() для равномерного распределения.
        Логирует переключение только если ключ реально меняется.
        """
        current = self.current
        best = self.best_slot(request_count, token_count)

        if best is None:
            return current  # Все исчерпаны — вернём текущий, вызывающий код обработает

        if best is not current:
            wait_current = current.limiter.wait_seconds(request_count, token_count)
            util_best = best.limiter.utilization()
            self._current_idx = self._slots.index(best)
            logger.debug(
                "[R] Ротация: %s...%s → %s...%s (wait_old=%.1fs, util_new=%.0f%%)",
                current.api_key[:4], current.api_key[-4:],
                best.api_key[:4], best.api_key[-4:],
                wait_current, util_best * 100,
            )
            return best

        return current

    def mark_current_exhausted(self) -> Optional[_KeySlot]:
        """Пометить текущий ключ как суточно исчерпанный и переключиться."""
        key = self.current.api_key
        self.current.mark_exhausted()
        logger.warning(
            "[!] Суточный лимит: ключ %s...%s помечен исчерпанным",
            key[:4],
            key[-4:],
        )
        available = [s for s in self._slots if not s.exhausted]
        if available:
            self._current_idx = self._slots.index(available[0])
            return self.current
        logger.error("[ERROR] Все ключи в пуле исчерпаны (суточный лимит)")
        return None

    def available_count(self) -> int:
        return sum(1 for s in self._slots if not s.exhausted)


class EmbeddingService:
    """Управление векторной базой с пулом ключей."""

    # Сигналы суточного (RPD) исчерпания квоты.
    _RPD_SIGNALS = ("PER_DAY", "PER-DAY", "DAILY_LIMIT", "DAILY-LIMIT", "DAY-LIMIT", "QUOTAID: EMBEDCONTENTREQUESTSPERDAY")
    # Сигналы исчерпания токенов в минуту (TPM)
    _TPM_SIGNALS = ("TOKENS_PER_MINUTE", "TPM", "TOKENS PER MINUTE")

    def _extract_retry_delay(self, error_str: str) -> float:
        """
        Extract recommended retry delay from API error message.
        
        Args:
            error_str: Complete error message from API
            
        Returns:
            float: Retry delay in seconds, 0 if not found
        """
        import re
        
        # Search for patterns like "RETRY IN 23S" or "RETRYDELAY": "22S"
        patterns = [
            r'RETRY.*?IN\s+(\d+)S',
            r'"RETRYDELAY":\s*"(\d+)S"',
            r'PLEASE RETRY IN (\d+\.\d+)S',
            r'RETRY IN (\d+)S'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return 0.0

    def __init__(self, config: Config):
        self.config = config
        self.client_manager = ClientManager.get_instance(config)
        self._current_model_idx = 0
        
        # FIX: Создаем пулы для ВСЕХ моделей заранее, чтобы сохранять состояние лимитеров
        self._pools: Dict[str, KeyPool] = {}
        models = config.embedding_models or [config.embedding_model]
        for m_name in models:
            try:
                self._pools[m_name] = self._build_pool(config, model_name=m_name)
            except Exception as e:
                logger.error("Не удалось инициализировать пул для модели %s: %s", m_name, e)

        if not self._pools:
            raise ValueError("Не удалось инициализировать ни один пул эмбеддингов")

        # FIX #2: Lock против race condition при параллельном async-доступе к слотам
        self._slot_lock: asyncio.Lock = asyncio.Lock()
        # FIX #6: явный пул потоков. 
        # Увеличиваем до 10, чтобы даже при 1 ключе были свободные потоки для новых попыток,
        # если предыдущие "повисли" на сетевом таймауте.
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="emb_")

        # Инициализация модели для генерации разреженных векторов
        logger.info("Инициализация модели разреженных векторов Qdrant/bm25...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def _build_pool(self, config: Config, model_name: Optional[str] = None) -> KeyPool:
        """Создать пул слотов для конкретной модели через ApiKeyManager."""
        from google import genai
        from google.genai import types as gtypes

        target_model = model_name or config.embedding_model
        logger.info("--- Инициализация пула ключей для модели: %s ---", target_model)

        # Используем ApiKeyManager из ClientManager
        api_keys = []
        if self.client_manager.api_key_manager:
            api_keys = self.client_manager.api_key_manager.api_keys
        elif config.gemini_api_key:
            api_keys = [config.gemini_api_key]

        if not api_keys:
            raise ValueError("Не задан ни один GEMINI_API_KEY")

        slots = []
        for key in api_keys:
            http_options = gtypes.HttpOptions(
                api_version=config.embedding_api_version,
                httpxClient=self.client_manager.get_gemini_http_client(),
            )
            gemini_client = genai.Client(api_key=key, http_options=http_options)
            embedder = GeminiEmbedder(
                model_name=target_model,
                gemini_client=gemini_client,
                output_dimensionality=config.vector_size,
            )
            limiter = AdaptiveRateLimiter(max_rpm=100, max_tpm=30000)
            slots.append(_KeySlot(api_key=key, limiter=limiter, embedder=embedder))
            logger.info("[+] Добавлен ключ в пул [%s]: %s...%s", target_model, key[:4], key[-4:])

        logger.info("[+] KeyPool [%s]: %d ключей", target_model, len(slots))
        return KeyPool(slots)

    def _estimate_tokens(self, texts: List[str]) -> int:
        return int(sum(len(t) / 2.0 for t in texts))

    async def _encode_with_retry(self, loop: asyncio.AbstractEventLoop, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """
        Энкодинг с автоматической ротацией ключей, моделей (fallback) и ретраями.
        """
        request_count = 1
        token_count = self._estimate_tokens(texts)
        models = self.config.embedding_models or [self.config.embedding_model]

        # Пытаемся по очереди модели, начиная с текущей (обычно 0)
        for m_idx in range(len(models)):
            # Пробуем текущую модель (или fallback, если переключились)
            current_model = models[m_idx]
            pool = self._pools.get(current_model)
            
            if not pool or all(s.exhausted for s in pool._slots):
                continue

            for attempt in range(max_retries):
                # FIX #2: Сериализация выбора слота
                async with self._slot_lock:
                    slot = pool.select(request_count, token_count)
                    wait = slot.limiter.wait_seconds(request_count, token_count)
                    
                    if wait > 0:
                        # Если текущая модель требует ожидания, а есть следующая (fallback) — 
                        # попробуем сначала её, прежде чем спать здесь.
                        if m_idx < len(models) - 1:
                            next_model = models[m_idx + 1]
                            next_pool = self._pools.get(next_model)
                            # Если у следующей модели есть свободный слот — переключаемся
                            if next_pool and any(not s.exhausted and s.limiter.wait_seconds(request_count, token_count) == 0 for s in next_pool._slots):
                                break # Уходим к следующей модели (fallback)
                        
                        # Если fallbacks нет или они тоже заняты — спим
                        if wait > 0.1:
                            logger.info("  --- [%s...%s] %s, ожидание %.1fs...", 
                                        slot.api_key[:4], slot.api_key[-4:], slot.limiter.stats(), wait)
                            await asyncio.sleep(wait)
                    
                    slot.limiter.debit(request_count, token_count)

                try:
                    embedder = slot.embedder
                    # FIX #6: таймаут 90с на случай зависания сетевого запроса в потоке
                    res = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            lambda e=embedder: e.encode(texts, task_type="RETRIEVAL_DOCUMENT", normalize=True),
                        ),
                        timeout=90.0
                    )
                    return res
                except Exception as e:
                    err_str = str(e).upper()
                    if not err_str or err_str == "()":
                        err_str = repr(e).upper()
                        
                    is_rate_error = any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "QUOTA"])

                    if is_rate_error:
                        retry_delay = self._extract_retry_delay(err_str)
                        
                        # Умное определение RPD (суточного лимита)
                        # Если в тексте есть "Day" или "Daily" и нет "Minute" — это суточный лимит.
                        is_rpd = ("QUOTA" in err_str) and \
                                 (any(x in err_str for x in ["DAY", "DAILY"]) or any(s in err_str for s in self._RPD_SIGNALS)) and \
                                 ("MINUTE" not in err_str)
                        
                        if is_rpd:
                            slot.mark_exhausted()
                            # Извлекаем лимит из сообщения для наглядности
                            limit_info = "???"
                            if "LIMIT:" in err_str:
                                try:
                                    limit_info = err_str.split("LIMIT:")[1].split()[0].replace(",", "")
                                except: pass
                                
                            logger.warning("![RPD] Ключ %s...%s ИСЧЕРПАН СУТОЧНО (Limit: %s) для модели %s", 
                                           slot.api_key[:4], slot.api_key[-4:], limit_info, current_model)
                            logger.warning("![RPD] Ключ %s...%s исчерпан суточно для модели %s", 
                                           slot.api_key[:4], slot.api_key[-4:], current_model)
                            
                            if all(s.exhausted for s in pool._slots):
                                logger.warning("⚠️ Все ключи модели %s исчерпаны (RPD).", current_model)
                                if m_idx < len(models) - 1:
                                    break # К следующей модели (fallback)
                                else:
                                    # Это была последняя модель. 
                                    # Если мы в fallback (m_idx > 0), просто возвращаемся к началу (m_idx=0)
                                    # и ждем сброса лимита на основной модели.
                                    if m_idx > 0:
                                        logger.warning("⚠️ Все fallback модели исчерпаны. Возврат к основной.")
                                        return await self._encode_with_retry(loop, texts, max_retries)
                                    raise RuntimeError("Все доступные модели исчерпали суточный лимит (RPD).")
                            continue

                        # RPM/TPM
                        retry_delay = self._extract_retry_delay(err_str)
                        wait_time = retry_delay if retry_delay > 0 else 65.0
                        slot.limiter.force_wait(wait_time)
                        
                        logger.warning("⏳ RPM/TPM на %s...%s. Пауза %.0fs...", 
                                       slot.api_key[:4], slot.api_key[-4:], wait_time)
                        
                        # Пробуем fallback только если он ЖИВОЙ
                        if m_idx < len(models) - 1:
                            next_pool = self._pools.get(models[m_idx + 1])
                            if next_pool and any(not s.exhausted for s in next_pool._slots):
                                break # Переходим к fallback
                        
                        # Иначе — остаемся на этой модели и пробуем снова (со сном в начале цикла)
                        continue

                    if attempt == max_retries - 1:
                        logger.error("❌ Ошибка %s: %s", current_model, err_str)
                        break

                    wait_time = 3.0 * (2 ** attempt)
                    await asyncio.sleep(wait_time)

        # Если мы здесь, значит за один проход по всем моделям не удалось получить результат.
        # Вместо падения, подождем немного и попробуем еще раз с первой модели (если она не RPD).
        first_model = models[0]
        first_pool = self._pools.get(first_model)
        if first_pool and any(not s.exhausted for s in first_pool._slots):
            logger.warning("🔄 Все модели заняты или RPD-исчерпаны. Ожидание 30с перед повторным циклом...")
            await asyncio.sleep(30.0)
            return await self._encode_with_retry(loop, texts, max_retries)

        raise RuntimeError("Критическая ошибка: все модели эмбеддингов исчерпали СУТОЧНЫЙ лимит (RPD).")

    async def _encode_sparse(self, loop, texts: List[str]) -> List[SparseVector]:
        """Генерация разреженных векторов с использованием fastembed и конвертация в SparseVector."""
        def _run():
            raw_embeddings = self.sparse_model.embed(texts)
            results = []
            for emb in raw_embeddings:
                results.append(
                    SparseVector(
                        indices=emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices),
                        values=emb.values.tolist() if hasattr(emb.values, "tolist") else list(emb.values)
                    )
                )
            return results

        return await loop.run_in_executor(self._executor, _run)

    async def _qdrant_op(self, loop, func, *args):
        """Простой вызов Qdrant-операции с ретраями."""
        for attempt in range(3):
            try:
                return await loop.run_in_executor(None, lambda f=func, a=args: f(*a))
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2.0 * (2**attempt))
                logger.warning("⚠️ Qdrant error (attempt %d): %s", attempt + 1, e)

    async def update_database(self, chunks: List[Dict[str, Any]]):
        """Полное обновление базы (recreate collection)."""
        client = self.client_manager.get_qdrant_client()
        loop = asyncio.get_event_loop()

        # Явное удаление и создание для надежности в локальном режиме
        if client.collection_exists(self.config.collection_name):
            logger.info("[-] Удаление старой коллекции %s...", self.config.collection_name)
            client.delete_collection(self.config.collection_name)

        logger.info("[+] Создание коллекции %s (size=%d, с разреженными векторами)...", self.config.collection_name, self.config.vector_size)
        await self._qdrant_op(
            loop,
            lambda: client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=self.config.vector_size, distance=Distance.COSINE),
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))},
            ),
        )

        batch_size = 8
        total = len(chunks)
        print(f"Генерация векторов через {self.config.embedding_model}...")

        start_time = time.time()
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            await self._upsert_chunks(batch, client, loop)
            elapsed = time.time() - start_time
            # Берем статистику из пула основной модели
            main_model = self.config.embedding_models[0] if self.config.embedding_models else self.config.embedding_model
            pool = self._pools.get(main_model)
            slot_stats = pool.current.limiter.stats() if pool else "N/A"
            print(f"  [{slot_stats}] Обработано {min(i + batch_size, total)}/{total} за {elapsed:.1f}s")

        print(f"[+] База знаний обновлена за {time.time() - start_time:.1f}s")

    async def incremental_update(self, chunks: List[Dict[str, str]], target_sources: Optional[List[str]] = None):
        """
        Инкрементальное обновление: удаление по source + добавление новых чанков.

        Args:
            chunks: Список чанков для загрузки.
            target_sources: Явный список source-ключей для обработки.
                            Гарантирует удаление даже пустых (удалённых) файлов.
        """
        client = self.client_manager.get_qdrant_client()
        loop = asyncio.get_event_loop()

        collection_exists = await loop.run_in_executor(
            None, lambda: client.collection_exists(self.config.collection_name)
        )
        if not collection_exists:
            await self._qdrant_op(
                loop,
                lambda: client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(size=self.config.vector_size, distance=Distance.COSINE),
                    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))},
                ),
            )
            print("Создана новая коллекция")

        chunks_by_source: Dict[str, List[Dict[str, str]]] = {}
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            chunks_by_source.setdefault(source, []).append(chunk)

        sources_to_process = target_sources if target_sources is not None else list(chunks_by_source.keys())
        batch_size = 8
        total_chunks = len(chunks)
        processed = 0

        print(f"Генерация векторов для {len(sources_to_process)} документов ({total_chunks} чанков)")

        start_time = time.time()
        for idx, source in enumerate(sources_to_process, 1):
            source_chunks = chunks_by_source.get(source, [])
            print(f"\n[{idx}/{len(sources_to_process)}] {source} ({len(source_chunks)} чанков)")
            await self._delete_by_source(client, loop, source)

            for i in range(0, len(source_chunks), batch_size):
                batch = source_chunks[i : i + batch_size]
                await self._upsert_chunks(batch, client, loop)
                processed += len(batch)
                elapsed = time.time() - start_time
                main_model = self.config.embedding_models[0] if self.config.embedding_models else self.config.embedding_model
                pool = self._pools.get(main_model)
                slot_stats = pool.current.limiter.stats() if pool else "N/A"
                print(f"  [{slot_stats}] {processed}/{total_chunks} за {elapsed:.1f}s")

        print(f"\n[+] Обновление завершено за {time.time() - start_time:.1f}s")

    async def _delete_by_source(self, client, loop, source: str):
        filter_obj = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        await self._qdrant_op(loop, client.delete, self.config.collection_name, filter_obj)

    async def _upsert_chunks(self, batch: List[Dict], client, loop):
        valid_items = [doc for doc in batch if doc.get("text", "").strip()]
        if not valid_items:
            return

        texts = [item["text"] for item in valid_items]
        # Генерация плотных векторов
        dense_embeddings = await self._encode_with_retry(loop, texts)
        # Генерация разреженных векторов
        sparse_embeddings = await self._encode_sparse(loop, texts)

        points = []
        for doc, dense_emb, sparse_emb in zip(valid_items, dense_embeddings, sparse_embeddings, strict=False):
            payload = {
                k: v for k, v in doc.items() if k != "document_text" and isinstance(v, (str, int, float, bool, list, dict))
            }
            # Передаем именованные векторы: "" (dense) и "sparse" (sparse)
            vector_data = {
                "": dense_emb if isinstance(dense_emb, list) else dense_emb.tolist(),
                "sparse": sparse_emb
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector_data, payload=payload))

        await self._qdrant_op(loop, client.upsert, self.config.collection_name, points)

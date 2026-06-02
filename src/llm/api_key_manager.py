"""
Менеджер API ключей с циклическим fallback.

Управляет множественными API ключами Gemini и автоматически
переключается между ними при исчерпании лимитов.
"""

import logging
import time
from threading import RLock
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class ApiKeyManager:
    """
    Thread-safe менеджер для управления множественными API ключами.

    Особенности:
    - Циклическое переключение между ключами
    - Отслеживание исчерпанных ключей
    - Thread-safe операции
    - Автоматическое восстановление исчерпанных ключей через время
    """

    # Классовые переменные для совместного использования состояния между всеми инстансами/потоками
    _global_lock = RLock()
    _global_exhausted_keys: Set[str] = set()
    _global_current_index = 0
    _global_last_reset_time = time.time()

    @property
    def _current_index(self) -> int:
        return ApiKeyManager._global_current_index

    @_current_index.setter
    def _current_index(self, value: int):
        ApiKeyManager._global_current_index = value

    @property
    def _last_reset_time(self) -> float:
        return ApiKeyManager._global_last_reset_time

    @_last_reset_time.setter
    def _last_reset_time(self, value: float):
        ApiKeyManager._global_last_reset_time = value

    def __init__(self, api_keys: List[str], reset_interval: int = 3600, auto_rotate: bool = False):
        """
        Args:
            api_keys: Список API ключей
            reset_interval: Интервал сброса исчерпанных ключей (в секундах)
            auto_rotate: Если True, автоматически переключать ключ для каждого вызова get_current_key()
        """
        if not api_keys:
            raise ValueError("Необходим хотя бы один API ключ")

        self.api_keys = api_keys
        self.reset_interval = reset_interval
        self.auto_rotate = auto_rotate
        self._lock = ApiKeyManager._global_lock
        self._exhausted_keys = ApiKeyManager._global_exhausted_keys

        logger.info(f"ApiKeyManager инициализирован с {len(api_keys)} ключами, auto_rotate={auto_rotate}")

    def get_masked_key(self, key: str) -> str:
        """
        Получить маскированную версию API ключа для логирования.

        Args:
            key: API ключ для маскирования

        Returns:
            str: Маскированный ключ в формате AIza...xX42
        """
        return self._mask_key(key)

    def get_current_key(self) -> str:
        """
        Get current active API key.
        
        If auto_rotate is enabled, automatically switches to next available key
        for each call (round-robin across non-exhausted keys).

        Returns:
            str: Current API key
        """
        with self._lock:
            self._check_reset_exhausted()
            
            # First try to use current key
            current_key = self.api_keys[self._current_index]
            
            # If current key is exhausted and auto_rotate is enabled, find next available key
            if current_key in self._exhausted_keys and self.auto_rotate:
                # Start from current position and find next non-exhausted key
                attempts = 0
                max_attempts = len(self.api_keys)
                
                while attempts < max_attempts:
                    self._current_index = (self._current_index + 1) % len(self.api_keys)
                    next_key = self.api_keys[self._current_index]
                    
                    if next_key not in self._exhausted_keys:
                        # Found available key
                        masked_key = self.get_masked_key(next_key)
                        logger.debug(f"Key auto-rotated to: {masked_key} (Index: {self._current_index})")
                        return next_key
                    
                    attempts += 1
                
                # If all keys are exhausted, reset to first key and use it anyway
                self._current_index = 0
                current_key = self.api_keys[self._current_index]
                logger.warning(f"All keys exhausted, using first key: {self.get_masked_key(current_key)}")
                return current_key
            
            # Return current key if it's available
            masked_key = self.get_masked_key(current_key)
            logger.debug(f"Using API key: {masked_key} (Index: {self._current_index})")
            return current_key

    def rotate_key(self, reason: str = "unknown") -> Optional[str]:
        """
        Переключиться на следующий доступный ключ.

        Args:
            reason: Причина переключения (для логирования)

        Returns:
            Optional[str]: Новый активный ключ или None, если все ключи исчерпаны
        """
        with self._lock:
            self._check_reset_exhausted()

            current_key = self.api_keys[self._current_index]

            # Пытаемся найти следующий неисчерпанный ключ
            attempts = 0
            max_attempts = len(self.api_keys)

            while attempts < max_attempts:
                self._current_index = (self._current_index + 1) % len(self.api_keys)
                next_key = self.api_keys[self._current_index]

                if next_key not in self._exhausted_keys:
                    # Нашли доступный ключ
                    masked_old = self.get_masked_key(current_key)
                    masked_new = self.get_masked_key(next_key)
                    if reason == "rate limit" or "429" in reason:
                        logger.debug(f"⚠️ Лимит исчерпан. Переключаюсь на ключ №{self._current_index}: {masked_new}")
                    else:
                        logger.debug(f"🔄 Переключение API ключа: {masked_old} → {masked_new} (причина: {reason})")
                    return next_key

                attempts += 1

            # Все ключи исчерпаны
            self.get_pool_health()
            logger.debug(
                f"❌ Все {len(self.api_keys)} API ключей исчерпаны! "
                f"Следующий сброс через {self._time_until_reset():.0f} сек."
            )
            return None

    def mark_key_exhausted(self, api_key: str, reason: str = "rate limit"):
        """
        Пометить ключ как исчерпанный.

        Args:
            api_key: API ключ для пометки
            reason: Причина исчерпания
        """
        with self._lock:
            if api_key in self.api_keys:
                self._exhausted_keys.add(api_key)
                masked = self.get_masked_key(api_key)
                logger.debug(
                    f"⚠️ Ключ {masked} помечен как исчерпанный "
                    f"(причина: {reason}). "
                    f"Исчерпано: {len(self._exhausted_keys)}/{len(self.api_keys)}"
                )

    def reset_exhausted_keys(self):
        """
        Сбросить все пометки об исчерпанных ключах.

        Используется для периодического восстановления ключей
        после истечения времени ожидания лимитов.
        """
        with self._lock:
            if self._exhausted_keys:
                count = len(self._exhausted_keys)
                self._exhausted_keys.clear()
                self._last_reset_time = time.time()
                logger.info(f"♻️ Сброшено {count} исчерпанных ключей")

    def get_available_keys_count(self) -> int:
        """
        Получить количество доступных (неисчерпанных) ключей.

        Returns:
            int: Количество доступных ключей
        """
        with self._lock:
            self._check_reset_exhausted()
            return len(self.api_keys) - len(self._exhausted_keys)

    def get_pool_health(self) -> dict:
        """
        Получить информацию о "здоровье" пула ключей.

        Returns:
            dict: Информация о состоянии пула ключей
        """
        with self._lock:
            self._check_reset_exhausted()
            active_count = len(self.api_keys) - len(self._exhausted_keys)
            exhausted_count = len(self._exhausted_keys)
            total_count = len(self.api_keys)
            
            health_info = {
                'total_keys': total_count,
                'active_keys': active_count,
                'exhausted_keys': exhausted_count,
                'current_index': self._current_index,
                'time_until_reset': self._time_until_reset()
            }
            
            logger.info(
                f"🏊 Здоровье пула: {active_count} активных, {exhausted_count} на отдыхе "
                f"(всего: {total_count})"
            )
            
            return health_info

    def is_all_exhausted(self) -> bool:
        """
        Проверить, исчерпаны ли все ключи.

        Returns:
            bool: True если все ключи исчерпаны
        """
        with self._lock:
            self._check_reset_exhausted()
            return len(self._exhausted_keys) >= len(self.api_keys)

    def _check_reset_exhausted(self):
        """
        Проверить и сбросить исчерпанные ключи если прошло достаточно времени.

        ВНИМАНИЕ: Должен вызываться только внутри lock!
        """
        elapsed = time.time() - self._last_reset_time
        if elapsed >= self.reset_interval and self._exhausted_keys:
            count = len(self._exhausted_keys)
            self._exhausted_keys.clear()
            self._last_reset_time = time.time()
            logger.info(f"♻️ Автоматический сброс {count} исчерпанных ключей (прошло {elapsed:.0f} сек)")

    def _time_until_reset(self) -> float:
        """
        Время до следующего автоматического сброса.

        Returns:
            float: Секунды до сброса
        """
        elapsed = time.time() - self._last_reset_time
        return max(0, self.reset_interval - elapsed)

    @staticmethod
    def _mask_key(api_key: str) -> str:
        """
        Маскировать API ключ для безопасного логирования.

        Args:
            api_key: Полный API ключ

        Returns:
            str: Замаскированный ключ (показываем только первые/последние символы)
        """
        if len(api_key) <= 8:
            return "***"
        return f"{api_key[:4]}...{api_key[-4:]}"

"""
Менеджер для fallback переключения между текстовыми моделями при 503 ошибках.
"""

import logging
import time
from threading import RLock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Маппинг моделей на их API версии
MODEL_API_VERSIONS: Dict[str, str] = {
    # v1 models
    "gemini-1.5-flash": "v1",
    "gemini-1.5-pro": "v1",
    "gemini-1.0-pro": "v1",
    # v1beta models
    "gemini-2.0-flash-exp": "v1beta",
    "gemini-2.0-flash": "v1beta",
    "gemini-exp-1206": "v1beta",
    "gemini-2.0-flash-lite": "v1beta",
    "gemini-2.0-flash-lite-preview": "v1beta",
    "gemini-2.5-flash": "v1beta",
    "gemini-2.5-flash-lite": "v1beta",
    "gemini-3.1-flash-lite": "v1beta",
    "gemini-3.5-flash": "v1beta",
}

# Порог cooldown для разделения RPM/RPD (секунды).
# Cooldowns <= этого значения считаются RPM (короткие), > — RPD (длинные).
_RPM_COOLDOWN_THRESHOLD = 3600  # 1 час


class ModelFallbackManager:
    """
    Менеджер для переключения между текстовыми моделями с учетом кулдаунов.

    Различает два типа кулдаунов:
    - RPM (per-minute): короткий (30-60с), стоит подождать
    - RPD (daily): длинный (12ч), модель исчерпана на сегодня
    """

    # Классовые переменные для совместного использования состояния кулдаунов всеми инстансами
    _global_cooldowns: Dict[str, float] = {}
    _global_lock = RLock()

    def __init__(self, primary_model: str, fallback_models: List[str]):
        self.primary_model = primary_model
        self.fallback_models = fallback_models

        # Полный список моделей в порядке приоритета
        self.all_models: List[str] = []
        for model in [primary_model] + fallback_models:
            if model not in self.all_models:
                self.all_models.append(model)

        self._lock = ModelFallbackManager._global_lock
        self._cooldowns = ModelFallbackManager._global_cooldowns

        logger.info("📋 ModelFallbackManager инициализирован с поддержкой кулдаунов.")

    def get_best_available_model(self) -> str:
        """
        Найти лучшую доступную модель (самый высокий тир без активного кулдауна).
        """
        with self._lock:
            now = time.time()
            for model in self.all_models:
                unlock_time = self._cooldowns.get(model, 0)
                if now >= unlock_time:
                    return model

            # Если все в кулдауне, берем ту с кратчайшим RPM кулдауном
            # (RPD модели не рассматриваются — они исчерпаны на сегодня)
            rpm_models = []
            for model in self.all_models:
                unlock_time = self._cooldowns.get(model, 0)
                remaining = unlock_time - now
                if remaining <= _RPM_COOLDOWN_THRESHOLD:
                    rpm_models.append((remaining, model))

            if rpm_models:
                rpm_models.sort(key=lambda x: x[0])
                return rpm_models[0][1]

            # Если все в RPD кулдауне — возвращаем primary (будет RPD ошибка)
            return self.primary_model

    def mark_model_rpm_limit(self, model_name: str, seconds: int = 60):
        """Пометить модель как временно недоступную (лимит в минуту)."""
        with self._lock:
            self._cooldowns[model_name] = time.time() + seconds
            logger.debug(f"⏳ Модель {model_name} на кулдауне (RPM) на {seconds}с")

    def mark_model_daily_exhausted(self, model_name: str):
        """Пометить модель как исчерпанную на сегодня (дневной лимит)."""
        with self._lock:
            # Блокируем на 12 часов (или до рестарта)
            self._cooldowns[model_name] = time.time() + (12 * 3600)
            logger.debug(f"🚫 Модель {model_name} ИСЧЕРПАНА на сегодня (RPD limit)")

    def is_model_available(self, model_name: str) -> bool:
        """Проверить, доступна ли модель (нет кулдауна)."""
        with self._lock:
            return time.time() >= self._cooldowns.get(model_name, 0)

    def is_model_daily_exhausted(self, model_name: str) -> bool:
        """Проверить, исчерпана ли модель на сегодня (RPD)."""
        with self._lock:
            unlock_time = self._cooldowns.get(model_name, 0)
            remaining = unlock_time - time.time()
            return remaining > _RPM_COOLDOWN_THRESHOLD

    def get_min_cooldown_wait(self) -> Optional[float]:
        """
        Получить минимальное время ожидания до разблокировки хотя бы одной RPM-модели.
        Возвращает None если есть доступная модель или все в RPD.
        """
        with self._lock:
            now = time.time()
            min_wait = None
            for model in self.all_models:
                unlock_time = self._cooldowns.get(model, 0)
                remaining = unlock_time - now
                if remaining <= 0:
                    return 0.0  # Есть доступная модель
                if remaining <= _RPM_COOLDOWN_THRESHOLD:
                    # Это RPM cooldown — стоит подождать
                    if min_wait is None or remaining < min_wait:
                        min_wait = remaining
            return min_wait

    def get_api_version_for_model(self, model_name: str) -> str:
        if model_name in MODEL_API_VERSIONS:
            return MODEL_API_VERSIONS[model_name]
        for known_model, api_version in MODEL_API_VERSIONS.items():
            if model_name.startswith(known_model):
                return api_version
        return "v1beta"

    def rotate_model(self, current_model: str, reason: str = "error") -> Optional[str]:
        """
        Используется во время ретраев внутри одного запроса.
        Возвращает следующую доступную модель (не в cooldown).
        Возвращает None если все модели заблокированы.
        """
        with self._lock:
            try:
                idx = self.all_models.index(current_model)
            except ValueError:
                idx = 0

            now = time.time()
            for i in range(1, len(self.all_models)):
                next_idx = (idx + i) % len(self.all_models)
                candidate = self.all_models[next_idx]
                if now >= self._cooldowns.get(candidate, 0):
                    return candidate

            # Все заблокированы — возвращаем None.
            # Вызывающий код должен подождать (asyncio.sleep) и повторить.
            return None

    def reset_to_primary(self):
        """Сброс не требуется, так как get_best_available_model всегда проверяет приоритеты."""
        pass

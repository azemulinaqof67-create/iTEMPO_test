import os
import yaml  # type: ignore
from typing import Dict, Any, Optional

class PromptManager:
    """
    Класс для управления системными промптами (Prompt as a Configuration).
    Загружает шаблоны из YAML-файлов и предоставляет к ним доступ.
    Реализует паттерн Singleton.
    """
    _instance = None

    def __init__(self, prompts_dir: Optional[str] = None):
        if PromptManager._instance is not None:
            raise RuntimeError("Используйте PromptManager.get_instance() для получения экземпляра.")
        
        if prompts_dir is None:
            # src/core/prompt_manager.py -> src/prompts
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompts_dir = os.path.join(base_dir, "prompts")
            
        self.prompts_dir = prompts_dir
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.load_prompts()

    @classmethod
    def get_instance(cls, prompts_dir: Optional[str] = None) -> "PromptManager":
        if cls._instance is None:
            cls._instance = cls(prompts_dir)
        return cls._instance

    @classmethod
    def reset(cls):
        """Сброс синглтона (используется для тестов)."""
        cls._instance = None

    def load_prompts(self):
        """Загрузка всех YAML-конфигураций промптов из директории."""
        if not os.path.exists(self.prompts_dir):
            return
            
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(self.prompts_dir, filename)
                name = os.path.splitext(filename)[0]
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                        self.prompts[name] = data
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Ошибка загрузки промпта {filename}: {e}")

    def get_prompt(self, name: str, key: str = "template") -> str:
        """
        Возвращает шаблон промпта по имени файла (без расширения) и ключу.
        Если шаблон не найден, возвращает пустую строку.
        """
        prompt_data = self.prompts.get(name, {})
        return prompt_data.get(key, "")

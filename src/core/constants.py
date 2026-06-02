"""
Глобальные константы проекта.
"""

import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_COMPANIES = {
    "itz": 'АО "ИТЗ"',
    "kmk": 'АО "КМК "ТЭМПО"',
    "ntz": 'АО "НТЗ "ТЭМ-ПО"',
    "technotron": 'АО "ПТФК "Технотрон"',
    "metiz": 'ООО "Технотрон-Метиз"',
    "kzmk": 'АО "КЗМК "ТЭМПО"',
    "zteo": 'АО "ПТФК "ЗТЭО"',
    "td": 'АО "ТД "ТЭМПО"',
    "sks": 'АО "СКС "ТЭМПО"',
    "port": 'ООО "ТЭМПО-ПОРТ"',
    "it": 'ООО "АЙТИ "ТЭМПО"',
}

def load_companies() -> dict:
    # Ищем models_config.yaml начиная с текущей папки и вверх по дереву
    current_dir = Path(__file__).resolve().parent
    for _ in range(5):
        yaml_file = current_dir / "models_config.yaml"
        if yaml_file.exists():
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data and isinstance(data, dict) and "companies" in data:
                        return data["companies"]
            except Exception as e:
                logger.warning(f"Ошибка загрузки компаний из {yaml_file}: {e}")
        current_dir = current_dir.parent
    return DEFAULT_COMPANIES

COMPANIES = load_companies()


import logging
from typing import Optional

from src.services.weather_service import get_weather

logger = logging.getLogger(__name__)


class WeatherSearchTool:
    """
    Инструмент для получения текущей погоды и прогноза.
    """

    def __init__(self, default_city: str = "Набережные Челны"):
        self.default_city = default_city

    async def search(self, location: Optional[str] = None) -> str:
        """
        Выполняет поиск погоды.
        """
        search_city = location or self.default_city
        logger.info(f"[WEATHER] Searching for: {search_city}")

        try:
            return await get_weather(search_city)
        except Exception as e:
            logger.error(f"Weather tool error: {e}")
            return f"К сожалению, не удалось получить данные о погоде для {search_city}."

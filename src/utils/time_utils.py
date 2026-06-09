"""
Утилиты для работы со временем и датами.
"""

import datetime
from typing import Any, Dict


def get_current_time_info(timezone_offset: int = 3) -> Dict[str, Any]:
    """
    Получает текущую информацию о времени с учетом временной зоны.

    Args:
        timezone_offset: Смещение в часах от UTC (по умолчанию +3 для Московского времени)

    Returns:
        Dict[str, Any]: Словарь с информацией о времени
    """
    # Получаем текущее UTC время
    now_utc = datetime.datetime.utcnow()

    # Применяем смещение временной зоны
    current_time = now_utc + datetime.timedelta(hours=timezone_offset)

    # Форматируем различные представления времени
    time_info = {
        "current_datetime": current_time,
        "current_date": current_time.strftime("%d.%m.%Y"),
        "current_time": current_time.strftime("%H:%M"),
        "current_time_full": current_time.strftime("%H:%M:%S"),
        "current_day_of_week": current_time.strftime("%A"),
        "current_day_of_week_ru": _get_day_of_week_russian(current_time.weekday()),
        "is_weekend": current_time.weekday() >= 5,  # 5=Суббота, 6=Воскресенье
        "timezone_offset": timezone_offset,
        "timezone_name": "Московское время (UTC+3)",
    }

    return time_info


def _get_day_of_week_russian(day_index: int) -> str:
    """
    Возвращает название дня недели на русском.

    Args:
        day_index: Индекс дня недели (0=Понедельник, 6=Воскресенье)

    Returns:
        str: Название дня недели на русском
    """
    days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    return days[day_index]


def format_time_for_prompt(time_info: Dict[str, Any]) -> str:
    """
    Formats time information for prompt insertion.

    Args:
        time_info: Time information

    Returns:
        str: Formatted text for prompt
    """
    return f"""
CURRENT TIME INFORMATION:
- Current date: {time_info["current_date"]} ({time_info["current_day_of_week_ru"]})
- Current time: {time_info["current_time"]}
- Time zone: {time_info["timezone_name"]}
- Day type: {"Weekend" if time_info["is_weekend"] else "Weekday"}
""".strip()

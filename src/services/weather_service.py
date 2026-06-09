import logging

import aiohttp

logger = logging.getLogger(__name__)


async def get_weather(location: str) -> str:
    """
    Получает текущую погоду для указанного города.

    Args:
        location: Название города (например, 'Набережные Челны' или 'Москва')
    """
    try:
        # 1. Геокодинг (ищем координаты города)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=ru&format=json"

        async with aiohttp.ClientSession() as session:
            async with session.get(geo_url) as resp:
                if resp.status != 200:
                    return f"Ошибка геокодинга: не удалось связаться с сервисом (код {resp.status})"

                geo_data = await resp.json()
                if not geo_data.get("results"):
                    return f"Город '{location}' не найден. Пожалуйста, уточни название."

                city_info = geo_data["results"][0]
                lat = city_info["latitude"]
                lon = city_info["longitude"]
                city_name = city_info.get("name", location)

            # 2. Получение погоды по координатам (текущая + прогноз на завтра)
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m"
                f"&daily=weather_code,temperature_2m_max,temperature_2m_min"
                f"&wind_speed_unit=ms&timezone=auto"
            )

            async with session.get(weather_url) as resp:
                if resp.status != 200:
                    return f"Ошибка API погоды (код {resp.status})"

                w_data = await resp.json()
                current = w_data.get("current", {})
                daily = w_data.get("daily", {})

                # Текущая
                temp = current.get("temperature_2m")
                apparent = current.get("apparent_temperature")
                humidity = current.get("relative_humidity_2m")
                wind = current.get("wind_speed_10m")
                weather_desc = _decode_weather(current.get("weather_code", 0))

                result = (
                    f"Погода в г. {city_name} сейчас:\n"
                    f"🌡 Температура: {temp}°C (ощущается как {apparent}°C)\n"
                    f"☁ Состояние: {weather_desc}\n"
                    f"💧 Влажность: {humidity}%\n"
                    f"💨 Ветер: {wind} м/с\n\n"
                )

                # Прогноз на 7 дней (включая сегодня)
                if daily and "time" in daily:
                    result += "Прогноз на неделю (ИСПОЛЬЗУЙ ТОЛЬКО ДЛЯ ОТВЕТА НА ЗАПРОШЕННЫЕ ДНИ):\n"
                    times = daily["time"]
                    max_temps = daily["temperature_2m_max"]
                    min_temps = daily["temperature_2m_min"]
                    codes = daily["weather_code"]

                    for i in range(len(times)):
                        date_str = times[i]
                        max_t = max_temps[i]
                        min_t = min_temps[i]
                        desc = _decode_weather(codes[i])
                        result += f"📅 {date_str}: от {min_t}°C до {max_t}°C, {desc}\n"

                return result

    except Exception as e:
        logger.error(f"Error fetching weather for {location}: {e}")
        return f"К сожалению, не удалось получить данные о погоде для '{location}' из-за технической ошибки."


def _decode_weather(code: int) -> str:
    """Расшифровка кодов WMO"""
    codes = {
        0: "Ясно ☀",
        1: "Преимущественно ясно 🌤",
        2: "Переменная облачность 🌥",
        3: "Пасмурно ☁",
        45: "Туман 🌫",
        48: "Иней 🌫",
        51: "Легкая морось 🌧",
        53: "Умеренная морось 🌧",
        55: "Плотная морось 🌧",
        61: "Небольшой дождь 🌧",
        63: "Умеренный дождь 🌧",
        65: "Сильный дождь 🌧",
        71: "Небольшой снегопад 🌨",
        73: "Умеренный снегопад 🌨",
        75: "Сильный снегопад 🌨",
        77: "Ледяные иглы ❄",
        80: "Слабый ливень 🌦",
        81: "Умеренный ливень 🌦",
        82: "Сильный ливень 🌧",
        85: "Небольшой снежный ливень 🌨",
        86: "Сильный снежный ливень 🌨",
        95: "Гроза ⚡",
        96: "Гроза со слабым градом ⚡🌨",
        99: "Гроза с сильным градом ⚡🌨",
    }
    return codes.get(code, "Неопределенно")

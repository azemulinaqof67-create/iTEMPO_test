#!/bin/bash
# VPN Watchdog (Автоматическое восстановление туннеля)
# Проверяет страну текущего внешнего IP-адреса. 
# Если возвращается Россия (RU) или случается таймаут — принудительно перезапускает VPN.

LOG_FILE="/var/log/vpn_watchdog.log"
TIMEOUT=10

# Запрашиваем код страны текущего интернет-соединения
COUNTRY=$(curl -s --max-time $TIMEOUT ipinfo.io/country)

# Очищаем ответ от лишних пробелов/переносов строк
COUNTRY=$(echo "$COUNTRY" | tr -d '[:space:]')

# Если переменная пустая (таймаут/нет сети) или равна RU (голый российский IP)
if [ -z "$COUNTRY" ] || [ "$COUNTRY" == "RU" ]; then
    echo "$(date): Обнаружена проблема с VPN (Страна: ${COUNTRY:-ОШИБКА СЕТИ}). Перезапуск sing-box..." >> "$LOG_FILE"
    systemctl restart sing-box
else
    # Всё хорошо. Строка закомментирована, чтобы не спамить в лог каждую минуту.
    # echo "$(date): VPN работает стабильно (Страна: $COUNTRY)." >> "$LOG_FILE"
    exit 0
fi

#!/bin/bash
# Скрипт для обновления зависимостей на сервере

echo "🔄 Обновление зависимостей для работы с SOCKS5 прокси..."

# Обновление uv
echo "1. Обновление uv..."
pip install --upgrade uv

# Синхронизация зависимостей с поддержкой SOCKS
echo "2. Установка зависимостей..."
uv sync

# Проверка установки socksio
echo "3. Проверка socksio..."
uv run python -c "import socksio; print('✅ socksio установлен')" || echo "❌ socksio не установлен"

# Проверка httpx с socks
echo "4. Проверка httpx[socks]..."
uv run python -c "import httpx; print('✅ httpx готов к работе с SOCKS')" || echo "❌ httpx не готов"

echo "✅ Обновление завершено!"
echo "Теперь запустите: uv run python test_proxy_simple.py"

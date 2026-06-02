#!/bin/bash
echo "Запускаем Cloudflare..."
pkill -f "cloudflared tunnel --url"
cloudflared tunnel --url http://127.0.0.1:8000 > cloudflare.log 2>&1 &
echo "Ожидание ссылки от серверов Cloudflare..."
for i in {1..15}; do
    URL=$(grep -a -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' cloudflare.log | head -n 1)
    if [ -n "$URL" ]; then
        break
    fi
    sleep 1
done
if [ -n "$URL" ]; then
    echo "✅ Тоннель получен: $URL"
    sed -i "s|^MAX_WEBHOOK_URL=.*|MAX_WEBHOOK_URL=$URL/webhook/max|g" .env
    /home/administrator/.local/bin/uv run run_all_bots.py
else
    echo "❌ Ошибка тоннеля!"
    cat cloudflare.log
fi

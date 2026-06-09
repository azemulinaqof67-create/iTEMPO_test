import logging
import sqlite3
import os
import requests
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    """Очистка и приведение ФИО к единому регистру для сопоставления."""
    if not name:
        return ""
    return " ".join(name.lower().strip().split())

def fetch_yandex_users(token: str, org_id: str) -> List[Dict[str, Any]]:
    """Получить полный список пользователей организации из Яндекс 360 API."""
    users = []
    page = 1
    per_page = 1000
    
    logger.info("Запуск получения пользователей из Яндекс 360 API...")
    
    # Рекомендуемый хост согласно документации Яндекс 360
    base_url = "https://api360.yandex.net"
    
    while True:
        url = f"{base_url}/directory/v1/org/{org_id}/users"
        params = {"page": page, "per_page": per_page}
        headers = {
            "Authorization": f"OAuth {token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30.0)
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Ошибка запроса к Яндекс 360 API (HTTP {response.status_code}): {response.text}"
            )
            
        data = response.json()
        page_users = data.get("users", [])
        if not page_users:
            break
            
        users.extend(page_users)
        
        total_pages = data.get("pages", 1)
        logger.info(f"Загружена страница {page} из {total_pages} (пользователей на странице: {len(page_users)})")
        
        if page >= total_pages:
            break
            
        page += 1
        
    logger.info(f"Всего получено {len(users)} аккаунтов из Яндекс 360.")
    return users

def build_yandex_name_map(yandex_users: List[Dict[str, Any]]) -> Dict[str, str]:
    """Строит маппинг ФИО -> Email из пользователей Яндекс 360."""
    name_map = {}
    for u in yandex_users:
        email = u.get("email")
        if not email:
            continue
            
        # 1. Сопоставление по displayName
        display_name = u.get("displayName")
        if display_name:
            name_map[normalize_name(display_name)] = email
            
        # 2. Сопоставление по составным ФИО (Фамилия + Имя + Отчество)
        name_info = u.get("name") or {}
        first = name_info.get("first") or ""
        last = name_info.get("last") or ""
        middle = name_info.get("middle") or ""
        
        if last and first:
            if middle:
                name_map[normalize_name(f"{last} {first} {middle}")] = email
            name_map[normalize_name(f"{last} {first}")] = email
            
    return name_map

def sync_emails(config=None) -> Dict[str, Any]:
    """Основная функция синхронизации почт. Возвращает статистику."""
    # Загружаем конфигурацию
    if config is None:
        try:
            from src.core.config import Config
            config = Config.from_env()
        except Exception as e:
            return {"success": False, "error": f"Не удалось загрузить конфигурацию: {e}"}
            
    token = config.yandex_360_token
    org_id = config.yandex_360_org_id
    
    if not token or not org_id:
        return {
            "success": False, 
            "error": "Параметры YANDEX_360_TOKEN или YANDEX_360_ORG_ID не заданы в .env"
        }
        
    db_path = config.data_path / "contacts.db"
    if not db_path.exists():
        return {"success": False, "error": f"Файл базы контактов не найден по пути: {db_path}"}
        
    try:
        # 1. Запрос пользователей из API Яндекс 360
        yandex_users = fetch_yandex_users(token, org_id)
        yandex_map = build_yandex_name_map(yandex_users)
        
        # 2. Чтение контактов из SQLite
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        
        # Устанавливаем WAL для безопасности конкурентного доступа
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception as e:
            logger.warning(f"Не удалось установить PRAGMA для SQLite в скрипте синхронизации: {e}")
            
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name, email FROM contacts")
        db_contacts = cursor.fetchall()
        
        # 3. Поиск изменений
        updates = []
        for c in db_contacts:
            full_name = c["full_name"]
            curr_email = c["email"] or ""
            
            normalized = normalize_name(full_name)
            y_email = yandex_map.get(normalized)
            
            # Обновляем, если почта в Яндексе есть и она отличается от текущей в БД
            if y_email and y_email.strip().lower() != curr_email.strip().lower():
                updates.append((y_email.strip().lower(), c["id"]))
                
        # 4. Сохранение изменений в БД
        if updates:
            logger.info(f"Найдено изменений для записи: {len(updates)} контактов.")
            cursor.executemany("UPDATE contacts SET email = ? WHERE id = ?", updates)
            conn.commit()
        else:
            logger.info("Изменений почт не обнаружено.")
            
        conn.close()
        
        return {
            "success": True,
            "total_yandex": len(yandex_users),
            "updated_count": len(updates)
        }
        
    except Exception as e:
        logger.exception("Исключение во время синхронизации почт Яндекс 360")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Локальный CLI запуск
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Включаем логирование библиотеки urllib3/requests
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    res = sync_emails()
    if res["success"]:
        print("\n=== Результат синхронизации ===")
        print(f"Всего пользователей в Яндекс 360: {res['total_yandex']}")
        print(f"Обновлено почт в SQLite: {res['updated_count']}")
    else:
        print(f"\nОшибка синхронизации: {res['error']}")

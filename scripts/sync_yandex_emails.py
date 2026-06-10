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
    
    logger.info(f"Запуск получения пользователей из Яндекс 360 API для организации ID: {org_id}...")
    
    # Рекомендуемый хост согласно документации Яндекс 360
    base_url = "https://api360.yandex.net"
    
    while True:
        url = f"{base_url}/directory/v1/org/{org_id}/users"
        params = {"page": page, "perPage": per_page}
        headers = {
            "Authorization": f"OAuth {token}",
            "Accept": "application/json"
        }
        
        import time
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            response = requests.get(url, headers=headers, params=params, timeout=30.0)
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    logger.warning(f"Яндекс 360 API Rate Limit (HTTP 429). Ждем {retry_delay} сек (попытка {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Экспоненциальная задержка
                    continue
                else:
                    raise RuntimeError(f"Превышен лимит запросов к Яндекс 360 (HTTP 429) после {max_retries} попыток.")
                    
            elif response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка запроса к Яндекс 360 API (HTTP {response.status_code}) для организации {org_id}: {response.text}"
                )
            break
        data = response.json()
        page_users = data.get("users", [])
        if not page_users:
            break
            
        users.extend(page_users)
        
        total_pages = data.get("pages", 1)
        logger.info(f"Организация {org_id}: загружена страница {page} из {total_pages} (пользователей: {len(page_users)})")
        
        if page >= total_pages:
            break
            
        page += 1
        time.sleep(0.5)  # Небольшая пауза между страницами для предотвращения новых лимитов
    logger.info(f"Организация {org_id}: получено {len(users)} аккаунтов.")
    return users

def build_yandex_name_map(yandex_users: List[Dict[str, Any]]) -> Dict[str, str]:
    """Строит маппинг ФИО -> Email из пользователей Яндекс 360."""
    name_map = {}
    
    def add_email(name_key: str, new_email: str):
        if not name_key or not new_email:
            return
        new_email = new_email.strip().lower()
        if name_key in name_map:
            if new_email not in name_map[name_key]:
                name_map[name_key].append(new_email)
        else:
            name_map[name_key] = [new_email]

    for u in yandex_users:
        email = u.get("email")
        if not email:
            continue
            
        # 1. Сопоставление по displayName
        display_name = u.get("displayName")
        if display_name:
            add_email(normalize_name(display_name), email)
            
        # 2. Сопоставление по составным ФИО (Фамилия + Имя + Отчество)
        name_info = u.get("name") or {}
        first = name_info.get("first") or ""
        last = name_info.get("last") or ""
        middle = name_info.get("middle") or ""
        
        if last and first:
            if middle:
                add_email(normalize_name(f"{last} {first} {middle}"), email)
            add_email(normalize_name(f"{last} {first}"), email)
            
    # Склеиваем массивы почт в строку через запятую с сортировкой,
    # чтобы порядок всегда был детерминированным (стабильным)
    result_map = {}
    for k, v in name_map.items():
        result_map[k] = ", ".join(sorted(v))
        
    return result_map

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
    org_id_raw = config.yandex_360_org_id
    
    # Парсинг org_id, поддержка как новых строк, так и запятых
    org_ids = []
    if org_id_raw:
        # Заменяем запятые на переносы строк для единообразной обработки
        normalized_raw = str(org_id_raw).replace(',', '\n')
        for line in normalized_raw.split('\n'):
            line = line.split('#')[0].strip()
            if line:
                org_ids.append(line)
    
    if not token or not org_ids:
        return {
            "success": False, 
            "error": "Параметры YANDEX_360_TOKEN или YANDEX_360_ORG_ID не заданы в .env"
        }
        
    db_path = config.data_path / "contacts.db"
    if not db_path.exists():
        return {"success": False, "error": f"Файл базы контактов не найден по пути: {db_path}"}
        
    try:
        # 1. Запрос пользователей из API Яндекс 360 для всех организаций
        yandex_users = []
        for o_id in org_ids:
            try:
                org_users = fetch_yandex_users(token, o_id)
                yandex_users.extend(org_users)
            except Exception as e:
                logger.error(f"Не удалось получить пользователей для организации {o_id}: {e}")
                raise RuntimeError(f"Сбой при запросе организации {o_id}: {e}")
                
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
            
            # Нормализуем текущие email-ы (сортируем через запятую) для сравнения
            curr_emails_sorted = ", ".join(sorted([e.strip().lower() for e in curr_email.split(",") if e.strip()]))
            
            # Обновляем, если почта в Яндексе есть и она отличается от текущей в БД
            if y_email and y_email != curr_emails_sorted:
                updates.append((y_email, c["id"]))
                
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

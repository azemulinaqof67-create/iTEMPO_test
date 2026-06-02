"""
Веб-панель администратора для управления ботами iTEMPO.
FastAPI backend с REST API и WebSocket для real-time логов.
"""

import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Глобальное состояние (устанавливается при запуске)
_config = None
_assistant = None
_tg_app = None  # Telegram Application для проверки статуса
_max_running = False
_admin_app: Optional[FastAPI] = None

# Непримененные изменения документов
_pending_changes = {
    "to_index": set(),
    "to_delete": set(),
}

# WebSocket подключения для real-time логов
_ws_clients: Set[WebSocket] = set()


def _extract_doc_title(file_path: Path) -> Optional[str]:
    """Извлечь заголовок из .md файла (frontmatter title или # H1)."""
    if file_path.suffix.lower() != ".md":
        return None
    try:
        # Читаем только первые 20 строк для скорости
        lines = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
        
        # Сначала ищем title: в yaml frontmatter
        in_frontmatter = False
        for line in lines:
            if line == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter or line.startswith("title:"):
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                    return title.strip('"').strip("'")

        # Если не нашли, ищем H1 заголовок (# Заголовок)
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
    except Exception:
        pass
    return None


def _count_files_in_dir(directory: Path) -> int:
    """Рекурсивно считает файлы с допустимыми расширениями, исключая файлы index.md."""
    if not directory.exists() or not directory.is_dir():
        return 0
    count = 0
    try:
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
                if p.name != "index.md":
                    count += 1
    except Exception:
        pass
    return count


def create_admin_app(config, assistant=None) -> FastAPI:
    """Создать FastAPI приложение для панели администратора."""
    global _config, _assistant, _admin_app
    _config = config
    _assistant = assistant

    app = FastAPI(title="iTEMPO Admin Panel", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Подключаем статические файлы
    static_path = Path(__file__).parent / "static"
    static_path.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # ─── Auth ─────────────────────────────────────────────────────────────

    _active_sessions = {} # token -> user dict

    async def check_auth(request: Request) -> Optional[Dict]:
        """Возвращает пользователя, если он авторизован, иначе None."""
        token = request.cookies.get("admin_token")
        if token and token in _active_sessions:
            return _active_sessions[token]
        
        # Проверяем Basic Auth (для обратной совместимости/скриптов)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            import base64
            try:
                creds = base64.b64decode(auth[6:]).decode()
                username, pwd = creds.split(":", 1)
                if _assistant and _assistant.chat_history:
                    admin_user = await _assistant.chat_history.get_admin_user_by_username(username)
                    if admin_user:
                        from src.storage.chat_history import verify_password
                        if verify_password(admin_user["password_hash"], pwd):
                            return admin_user
            except Exception:
                pass
        return None

    async def require_auth(request: Request) -> Dict:
        user = await check_auth(request)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user

    def require_permission(permission: str):
        async def dependency(user: Dict = Depends(require_auth)):
            if user["role"] == "superadmin":
                return user
            if permission in user["permissions"]:
                return user
            raise HTTPException(status_code=403, detail="Forbidden")
        return dependency

    async def check_bot_user_company_access(user_id: str, admin_user: Dict):
        if admin_user["role"] == "superadmin" or "all" in admin_user.get("company_ids", []):
            return True
        user_company = await _assistant.chat_history.get_user_company(user_id)
        if user_company in admin_user.get("company_ids", []):
            return True
        raise HTTPException(status_code=403, detail="Нет прав на управление пользователем другой организации")

    # ─── Главная страница ─────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Главная страница — проверяем авторизацию, отдаём SPA."""
        static_dir = Path(__file__).parent / "static"
        html_path = static_dir / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Admin Panel Loading...</h1>")

    @app.post("/api/auth/login")
    async def login(request: Request):
        """Авторизация — возвращает токен сессии."""
        body = await request.json()
        username = body.get("username", "")
        password = body.get("password", "")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Не указаны имя пользователя или пароль")
            
        if not _assistant or not _assistant.chat_history:
            raise HTTPException(status_code=503, detail="База данных недоступна")
            
        admin_user = await _assistant.chat_history.get_admin_user_by_username(username)
        if not admin_user:
            raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")
            
        from src.storage.chat_history import verify_password
        if not verify_password(admin_user["password_hash"], password):
            raise HTTPException(status_code=401, detail="Неверное имя пользователя или пароль")
            
        # Генерируем случайный безопасный токен
        token = secrets.token_hex(32)
        _active_sessions[token] = {
            "id": admin_user["id"],
            "username": admin_user["username"],
            "role": admin_user["role"],
            "company_id": admin_user["company_id"],
            "company_ids": admin_user.get("company_ids", []),
            "permissions": admin_user["permissions"],
        }
        
        response = JSONResponse({
            "success": True, 
            "token": token, 
            "user": {
                "username": admin_user["username"],
                "role": admin_user["role"],
                "company_id": admin_user["company_id"],
                "company_ids": admin_user.get("company_ids", []),
                "permissions": admin_user["permissions"],
            }
        })
        response.set_cookie("admin_token", token, max_age=86400 * 7, httponly=True)
        return response

    @app.post("/api/auth/logout")
    async def logout(request: Request):
        token = request.cookies.get("admin_token")
        if token:
            _active_sessions.pop(token, None)
        response = JSONResponse({"success": True})
        response.delete_cookie("admin_token")
        return response

    @app.get("/api/auth/check")
    async def auth_check(request: Request):
        user = await check_auth(request)
        if user:
            return {
                "authenticated": True, 
                "user": {
                    "username": user["username"],
                    "role": user["role"],
                    "company_id": user["company_id"],
                    "company_ids": user.get("company_ids", []),
                    "permissions": user["permissions"],
                }
            }
        return {"authenticated": False}

    @app.get("/api/companies")
    async def get_companies():
        from src.core.constants import COMPANIES
        return COMPANIES

    # ─── Управление администраторами панели (только superadmin) ───────────

    class AdminUserCreate(BaseModel):
        username: str
        password: str
        role: str
        company_id: Optional[Any] = None
        permissions: List[str]

    class AdminUserUpdate(BaseModel):
        username: str
        password: Optional[str] = None
        role: str
        company_id: Optional[Any] = None
        permissions: List[str]

    @app.get("/api/admin/users")
    async def get_admin_users(user: Dict = Depends(require_auth)):
        if user["role"] != "superadmin":
            raise HTTPException(status_code=403, detail="Forbidden")
        users_list = await _assistant.chat_history.get_all_admin_users()
        return {"users": users_list}

    @app.post("/api/admin/users")
    async def create_admin(body: AdminUserCreate, user: Dict = Depends(require_auth)):
        if user["role"] != "superadmin":
            raise HTTPException(status_code=403, detail="Forbidden")
        
        # Проверяем уникальность логина
        existing = await _assistant.chat_history.get_admin_user_by_username(body.username)
        if existing:
            raise HTTPException(status_code=400, detail="Пользователь с таким именем уже существует")
            
        from src.storage.chat_history import hash_password
        pwd_hash = hash_password(body.password)
        
        comp_id_val = body.company_id
        if isinstance(comp_id_val, list):
            db_company_id = json.dumps(comp_id_val)
        else:
            db_company_id = comp_id_val

        new_id = await _assistant.chat_history.create_admin_user(
            username=body.username,
            password_hash=pwd_hash,
            role=body.role,
            company_id=db_company_id,
            permissions=body.permissions
        )
        return {"success": True, "id": new_id}

    @app.put("/api/admin/users/{admin_id}")
    async def update_admin(admin_id: int, body: AdminUserUpdate, user: Dict = Depends(require_auth)):
        if user["role"] != "superadmin":
            raise HTTPException(status_code=403, detail="Forbidden")
            
        pwd_hash = None
        if body.password:
            from src.storage.chat_history import hash_password
            pwd_hash = hash_password(body.password)
        
        comp_id_val = body.company_id
        if isinstance(comp_id_val, list):
            db_company_id = json.dumps(comp_id_val)
            c_ids = comp_id_val
        elif comp_id_val == "all":
            db_company_id = comp_id_val
            c_ids = ["all"]
        elif comp_id_val:
            db_company_id = comp_id_val
            try:
                c_ids = json.loads(comp_id_val) if comp_id_val.startswith("[") else [comp_id_val]
            except Exception:
                c_ids = [comp_id_val]
        else:
            db_company_id = comp_id_val
            c_ids = []

        await _assistant.chat_history.update_admin_user(
            user_id=admin_id,
            username=body.username,
            password_hash=pwd_hash,
            role=body.role,
            company_id=db_company_id,
            permissions=body.permissions
        )
        
        # Обновляем активные сессии измененного пользователя, если он залогинен
        for token, active_user in list(_active_sessions.items()):
            if active_user["id"] == admin_id:
                active_user["username"] = body.username
                active_user["role"] = body.role
                active_user["company_id"] = db_company_id
                active_user["company_ids"] = c_ids
                active_user["permissions"] = body.permissions
                
        return {"success": True}

    @app.delete("/api/admin/users/{admin_id}")
    async def delete_admin(admin_id: int, user: Dict = Depends(require_auth)):
        if user["role"] != "superadmin":
            raise HTTPException(status_code=403, detail="Forbidden")
            
        # Нельзя удалить самого себя
        if admin_id == user["id"]:
            raise HTTPException(status_code=400, detail="Нельзя удалить собственную учетную запись")
            
        await _assistant.chat_history.delete_admin_user(admin_id)
        
        # Удаляем активные сессии удаленного пользователя
        for token, active_user in list(_active_sessions.items()):
            if active_user["id"] == admin_id:
                _active_sessions.pop(token, None)
                
        return {"success": True}

    # ─── Дашборд / Статистика ─────────────────────────────────────────────

    @app.get("/api/stats")
    async def get_stats(user: Dict = Depends(require_permission("view_stats"))):
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "База данных не подключена"}, status_code=503)
        try:
            company_ids = None if user["role"] == "superadmin" or "all" in user.get("company_ids", []) else user.get("company_ids", [])
            stats = await _assistant.chat_history.get_stats(company_ids=company_ids)
            stats["tg_status"] = "online" if _tg_app and _tg_app.running else "offline"
            stats["max_status"] = "online" if _max_running else "offline"
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    # ─── Пользователи ─────────────────────────────────────────────────────

    @app.get("/api/users")
    async def get_users(
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
        user: Dict = Depends(require_permission("manage_bot_users")),
    ):
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "База данных не подключена"}, status_code=503)
        from src.core.constants import COMPANIES
        try:
            company_ids = None if user["role"] == "superadmin" or "all" in user.get("company_ids", []) else user.get("company_ids", [])
            users = await _assistant.chat_history.get_all_users(limit=limit, offset=offset, company_ids=company_ids)
            total = await _assistant.chat_history.get_users_count(company_ids=company_ids)
            for u in users:
                u["company_name"] = COMPANIES.get(u["company_id"], u["company_id"]) if u["company_id"] else None
            if search:
                users = [u for u in users if search.lower() in u["user_id"].lower()
                         or (u["company_id"] and search.lower() in u["company_id"].lower())]
            return {"users": users, "total": total}
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    class CompanyUpdate(BaseModel):
        company_id: str

    @app.post("/api/users/{user_id}/company")
    async def update_user_company(user_id: str, body: CompanyUpdate, user: Dict = Depends(require_permission("manage_bot_users"))):
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        await check_bot_user_company_access(user_id, user)
        # Ограничение по компании: ограниченный админ не может менять компанию пользователя на чужую
        if user["role"] != "superadmin" and "all" not in user.get("company_ids", []) and body.company_id not in user.get("company_ids", []):
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            await _assistant.chat_history.set_user_company(user_id, body.company_id)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/users/{user_id}/block")
    async def block_user(user_id: str, user: Dict = Depends(require_permission("manage_bot_users"))):
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        await check_bot_user_company_access(user_id, user)
        try:
            await _assistant.chat_history.block_user(user_id)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/users/{user_id}/unblock")
    async def unblock_user(user_id: str, user: Dict = Depends(require_permission("manage_bot_users"))):
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        await check_bot_user_company_access(user_id, user)
        try:
            await _assistant.chat_history.unblock_user(user_id)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.delete("/api/users/{user_id}/history")
    async def clear_user_history(user_id: str, user: Dict = Depends(require_permission("manage_bot_users"))):
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        await check_bot_user_company_access(user_id, user)
        try:
            await _assistant.chat_history.clear_history(user_id, clear_summary=True)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ─── Логи ─────────────────────────────────────────────────────────────

    @app.get("/api/logs")
    async def get_logs(
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        search: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        user: Dict = Depends(require_permission("view_logs")),
    ):
        if not _assistant or not _assistant.chat_history:
            # Fallback — читаем CSV файл (только если суперадмин, иначе для локальных админов CSV не фильтруется)
            if user["role"] != "superadmin" and "all" not in user.get("company_ids", []):
                return JSONResponse({"error": "Чтение логов из CSV недоступно для локальных администраторов"}, status_code=403)
            return _read_csv_logs(limit, offset, search)
        try:
            company_ids = None if user["role"] == "superadmin" or "all" in user.get("company_ids", []) else user.get("company_ids", [])
            logs = await _assistant.chat_history.get_logs(
                limit=limit, offset=offset,
                user_id=user_id, platform=platform,
                search=search, date_from=date_from, date_to=date_to,
                company_ids=company_ids
            )
            total = await _assistant.chat_history.get_logs_count(
                user_id=user_id, platform=platform, search=search,
                company_ids=company_ids
            )
            return {"logs": logs, "total": total}
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    def _read_csv_logs(limit: int, offset: int, search: Optional[str] = None) -> Dict:
        """Fallback: читаем логи из CSV файла."""
        import csv
        csv_path = Path("logs/requests_log.csv")
        if not csv_path.exists():
            return {"logs": [], "total": 0}
        logs = []
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            rows.reverse()
            if search:
                rows = [r for r in rows if search.lower() in r.get("Query", "").lower()
                        or search.lower() in r.get("User_ID", "").lower()]
            total = len(rows)
            for row in rows[offset:offset + limit]:
                logs.append({
                    "session_id": row.get("User_ID", ""),
                    "platform": row.get("Platform", ""),
                    "role": "user",
                    "message": row.get("Query", ""),
                    "timestamp": 0,
                    "metadata": {
                        "response": row.get("Response", ""),
                        "model": row.get("Model", ""),
                        "response_time": row.get("Response_Time_sec", ""),
                    }
                })
        except Exception as e:
            logger.error(f"CSV read error: {e}")
        return {"logs": logs, "total": total if 'total' in locals() else 0}

    @app.get("/api/logs/export")
    async def export_logs_csv(user: Dict = Depends(require_permission("view_logs"))):
        if user["role"] != "superadmin" and user["company_id"] and user["company_id"] != "all":
            raise HTTPException(status_code=403, detail="Экспорт логов недоступен для локальных администраторов")
        csv_path = Path("logs/requests_log.csv")
        if csv_path.exists():
            return Response(
                content=csv_path.read_bytes(),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=requests_log.csv"}
            )
        return JSONResponse({"error": "Файл логов не найден"}, status_code=404)

    # ─── Документы ────────────────────────────────────────────────────────

    @app.get("/api/documents")
    async def get_documents(path: str = "", user: Dict = Depends(require_permission("view_documents"))):
        from src.core.constants import COMPANIES
        data_path = Path(_config.data_path)
        
        CATEGORIES_NAMES = {
            "hr": "👥 Кадры и мотивация (hr)",
            "routine": "📋 Внутренний распорядок и регламенты (routine)",
            "logistics": "🚚 Транспорт и логистика (logistics)",
            "locations": "📍 Расположение и навигация цехов (locations)",
            "it_support": "🔧 ИТ-поддержка пользователей (it_support)",
            "infrastructure": "⚙️ Инфраструктура и ЦОД (infrastructure)",
            "helpdesk": "🎟 Система заявок Helpdesk (helpdesk)",
            "social": "🏖 Социальная сфера и отдых (social)",
            "calendar": "📅 Производственный календарь (calendar)",
            "company": "🏢 О компании и руководство (company)",
            "general": "🔌 Корневой раздел (general)"
        }
        
        # Нормализуем путь
        normalized_path = path.strip().replace("\\", "/").strip("/")
        
        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        is_restricted = not is_super
        
        parts = Path(normalized_path).parts if normalized_path else ()
        
        # Проверка прав доступа
        if normalized_path:
            first_segment = parts[0] if parts else ""
            if is_restricted and first_segment not in user_company_ids:
                raise HTTPException(status_code=403, detail="Нет доступа к этой директории")
                
        target_dir = (data_path / normalized_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            raise HTTPException(status_code=404, detail="Директория не найдена")
            
        # Защита от выхода за пределы data_path
        if not str(target_dir).startswith(str(data_path.resolve())):
            raise HTTPException(status_code=403, detail="Недопустимый путь")
            
        items = []
        
        if not normalized_path:
            # Корневой уровень
            allowed_dirs = set(COMPANIES.keys()) | {"common"}
            if is_restricted:
                allowed_dirs = {d for d in allowed_dirs if d in user_company_ids}
                
            for item in sorted(data_path.iterdir()):
                if item.is_dir():
                    if item.name in allowed_dirs:
                        items.append({
                            "name": COMPANIES.get(item.name, "Общие документы") if item.name != "common" else "Общие документы",
                            "path": item.name,
                            "is_dir": True,
                            "company_name": COMPANIES.get(item.name, "Общие документы") if item.name != "common" else "Общие документы",
                            "files_count": _count_files_in_dir(item)
                        })
                    elif is_super and item.name not in {".chunks_cache"}:
                        items.append({
                            "name": COMPANIES.get(item.name, f"Папка / {item.name}") if item.name != "common" else "Общие документы",
                            "path": item.name,
                            "is_dir": True,
                            "company_name": f"Папка / {item.name}",
                            "files_count": _count_files_in_dir(item)
                        })
                elif item.is_file() and item.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
                    items.append({
                        "name": item.name,
                        "path": item.name,
                        "is_dir": False,
                        "title": _extract_doc_title(item),
                        "company_name": "Все предприятия",
                        "size": item.stat().st_size,
                        "modified": item.stat().st_mtime
                    })
        else:
            # Внутри какой-то директории
            first_segment = parts[0] if parts else ""
            company_name = COMPANIES.get(first_segment, "Общие документы") if first_segment != "common" else "Общие документы"
            
            for item in sorted(target_dir.iterdir()):
                rel_path = str(item.resolve().relative_to(data_path.resolve())).replace("\\", "/")
                if item.is_dir() and item.name not in {".chunks_cache"}:
                    items.append({
                        "name": CATEGORIES_NAMES.get(item.name, item.name),
                        "path": rel_path,
                        "is_dir": True,
                        "company_name": company_name,
                        "files_count": _count_files_in_dir(item)
                    })
                elif item.is_file() and item.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
                    items.append({
                        "name": item.name,
                        "path": rel_path,
                        "is_dir": False,
                        "title": _extract_doc_title(item),
                        "company_name": company_name,
                        "size": item.stat().st_size,
                        "modified": item.stat().st_mtime
                    })
                    
        # Конструируем хлебные крошки
        breadcrumbs = [{"name": "🌍 База знаний", "path": ""}]
        current_accumulated = []
        for part in parts:
            current_accumulated.append(part)
            part_path = "/".join(current_accumulated)
            
            name = part
            if part in COMPANIES:
                name = COMPANIES[part]
            elif part in CATEGORIES_NAMES:
                name = CATEGORIES_NAMES[part]
            elif part == "common":
                name = "Общие документы"
                
            breadcrumbs.append({"name": name, "path": part_path})
            
        return {
            "current_path": normalized_path,
            "breadcrumbs": breadcrumbs,
            "items": items,
            "total": len(items)
        }

    class MkdirRequest(BaseModel):
        path: str

    @app.post("/api/documents/mkdir")
    async def create_directory(body: MkdirRequest, user: Dict = Depends(require_permission("add_documents"))):
        if not body.path:
            return JSONResponse({"error": "Путь не указан"}, status_code=400)
            
        data_path = Path(_config.data_path)
        normalized_path = body.path.strip().replace("\\", "/").strip("/")
        
        parts = Path(normalized_path).parts
        if not parts:
            return JSONResponse({"error": "Недопустимый путь"}, status_code=400)
            
        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        
        first_segment = parts[0]
        if not is_super and first_segment not in user_company_ids:
            raise HTTPException(status_code=403, detail="Нет прав на создание папки в этом разделе")
            
        target_dir = (data_path / normalized_path).resolve()
        if not str(target_dir).startswith(str(data_path.resolve())):
            return JSONResponse({"error": "Недопустимый путь"}, status_code=403)
            
        target_dir.mkdir(parents=True, exist_ok=True)
        return {"success": True, "message": f"Папка создана: {normalized_path}"}
    async def preview_document(
        file: Optional[UploadFile] = File(None),
        text_content: Optional[str] = Form(None),
        company_id: Optional[str] = Form(None),
        company_ids: Optional[str] = Form(None),
        doc_title: Optional[str] = Form(None),
        user: Dict = Depends(require_auth),
    ):
        """ИИ обрабатывает загруженный документ и возвращает MD-предпросмотр."""
        if user["role"] != "superadmin":
            if "add_documents" not in user["permissions"] and "edit_documents" not in user["permissions"]:
                raise HTTPException(status_code=403, detail="Forbidden")
        
        selected_companies = []
        if company_ids:
            selected_companies = [c.strip() for c in company_ids.split(",") if c.strip()]

        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        if not is_super:
            # Проверяем все переданные компании
            for cid in selected_companies:
                if cid == "common":
                    if "common" not in user_company_ids:
                        raise HTTPException(status_code=403, detail="Нет прав на работу с общими документами")
                elif cid not in user_company_ids:
                    raise HTTPException(status_code=403, detail="Нет прав на работу с документами другого предприятия")
            if company_id:
                if company_id == "common":
                    if "common" not in user_company_ids:
                        raise HTTPException(status_code=403, detail="Нет прав на работу с общими документами")
                elif company_id not in user_company_ids:
                    raise HTTPException(status_code=403, detail="Нет прав на работу с документами другого предприятия")
        
        raw_text = ""
        original_filename = ""
        pdf_bytes = None

        if file and file.filename:
            original_filename = file.filename
            content = await file.read()
            ext = Path(file.filename).suffix.lower()
            if ext in {".txt", ".md"}:
                raw_text = content.decode("utf-8", errors="ignore")
            elif ext == ".pdf":
                pdf_bytes = content
                raw_text = _extract_pdf_text(content)
            elif ext == ".docx":
                raw_text = _extract_docx_text(content)
            else:
                return JSONResponse({"error": f"Формат {ext} не поддерживается"}, status_code=400)
        elif text_content:
            raw_text = text_content
            original_filename = "document.md"
        else:
            return JSONResponse({"error": "Нет содержимого"}, status_code=400)

        # Разрешаем пустой raw_text, только если у нас есть PDF-байты для отправки напрямую в ИИ
        if not raw_text.strip() and not pdf_bytes:
            return JSONResponse({"error": "Документ пустой"}, status_code=400)

        # Определяем, какую компанию передать ИИ для YAML frontmatter
        ai_company_id = company_id
        if selected_companies:
            non_common = [c for c in selected_companies if c != "common"]
            ai_company_id = non_common[0] if non_common else None

        # ИИ конвертирует в Markdown формат
        md_content = await _ai_convert_to_markdown(raw_text, doc_title or original_filename, ai_company_id, pdf_bytes=pdf_bytes)

        # Предлагаем имя файла
        safe_name = _safe_filename(doc_title or Path(original_filename).stem)
        suggested_filename = f"{safe_name}.md"

        return {
            "preview": md_content,
            "suggested_filename": suggested_filename,
            "original_filename": original_filename,
            "company_id": ai_company_id,
        }

    @app.post("/api/documents/save")
    async def save_document(request: Request, user: Dict = Depends(require_auth)):
        """Сохранить документ и переиндексировать его в Qdrant."""
        body = await request.json()
        content = body.get("content", "")
        filename = body.get("filename", "document.md")
        
        # Получаем список выбранных компаний (поддерживаем обратную совместимость)
        company_ids = body.get("company_ids")
        target_path = body.get("path")
        
        if company_ids is None:
            # Если не передан список, берем одиночное поле
            company_id = body.get("company_id")
            company_ids = [company_id] if company_id is not None else ["shared"]
            
        # Превращаем 'shared', 'common' и None/пустую строку в None (для общих)
        company_ids_processed = []
        for cid in company_ids:
            if cid in ("shared", "common", "", None):
                company_ids_processed.append(None)
            else:
                company_ids_processed.append(cid)

        # Проверка прав доступа: ограниченный админ может сохранять только в разрешенные ему компании
        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids

        if not content.strip():
            return JSONResponse({"error": "Содержимое пустое"}, status_code=400)

        data_path = Path(_config.data_path)
        saved_paths = []

        # Функция для адаптации YAML frontmatter для конкретного предприятия
        def adapt_frontmatter(md_text: str, target_cid: Optional[str]) -> str:
            import re
            # Ищем блок frontmatter в начале файла
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_text, re.DOTALL)
            if not match:
                # Если frontmatter нет, мы можем добавить его
                fm = f'---\ntitle: "{Path(filename).stem}"\ncompany_id: "{target_cid or "shared"}"\n---\n\n'
                return fm + md_text
                
            fm_content = match.group(1)
            # Ищем company_id в существующем frontmatter
            if re.search(r"^company_id\s*:", fm_content, re.MULTILINE):
                # Заменяем значение
                fm_content_updated = re.sub(
                    r"^company_id\s*:.*$",
                    f'company_id: "{target_cid or "shared"}"',
                    fm_content,
                    flags=re.MULTILINE
                )
            else:
                # Добавляем company_id
                fm_content_updated = fm_content + f'\ncompany_id: "{target_cid or "shared"}"'
                
            return f"---\n{fm_content_updated}\n---\n" + md_text[match.end():]

        if target_path:
            target_path_norm = target_path.strip().replace("\\", "/").strip("/")
            parts = Path(target_path_norm).parts
            first_segment = parts[0] if parts else ""
            
            # Проверка прав доступа к целевой папке
            if not is_super:
                if first_segment in ("shared", "common"):
                    if "common" not in user_company_ids and "shared" not in user_company_ids:
                        raise HTTPException(status_code=403, detail="Нет прав на сохранение в общие документы")
                elif first_segment not in user_company_ids:
                    raise HTTPException(status_code=403, detail="Нет прав на сохранение документов этого предприятия")
                    
            target_dir = data_path / target_path_norm
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Безопасное имя файла
            safe_name = _safe_filename(Path(filename).stem)
            file_path = target_dir / f"{safe_name}.md"
            
            # Адаптируем YAML frontmatter под целевую компанию
            target_company_id = None if first_segment in ("shared", "common") else first_segment
            adapted_content = adapt_frontmatter(content, target_company_id)
            
            file_path.write_text(adapted_content, encoding="utf-8")
            logger.info(f"Document saved: {file_path}")
            
            # Добавляем в список изменений для индексации
            _pending_changes["to_index"].add(str(file_path))
            rel_path = str(file_path.relative_to(data_path)).replace("\\", "/")
            _pending_changes["to_delete"].discard(rel_path)
            saved_paths.append(rel_path)
            
            # Запускаем автоматическое обновление index.md в фоне
            asyncio.create_task(_update_index_file_with_ai(str(file_path), target_company_id))
        else:
            if not is_super:
                for cid in company_ids_processed:
                    if cid is None:
                        if "common" not in user_company_ids and "shared" not in user_company_ids:
                            raise HTTPException(status_code=403, detail="Нет прав на сохранение общих документов")
                    elif cid not in user_company_ids:
                        raise HTTPException(status_code=403, detail="Нет прав на сохранение документов другого предприятия")

            for company_id in company_ids_processed:
                if company_id:
                    target_dir = data_path / company_id
                else:
                    target_dir = data_path / "shared"
                target_dir.mkdir(parents=True, exist_ok=True)

                # Безопасное имя файла
                safe_name = _safe_filename(Path(filename).stem)
                target_file_path = target_dir / f"{safe_name}.md"

                # Адаптируем YAML frontmatter под текущую компанию
                adapted_content = adapt_frontmatter(content, company_id)

                # Записываем файл
                target_file_path.write_text(adapted_content, encoding="utf-8")
                logger.info(f"Document saved: {target_file_path}")

                # Добавляем в список изменений для индексации
                _pending_changes["to_index"].add(str(target_file_path))
                # Убираем из списка удаления на случай, если файл перезаписан
                rel_path = str(target_file_path.relative_to(data_path)).replace("\\", "/")
                _pending_changes["to_delete"].discard(rel_path)
                saved_paths.append(rel_path)

                # Запускаем автоматическое обновление index.md в фоне (на диске)
                asyncio.create_task(_update_index_file_with_ai(str(target_file_path), company_id))

        paths_str = ", ".join(saved_paths)
        return {
            "success": True,
            "paths": saved_paths,
            "message": f"Документ сохранён на диск ({paths_str}). Автоматическое обновление index.md запущено. Для обновления векторной базы нажмите «Применить изменения»."
        }

    # ─── Human-in-the-loop Document Upload / AI Metadata Generation ───

    class MetadataRequest(BaseModel):
        text: str
        draft_title: str
        organization: Optional[str] = None
        category: Optional[str] = None

    class MetadataResponse(BaseModel):
        title: str
        description: str
        file_name: str
        tags: List[str]
        questions_answered: List[str]

    @app.post("/api/generate_metadata", response_model=MetadataResponse)
    async def generate_metadata(body: MetadataRequest, user: Dict = Depends(require_auth)):
        if not _assistant or not _assistant.text_llm:
            raise HTTPException(status_code=503, detail="ИИ-помощник недоступен")
            
        prompt = f"""Ты эксперт по разметке данных для RAG. Проанализируй текст корпоративного документа и его черновое название.
Верни строго JSON объект с 5 ключами:
1. "title": Короткое, официальное название на русском языке.
2. "description": Краткое описание (1-2 предложения).
3. "file_name": Переведи суть документа на английский и сформируй короткое имя файла в snake_case. Добавь расширение .md. Пример: "График отпусков" -> "vacation_schedule.md".
4. "tags": Массив из 5-7 ключевых слов, синонимов или аббревиатур, которые относятся к теме (на русском).
5. "questions_answered": Массив из 2-3 самых популярных вопросов сотрудников, на которые этот текст дает прямой ответ (например: "Как получить ДМС?").

Черновое название: {body.draft_title}
Текст документа:
{body.text}"""

        try:
            # Используем структурированную генерацию
            res = await _assistant.text_llm.generate_structured(
                prompt=prompt,
                response_schema=MetadataResponse,
                temperature=0.0
            )
            if isinstance(res, dict):
                return MetadataResponse(**res)
            return res
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка генерации метаданных: {str(e)}")

    class UploadRequest(BaseModel):
        text: str
        organization: str
        category: str
        title: str
        description: str
        file_name: str
        tags: List[str]
        questions_answered: List[str]
        last_updated: Optional[str] = None

    @app.post("/upload")
    async def upload_document(body: UploadRequest, user: Dict = Depends(require_auth)):
        # Проверка прав доступа
        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids

        if not is_super:
            target_cid = None if body.organization == "shared" else body.organization
            if target_cid is None:
                if "common" not in user_company_ids:
                    raise HTTPException(status_code=403, detail="Нет прав на сохранение общих документов")
            elif target_cid not in user_company_ids:
                raise HTTPException(status_code=403, detail="Нет прав на сохранение документов этого предприятия")

        import re
        import yaml
        from datetime import datetime

        # Санитизация file_name (только a-z, 0-9, _, .md)
        stem = Path(body.file_name).stem.lower()
        sanitized_stem = re.sub(r'[^a-z0-9_]', '', stem)
        if not sanitized_stem:
            sanitized_stem = "document"
        file_name = f"{sanitized_stem}.md"

        last_updated = body.last_updated
        if not last_updated:
            last_updated = datetime.now().strftime("%Y-%m-%d")
        source_file = f"{body.organization}/{body.category}/{file_name}"

        # Формируем YAML Front Matter в строго заданном порядке с красивым форматированием
        ordered_keys = [
            "organization",
            "category",
            "title",
            "description",
            "tags",
            "questions_answered",
            "last_updated",
            "source_file"
        ]
        
        yaml_lines = []
        for key in ordered_keys:
            if key == "organization":
                val = body.organization
            elif key == "category":
                val = body.category
            elif key == "title":
                val = body.title
            elif key == "description":
                val = body.description
            elif key == "tags":
                val = body.tags
            elif key == "questions_answered":
                val = body.questions_answered
            elif key == "last_updated":
                val = last_updated
            elif key == "source_file":
                val = source_file

            if val is None:
                if key in ["tags", "questions_answered"]:
                    val = []
                else:
                    val = ""

            if key == "tags":
                tags_str = ", ".join(f'"{t}"' for t in val)
                yaml_lines.append(f"tags: [{tags_str}]")
            elif key == "questions_answered":
                yaml_lines.append("questions_answered:")
                if not val:
                    yaml_lines[-1] = "questions_answered: []"
                else:
                    for q in val:
                        q_escaped = q.replace('"', '\\"')
                        yaml_lines.append(f'  - "{q_escaped}"')
            else:
                val_escaped = str(val).replace('"', '\\"')
                yaml_lines.append(f'{key}: "{val_escaped}"')

        yaml_str = "\n".join(yaml_lines)
        full_content = f"---\n{yaml_str}\n---\n\n{body.text.strip()}\n"

        data_path = Path(_config.data_path)
        target_dir = data_path / body.organization / body.category
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / file_name

        try:
            file_path.write_text(full_content, encoding="utf-8")
            logger.info(f"Document uploaded/saved: {file_path}")

            # Добавляем в список изменений для индексации
            _pending_changes["to_index"].add(str(file_path))
            _pending_changes["to_delete"].discard(source_file)

            # Запускаем автоматическое обновление index.md в фоне
            asyncio.create_task(_update_index_file_with_ai(str(file_path), body.organization))

            return {
                "success": True,
                "message": f"Документ сохранён как {source_file}. Автоматическое обновление index.md запущено. Для обновления векторной базы примените изменения."
            }
        except Exception as e:
            logger.error(f"Error saving uploaded document: {e}")
            raise HTTPException(status_code=500, detail=f"Не удалось сохранить файл: {str(e)}")


    class MoveDocumentRequest(BaseModel):
        path: str
        company_id: Optional[str] = None

    @app.post("/api/documents/move")
    async def move_document(body: MoveDocumentRequest, user: Dict = Depends(require_permission("edit_documents"))):
        if not body.path:
            return JSONResponse({"error": "Путь не указан"}, status_code=400)

        data_path = Path(_config.data_path)
        source_path = (data_path / body.path).resolve()
        
        # Проверка безопасности пути источника
        if not str(source_path).startswith(str(data_path.resolve())):
            return JSONResponse({"error": "Недопустимый путь"}, status_code=403)

        if not source_path.exists():
            return JSONResponse({"error": "Файл не найден"}, status_code=404)

        # Проверка прав по организации для ограниченных администраторов
        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        if not is_super:
            try:
                rel = source_path.resolve().relative_to(data_path.resolve())
                if not rel.parts:
                    raise HTTPException(status_code=403, detail="Нет доступа к исходному документу")
                first_part = rel.parts[0]
                if first_part in ("shared", "common"):
                    if "common" not in user_company_ids and "shared" not in user_company_ids:
                        raise HTTPException(status_code=403, detail="Нет доступа к исходному документу")
                elif first_part not in user_company_ids:
                    raise HTTPException(status_code=403, detail="Нет доступа к исходному документу")
            except ValueError:
                raise HTTPException(status_code=403, detail="Нет доступа к исходному документу")

            target_cid = body.company_id or "shared"
            if target_cid not in user_company_ids and target_cid != "shared":
                raise HTTPException(status_code=403, detail="Нельзя перемещать документ в эту организацию")

        # Вычисляем относительный путь источника
        old_rel_path = str(source_path.resolve().relative_to(data_path.resolve())).replace("\\", "/")

        # Определяем целевую папку
        if body.company_id:
            target_dir = data_path / body.company_id
        else:
            target_dir = data_path / "shared"

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name

        # Если файл с таким именем уже существует в целевой папке, переименовываем
        if target_path.exists() and target_path.resolve() != source_path.resolve():
            base_name = source_path.stem
            ext = source_path.suffix
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{base_name}_{counter}{ext}"
                counter += 1

        # Если целевой путь совпадает с исходным, ничего не делаем
        if target_path.resolve() == source_path.resolve():
            return {"success": True, "message": "Файл уже находится в целевой папке."}

        try:
            # Переносим файл физически
            import shutil
            shutil.move(str(source_path), str(target_path))
            logger.info(f"Document moved from {source_path} to {target_path}")

            # Добавляем старый путь в список удаления
            _pending_changes["to_delete"].add(old_rel_path)
            # Добавляем новый путь в список индексации
            _pending_changes["to_index"].add(str(target_path))
            
            # Убираем старый абсолютный путь из списка индексации если он там был
            old_abs_path = str(source_path)
            _pending_changes["to_index"].discard(old_abs_path)

            # Запускаем автоматическое обновление index.md для новой папки в фоне
            asyncio.create_task(_update_index_file_with_ai(str(target_path), body.company_id))

            return {
                "success": True,
                "path": str(target_path.relative_to(data_path)).replace("\\", "/"),
                "message": "Документ успешно перенесён. Автоматическое обновление index.md запущено. Для обновления векторной базы нажмите «Применить изменения»."
            }
        except Exception as e:
            logger.error(f"Error moving document: {e}")
            return JSONResponse({"error": f"Не удалось перенести файл: {str(e)}"}, status_code=500)

    @app.delete("/api/documents")
    async def delete_document(request: Request, user: Dict = Depends(require_permission("delete_documents"))):
        body = await request.json()
        doc_path = body.get("path", "")
        if not doc_path:
            return JSONResponse({"error": "Путь не указан"}, status_code=400)

        data_path = Path(_config.data_path)
        full_path = (data_path / doc_path).resolve()
        # Проверка что путь внутри data/
        if not str(full_path).startswith(str(data_path.resolve())):
            return JSONResponse({"error": "Недопустимый путь"}, status_code=403)

        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        if not is_super:
            parts = Path(doc_path).parts
            if not parts:
                raise HTTPException(status_code=403, detail="Нет прав на удаление этого документа")
            first_part = parts[0]
            if first_part == "common":
                if "common" not in user_company_ids:
                    raise HTTPException(status_code=403, detail="Нет прав на удаление этого документа")
            elif first_part not in user_company_ids:
                raise HTTPException(status_code=403, detail="Нет прав на удаление этого документа")

        if not full_path.exists():
            return JSONResponse({"error": "Файл не найден"}, status_code=404)

        full_path.unlink()
        logger.info(f"Document deleted: {full_path}")
        
        # Добавляем в список удаления
        _pending_changes["to_delete"].add(doc_path)
        # Убираем из списка индексации если он там был
        _pending_changes["to_index"].discard(str(full_path))

        return {"success": True, "message": "Документ удалён с диска. Для очистки векторной базы примените изменения."}

    @app.get("/api/documents/content")
    async def get_document_content(path: str, user: Dict = Depends(require_permission("view_documents"))):
        if not path:
            return JSONResponse({"error": "Путь не указан"}, status_code=400)

        data_path = Path(_config.data_path)
        full_path = (data_path / path).resolve()
        # Проверка что путь внутри data/
        if not str(full_path).startswith(str(data_path.resolve())):
            return JSONResponse({"error": "Недопустимый путь"}, status_code=403)

        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        if not is_super:
            parts = Path(path).parts
            if not parts:
                raise HTTPException(status_code=403, detail="Нет доступа к содержимому этого документа")
            first_part = parts[0]
            if first_part == "common":
                if "common" not in user_company_ids:
                    raise HTTPException(status_code=403, detail="Нет доступа к содержимому этого документа")
            elif first_part not in user_company_ids:
                raise HTTPException(status_code=403, detail="Нет доступа к содержимому этого документа")

        if not full_path.exists():
            return JSONResponse({"error": "Файл не найден"}, status_code=404)

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            return {"content": content}
        except Exception as e:
            return JSONResponse({"error": f"Ошибка чтения файла: {str(e)}"}, status_code=500)

    @app.get("/api/documents/pending")
    async def get_pending_changes(user: Dict = Depends(require_auth)):
        has_changes = len(_pending_changes["to_index"]) > 0 or len(_pending_changes["to_delete"]) > 0
        to_index = list(_pending_changes["to_index"])
        to_delete = list(_pending_changes["to_delete"])
        
        data_path = Path(_config.data_path)
        
        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        if not is_super:
            filtered_index = []
            for path in to_index:
                try:
                    rel = Path(path).resolve().relative_to(data_path.resolve())
                    if rel.parts:
                        first_part = rel.parts[0]
                        if first_part == "common" and "common" in user_company_ids:
                            filtered_index.append(path)
                        elif first_part in user_company_ids:
                            filtered_index.append(path)
                except ValueError:
                    pass
            to_index = filtered_index
            
            filtered_delete = []
            for path in to_delete:
                rel = Path(path)
                if rel.parts:
                    first_part = rel.parts[0]
                    if first_part == "common" and "common" in user_company_ids:
                        filtered_delete.append(path)
                    elif first_part in user_company_ids:
                        filtered_delete.append(path)
            to_delete = filtered_delete
            
            has_changes = len(to_index) > 0 or len(to_delete) > 0

        # Превращаем пути to_index из абсолютных в относительные для вывода в интерфейсе
        rel_to_index = []
        for path in to_index:
            try:
                rel = Path(path).resolve().relative_to(data_path.resolve())
                rel_to_index.append(str(rel).replace("\\", "/"))
            except ValueError:
                rel_to_index.append(path)
        to_index = rel_to_index
            
        return {
            "has_changes": has_changes,
            "to_index": to_index,
            "to_delete": to_delete
        }

    @app.post("/api/documents/apply")
    async def apply_documents_changes(user: Dict = Depends(require_permission("apply_changes"))):
        # Находим изменения, относящиеся к компании пользователя
        to_index = []
        to_delete = []
        
        data_path = Path(_config.data_path)
        
        user_company_ids = user.get("company_ids", [])
        is_super = user["role"] == "superadmin" or "all" in user_company_ids
        if is_super:
            # Суперадмин применяет всё
            to_index = list(_pending_changes["to_index"])
            to_delete = list(_pending_changes["to_delete"])
            _pending_changes["to_index"].clear()
            _pending_changes["to_delete"].clear()
        else:
            # Ограниченный админ применяет только свои
            # Фильтруем to_index (абсолютные пути)
            for path in list(_pending_changes["to_index"]):
                try:
                    rel = Path(path).resolve().relative_to(data_path.resolve())
                    if rel.parts:
                        first_part = rel.parts[0]
                        if first_part == "common" and "common" in user_company_ids:
                            to_index.append(path)
                            _pending_changes["to_index"].remove(path)
                        elif first_part in user_company_ids:
                            to_index.append(path)
                            _pending_changes["to_index"].remove(path)
                except ValueError:
                    pass
            
            # Фильтруем to_delete (относительные пути)
            for path in list(_pending_changes["to_delete"]):
                rel = Path(path)
                if rel.parts:
                    first_part = rel.parts[0]
                    if first_part == "common" and "common" in user_company_ids:
                        to_delete.append(path)
                        _pending_changes["to_delete"].remove(path)
                    elif first_part in user_company_ids:
                        to_delete.append(path)
                        _pending_changes["to_delete"].remove(path)
                    
        if not to_index and not to_delete:
            return {"success": True, "message": "Нет изменений для применения."}

        # Запускаем фоновую задачу для переиндексации
        async def run_apply():
            try:
                from src.rag.ingestion.embeddings import EmbeddingService
                embedder = EmbeddingService(_config)

                # 1. Удаляем векторы для удаленных файлов
                if to_delete:
                    logger.info(f"Applying changes: deleting vectors for {to_delete}")
                    await embedder.incremental_update([], target_sources=to_delete)

                # 2. Индексируем новые и измененные файлы
                if to_index:
                    logger.info(f"Applying changes: indexing files {to_index}")
                    for file_path in to_index:
                        await _reindex_document(file_path)
                logger.info("All pending changes applied successfully.")
            except Exception as e:
                logger.error(f"Error applying pending changes: {e}")

        asyncio.create_task(run_apply())

        return {
            "success": True,
            "message": "Применение изменений запущено в фоне. Переиндексация выполняется."
        }

    # ─── Рассылка ─────────────────────────────────────────────────────────

    class BroadcastRequest(BaseModel):
        text: str
        platform: str = "all"      # all, telegram, max
        company_id: Optional[str] = None
        active_days: Optional[int] = None  # None = все когда-либо писавшие

    @app.post("/api/broadcast")
    async def broadcast(body: BroadcastRequest, user: Dict = Depends(require_permission("send_broadcast"))):
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)

        users = await _assistant.chat_history.get_all_users(limit=10000)

        # Фильтрация по дням активности
        if body.active_days:
            cutoff = time.time() - body.active_days * 86400
            users = [u for u in users if u.get("last_activity") and u["last_activity"] > cutoff]

        # Фильтрация по компании
        target_company = body.company_id
        if user["role"] != "superadmin" and user["company_id"] and user["company_id"] != "all":
            target_company = user["company_id"]

        if target_company:
            users = [u for u in users if u.get("company_id") == target_company]

        # Исключаем заблокированных
        users = [u for u in users if not u.get("is_blocked")]

        # Фильтрация по платформе на основе префикса ID
        targeted_users = []
        for u in users:
            user_id = u.get("user_id", "")
            if isinstance(user_id, str) and user_id.startswith("max_"):
                detected_platform = "max"
            else:
                detected_platform = "telegram"

            if body.platform != "all" and body.platform != detected_platform:
                continue

            u["_detected_platform"] = detected_platform
            targeted_users.append(u)

        sent = 0
        failed = 0

        for user in targeted_users:
            user_id = user["user_id"]
            detected_platform = user["_detected_platform"]

            success = False

            if detected_platform == "telegram" and _tg_app:
                try:
                    await _tg_app.bot.send_message(
                        chat_id=int(user_id),
                        text=body.text,
                        parse_mode="HTML"
                    )
                    sent += 1
                    success = True
                    await asyncio.sleep(0.05)
                except Exception as e:
                    logger.error(f"Broadcast: failed to send Telegram message to {user_id}: {e}")

            elif detected_platform == "max" and _config.max_token:
                try:
                    # Очищаем user_id от префикса 'max_'
                    clean_chat_id = user_id
                    if isinstance(clean_chat_id, str) and clean_chat_id.startswith("max_"):
                        clean_chat_id = clean_chat_id[4:]
                    try:
                        clean_chat_id = int(clean_chat_id)
                    except ValueError:
                        pass

                    import httpx
                    headers = {
                        "Authorization": _config.max_token,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"https://platform-api.max.ru/messages?user_id={clean_chat_id}",
                            headers=headers,
                            json={"text": body.text, "format": "html"}
                        )
                        if resp.status_code in (200, 201):
                            sent += 1
                            success = True
                            await asyncio.sleep(0.05)
                        else:
                            logger.error(f"Broadcast: failed to send MAX message to {user_id}, status: {resp.status_code}, response: {resp.text}")
                except Exception as e:
                    logger.error(f"Broadcast: failed to send MAX message to {user_id}: {e}")

            if not success:
                failed += 1

        return {
            "success": True,
            "sent": sent,
            "failed": failed,
            "total_targeted": len(targeted_users),
        }

    # ─── API ключи ────────────────────────────────────────────────────────

    @app.get("/api/keys")
    async def get_api_keys(user: Dict = Depends(require_permission("manage_api_keys"))):
        keys_info = []
        for i, key in enumerate(_config.api_keys):
            masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "****"
            keys_info.append({
                "index": i,
                "masked": masked,
                "is_active": i == 0,  # TODO: интеграция с key manager
            })
        return {"keys": keys_info, "total": len(keys_info)}

    # ─── WebSocket для real-time логов ────────────────────────────────────

    @app.websocket("/ws/logs")
    async def ws_logs(websocket: WebSocket):
        # Проверяем токен из query params
        token = websocket.query_params.get("token", "")
        valid_token = secrets.token_hex(16)  # fallback
        if _config:
            import hashlib
            valid_token = hashlib.sha256(
                f"itempo_admin_{_config.admin_password}".encode()
            ).hexdigest()[:32]

        if token != valid_token:
            await websocket.close(code=4001)
            return

        await websocket.accept()
        _ws_clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            _ws_clients.discard(websocket)

    # ─── Вспомогательные функции ─────────────────────────────────────────

    _admin_app = app
    return app


def set_bot_instances(tg_app=None, max_running: bool = False):
    """Установить ссылки на запущенные боты для отслеживания статуса."""
    global _tg_app, _max_running
    _tg_app = tg_app
    _max_running = max_running


async def broadcast_ws_log(message: Dict):
    """Отправить лог всем подключённым WebSocket клиентам."""
    if not _ws_clients:
        return
    dead = set()
    payload = json.dumps(message, ensure_ascii=False)
    for ws in list(_ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


def _safe_filename(name: str) -> str:
    """Очистить строку для использования как имя файла."""
    import re
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = name.strip(". ").replace(" ", "_")
    return name[:100] or "document"


def _extract_pdf_text(content: bytes) -> str:
    """Извлечь текст из PDF."""
    try:
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception as e:
        logger.error(f"PDF extract error: {e}")
        return ""


def _extract_docx_text(content: bytes) -> str:
    """Извлечь текст из DOCX."""
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX extract error: {e}")
        return ""


async def _ai_convert_to_markdown(raw_text: str, title: str, company_id: Optional[str], pdf_bytes: Optional[bytes] = None) -> str:
    """Использовать ИИ для конвертации документа в Markdown формат."""
    if not _assistant:
        # Простой fallback без ИИ
        return f"# {title}\n\n{raw_text}"

    from src.core.constants import COMPANIES
    company_name = COMPANIES.get(company_id, "Все предприятия") if company_id else "Все предприятия"

    contents = []

    if pdf_bytes:
        from google.genai import types
        pdf_part = types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf",
        )
        contents.append(pdf_part)
        
        prompt = f"""Ты помощник по оформлению корпоративных документов.
Преобразуй прикрепленный PDF документ в структурированный Markdown-документ.

Требования:
- Добавь YAML frontmatter в начале: title, category (тип документа), company_id: "{company_id or 'common'}"
- Используй заголовки (##, ###) для структурирования
- Списки оформляй через - или нумерованно
- Выдели важные данные (даты, числа, имена) **жирным**
- Сохрани ВСЮ исходную информацию без потерь
- Не добавляй информацию, которой нет в исходнике
- Документ для предприятия: {company_name}

Верни ТОЛЬКО готовый Markdown, без пояснений."""
        contents.append(prompt)
    else:
        prompt = f"""Ты помощник по оформлению корпоративных документов.
Преобразуй следующий текст в структурированный Markdown-документ.

Требования:
- Добавь YAML frontmatter в начале: title, category (тип документа), company_id: "{company_id or 'common'}"
- Используй заголовки (##, ###) для структурирования
- Списки оформляй через - или нумерованно
- Выдели важные данные (даты, числа, имена) **жирным**
- Сохрани ВСЮ исходную информацию без потерь
- Не добавляй информацию, которой нет в исходнике
- Документ для предприятия: {company_name}

Исходный текст:
{raw_text[:8000]}

Верни ТОЛЬКО готовый Markdown, без пояснений."""
        contents.append(prompt)

    try:
        from src.core.clients import ClientManager
        client_manager = ClientManager.get_instance(_config)
        model_client = client_manager.get_gemini_client()
        response = await asyncio.to_thread(
            model_client.models.generate_content,
            model=_config.text_model,
            contents=contents,
        )
        return response.text
    except Exception as e:
        logger.error(f"AI convert error: {e}")
        # Fallback: простой MD
        return f"""---
title: "{title}"
company_id: "{company_id or 'common'}"
---

# {title}

{raw_text}
"""


async def _reindex_document(file_path: str):
    """Переиндексировать один файл в Qdrant (фоновая задача)."""
    try:
        logger.info(f"Reindexing document: {file_path}")
        from src.rag.ingestion.document_processor import DocumentProcessor
        from src.rag.ingestion.embeddings import EmbeddingService

        processor = DocumentProcessor(_config)
        file = Path(file_path)
        chunks = processor.prepare_chunks(files=[file])

        if chunks:
            embedder = EmbeddingService(_config)
            rel_path = str(file.relative_to(Path(_config.data_path))).replace("\\", "/")
            await embedder.incremental_update(chunks, target_sources=[rel_path])
            logger.info(f"Reindexed {len(chunks)} chunks from {file_path}")
        else:
            logger.warning(f"No chunks created from {file_path}")
    except Exception as e:
        logger.error(f"Reindex error for {file_path}: {e}")


async def _update_index_file_with_ai(new_doc_path: str, company_id: Optional[str]):
    """Автоматическое обновление index.md при помощи ИИ."""
    try:
        new_path = Path(new_doc_path)
        index_path = None
        
        # Поиск подходящего index.md
        if (new_path.parent / "index.md").exists():
            index_path = new_path.parent / "index.md"
        elif (new_path.parent.parent / "index.md").exists():
            index_path = new_path.parent.parent / "index.md"
        elif company_id:
            root_company = Path(_config.data_path) / company_id / "index.md"
            if root_company.exists():
                index_path = root_company
        else:
            root_common = Path(_config.data_path) / "common" / "index.md"
            if root_common.exists():
                index_path = root_common

        if not index_path:
            logger.info(f"Index file not found for {new_doc_path}, skipping AI index update.")
            return

        if index_path.resolve() == new_path.resolve():
            logger.info("New document is index.md itself, skipping self-update.")
            return

        logger.info(f"AI Index Update: Found index file {index_path} for doc {new_path}")

        index_content = index_path.read_text(encoding="utf-8")
        doc_content = new_path.read_text(encoding="utf-8")

        try:
            rel_doc_link = str(new_path.relative_to(index_path.parent)).replace("\\", "/")
            if rel_doc_link.endswith(".md"):
                rel_doc_link = rel_doc_link[:-3]
        except Exception:
            rel_doc_link = new_path.name[:-3] if new_path.name.endswith(".md") else new_path.name

        title = ""
        for line in doc_content.split("\n"):
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"').strip("'")
                break
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            title = new_path.stem.replace("_", " ")

        prompt = f"""Ты корпоративный ИИ-редактор базы знаний.
Твоя задача — аккуратно обновить существующий навигационный файл `index.md`, добавив ссылку на новый документ.

Существующий `index.md`:
```markdown
{index_content}
```

Новый документ:
- Название: "{title}"
- Путь ссылки: "{rel_doc_link}"
- Краткое содержание нового документа (первые 2000 символов):
```markdown
{doc_content[:2000]}
```

Правила обновления:
1. Добавь новый документ в список "Доступные материалы", "Документы" или аналогичный список ссылок в файле `index.md`.
2. Формат ссылки строго WikiLinks с описанием, например: `- **[[Название документа|{rel_doc_link}]]**: Краткое описание о чем этот документ (сформируй по содержанию).`
3. Сохрани исходный YAML frontmatter в начале `index.md` без изменений, но можешь обновить `last_updated` на текущую дату, если это поле есть (в формате YYYY-MM-DD).
4. Сохрани ВСЕ остальные разделы и ссылки без изменений.
5. Верни ТОЛЬКО обновленный текст `index.md` целиком, без каких-либо твоих комментариев или оберток вроде ```markdown.
"""

        from src.core.clients import ClientManager
        client_manager = ClientManager.get_instance(_config)
        model_client = client_manager.get_gemini_client()
        response = await asyncio.to_thread(
            model_client.models.generate_content,
            model=_config.text_model,
            contents=prompt,
        )
        updated_content = response.text.strip()
        
        # Убираем ```markdown и ``` обертки
        if updated_content.startswith("```markdown"):
            updated_content = updated_content[11:]
        elif updated_content.startswith("```"):
            updated_content = updated_content[3:]
        if updated_content.endswith("```"):
            updated_content = updated_content[:-3]
        updated_content = updated_content.strip()

        if updated_content and len(updated_content) > 50:
            index_path.write_text(updated_content, encoding="utf-8")
            logger.info(f"AI Index Update: index.md updated successfully at {index_path}")
            # Переиндексируем сам index.md
            asyncio.create_task(_reindex_document(str(index_path)))
        else:
            logger.warning("AI Index Update: Model returned empty or too short text, skipping save.")
    except Exception as e:
        logger.error(f"Error during AI index update: {e}")

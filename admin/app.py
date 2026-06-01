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

    def check_auth(request: Request) -> bool:
        """Проверка Basic Auth или сессионного токена."""
        # Проверяем cookie-токен
        token = request.cookies.get("admin_token")
        if token and token == _get_valid_token():
            return True
        # Проверяем Basic Auth
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            import base64
            try:
                creds = base64.b64decode(auth[6:]).decode()
                _, pwd = creds.split(":", 1)
                if pwd == _config.admin_password:
                    return True
            except Exception:
                pass
        return False

    def _get_valid_token() -> str:
        """Получить валидный токен сессии (детерминированный на основе пароля)."""
        import hashlib
        return hashlib.sha256(f"itempo_admin_{_config.admin_password}".encode()).hexdigest()[:32]

    def require_auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Unauthorized")

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
        password = body.get("password", "")
        if password != _config.admin_password:
            raise HTTPException(status_code=401, detail="Неверный пароль")
        token = _get_valid_token()
        response = JSONResponse({"success": True, "token": token})
        response.set_cookie("admin_token", token, max_age=86400 * 7, httponly=True)
        return response

    @app.post("/api/auth/logout")
    async def logout():
        response = JSONResponse({"success": True})
        response.delete_cookie("admin_token")
        return response

    @app.get("/api/auth/check")
    async def auth_check(request: Request):
        return {"authenticated": check_auth(request)}

    # ─── Дашборд / Статистика ─────────────────────────────────────────────

    @app.get("/api/stats")
    async def get_stats(request: Request):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "База данных не подключена"}, status_code=503)
        try:
            stats = await _assistant.chat_history.get_stats()
            # Добавляем статус ботов
            stats["tg_status"] = "online" if _tg_app and _tg_app.running else "offline"
            stats["max_status"] = "online" if _max_running else "offline"
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    # ─── Пользователи ─────────────────────────────────────────────────────

    @app.get("/api/users")
    async def get_users(
        request: Request,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "База данных не подключена"}, status_code=503)
        from src.core.constants import COMPANIES
        try:
            users = await _assistant.chat_history.get_all_users(limit=limit, offset=offset)
            total = await _assistant.chat_history.get_users_count()
            # Добавляем читаемое название компании
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
    async def update_user_company(user_id: str, body: CompanyUpdate, request: Request):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        try:
            await _assistant.chat_history.set_user_company(user_id, body.company_id)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/users/{user_id}/block")
    async def block_user(user_id: str, request: Request):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        try:
            await _assistant.chat_history.block_user(user_id)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/users/{user_id}/unblock")
    async def unblock_user(user_id: str, request: Request):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        try:
            await _assistant.chat_history.unblock_user(user_id)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.delete("/api/users/{user_id}/history")
    async def clear_user_history(user_id: str, request: Request):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)
        try:
            await _assistant.chat_history.clear_history(user_id, clear_summary=True)
            return {"success": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ─── Логи ─────────────────────────────────────────────────────────────

    @app.get("/api/logs")
    async def get_logs(
        request: Request,
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        search: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
    ):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            # Fallback — читаем CSV файл
            return _read_csv_logs(limit, offset, search)
        try:
            logs = await _assistant.chat_history.get_logs(
                limit=limit, offset=offset,
                user_id=user_id, platform=platform,
                search=search, date_from=date_from, date_to=date_to
            )
            total = await _assistant.chat_history.get_logs_count(
                user_id=user_id, platform=platform, search=search
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
    async def export_logs_csv(request: Request):
        require_auth(request)
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
    async def get_documents(request: Request):
        require_auth(request)
        from src.core.constants import COMPANIES
        data_path = Path(_config.data_path)
        docs = []
        # Общие документы
        common_path = data_path / "common"
        if common_path.exists():
            for f in sorted(common_path.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
                    docs.append({
                        "path": str(f.relative_to(data_path)).replace("\\", "/"),
                        "name": f.name,
                        "title": _extract_doc_title(f),
                        "company": None,
                        "company_name": "Все предприятия",
                        "size": f.stat().st_size,
                        "modified": f.stat().st_mtime,
                    })
        # По предприятиям
        for company_id in COMPANIES:
            company_path = data_path / company_id
            if company_path.exists():
                for f in sorted(company_path.rglob("*")):
                    if f.is_file() and f.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
                        docs.append({
                            "path": str(f.relative_to(data_path)).replace("\\", "/"),
                            "name": f.name,
                            "title": _extract_doc_title(f),
                            "company": company_id,
                            "company_name": COMPANIES[company_id],
                            "size": f.stat().st_size,
                            "modified": f.stat().st_mtime,
                        })
        # Остальные папки (01_company и т.д.)
        for item in sorted(data_path.iterdir()):
            if item.is_dir() and item.name not in {*COMPANIES, "common", ".chunks_cache"}:
                for f in sorted(item.rglob("*")):
                    if f.is_file() and f.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
                        docs.append({
                            "path": str(f.relative_to(data_path)).replace("\\", "/"),
                            "name": f.name,
                            "title": _extract_doc_title(f),
                            "company": None,
                            "company_name": f"Общие / {item.name}",
                            "size": f.stat().st_size,
                            "modified": f.stat().st_mtime,
                        })
        return {"documents": docs, "total": len(docs)}

    @app.post("/api/documents/preview")
    async def preview_document(
        request: Request,
        file: Optional[UploadFile] = File(None),
        text_content: Optional[str] = Form(None),
        company_id: Optional[str] = Form(None),
        doc_title: Optional[str] = Form(None),
    ):
        """ИИ обрабатывает загруженный документ и возвращает MD-предпросмотр."""
        require_auth(request)
        raw_text = ""
        original_filename = ""

        if file and file.filename:
            original_filename = file.filename
            content = await file.read()
            ext = Path(file.filename).suffix.lower()
            if ext in {".txt", ".md"}:
                raw_text = content.decode("utf-8", errors="ignore")
            elif ext == ".pdf":
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

        if not raw_text.strip():
            return JSONResponse({"error": "Документ пустой"}, status_code=400)

        # ИИ конвертирует в Markdown формат
        md_content = await _ai_convert_to_markdown(raw_text, doc_title or original_filename, company_id)

        # Предлагаем имя файла
        safe_name = _safe_filename(doc_title or Path(original_filename).stem)
        suggested_filename = f"{safe_name}.md"

        return {
            "preview": md_content,
            "suggested_filename": suggested_filename,
            "original_filename": original_filename,
            "company_id": company_id,
        }

    @app.post("/api/documents/save")
    async def save_document(request: Request):
        """Сохранить документ и переиндексировать его в Qdrant."""
        require_auth(request)
        body = await request.json()
        content = body.get("content", "")
        filename = body.get("filename", "document.md")
        company_id = body.get("company_id")  # None = общий

        if not content.strip():
            return JSONResponse({"error": "Содержимое пустое"}, status_code=400)

        data_path = Path(_config.data_path)
        if company_id:
            target_dir = data_path / company_id
        else:
            target_dir = data_path / "common"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Безопасное имя файла
        safe_name = _safe_filename(Path(filename).stem)
        target_path = target_dir / f"{safe_name}.md"

        # Записываем файл
        target_path.write_text(content, encoding="utf-8")
        logger.info(f"Document saved: {target_path}")

        # Добавляем в список изменений для индексации
        _pending_changes["to_index"].add(str(target_path))
        # Убираем из списка удаления на случай, если файл перезаписан
        rel_path = str(target_path.relative_to(data_path)).replace("\\", "/")
        _pending_changes["to_delete"].discard(rel_path)

        # Запускаем автоматическое обновление index.md в фоне (на диске)
        asyncio.create_task(_update_index_file_with_ai(str(target_path), company_id))

        return {
            "success": True,
            "path": rel_path,
            "message": "Документ сохранён на диск. Автоматическое обновление index.md запущено. Для обновления векторной базы нажмите «Применить изменения»."
        }

    class MoveDocumentRequest(BaseModel):
        path: str
        company_id: Optional[str] = None

    @app.post("/api/documents/move")
    async def move_document(body: MoveDocumentRequest, request: Request):
        require_auth(request)
        if not body.path:
            return JSONResponse({"error": "Путь не указан"}, status_code=400)

        data_path = Path(_config.data_path)
        source_path = (data_path / body.path).resolve()
        
        # Проверка безопасности пути источника
        if not str(source_path).startswith(str(data_path.resolve())):
            return JSONResponse({"error": "Недопустимый путь"}, status_code=403)

        if not source_path.exists():
            return JSONResponse({"error": "Файл не найден"}, status_code=404)

        # Вычисляем относительный путь источника
        old_rel_path = str(source_path.relative_to(data_path)).replace("\\", "/")

        # Определяем целевую папку
        if body.company_id:
            target_dir = data_path / body.company_id
        else:
            target_dir = data_path / "common"

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
    async def delete_document(request: Request):
        require_auth(request)
        body = await request.json()
        doc_path = body.get("path", "")
        if not doc_path:
            return JSONResponse({"error": "Путь не указан"}, status_code=400)

        data_path = Path(_config.data_path)
        full_path = (data_path / doc_path).resolve()
        # Проверка что путь внутри data/
        if not str(full_path).startswith(str(data_path.resolve())):
            return JSONResponse({"error": "Недопустимый путь"}, status_code=403)

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
    async def get_document_content(request: Request, path: str):
        require_auth(request)
        if not path:
            return JSONResponse({"error": "Путь не указан"}, status_code=400)

        data_path = Path(_config.data_path)
        full_path = (data_path / path).resolve()
        # Проверка что путь внутри data/
        if not str(full_path).startswith(str(data_path.resolve())):
            return JSONResponse({"error": "Недопустимый путь"}, status_code=403)

        if not full_path.exists():
            return JSONResponse({"error": "Файл не найден"}, status_code=404)

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            return {"content": content}
        except Exception as e:
            return JSONResponse({"error": f"Ошибка чтения файла: {str(e)}"}, status_code=500)

    @app.get("/api/documents/pending")
    async def get_pending_changes(request: Request):
        require_auth(request)
        has_changes = len(_pending_changes["to_index"]) > 0 or len(_pending_changes["to_delete"]) > 0
        return {
            "has_changes": has_changes,
            "to_index": list(_pending_changes["to_index"]),
            "to_delete": list(_pending_changes["to_delete"])
        }

    @app.post("/api/documents/apply")
    async def apply_documents_changes(request: Request):
        require_auth(request)
        if not _pending_changes["to_index"] and not _pending_changes["to_delete"]:
            return {"success": True, "message": "Нет изменений для применения."}

        # Копируем списки для фоновой задачи
        to_index = list(_pending_changes["to_index"])
        to_delete = list(_pending_changes["to_delete"])

        # Очищаем глобальные списки
        _pending_changes["to_index"].clear()
        _pending_changes["to_delete"].clear()

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
    async def broadcast(body: BroadcastRequest, request: Request):
        require_auth(request)
        if not _assistant or not _assistant.chat_history:
            return JSONResponse({"error": "БД недоступна"}, status_code=503)

        users = await _assistant.chat_history.get_all_users(limit=10000)

        # Фильтрация по дням активности
        if body.active_days:
            cutoff = time.time() - body.active_days * 86400
            users = [u for u in users if u.get("last_activity") and u["last_activity"] > cutoff]

        # Фильтрация по компании
        if body.company_id:
            users = [u for u in users if u.get("company_id") == body.company_id]

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
                            f"https://platform-api.max.ru/messages?chat_id={clean_chat_id}",
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
    async def get_api_keys(request: Request):
        require_auth(request)
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


async def _ai_convert_to_markdown(raw_text: str, title: str, company_id: Optional[str]) -> str:
    """Использовать ИИ для конвертации документа в Markdown формат."""
    if not _assistant:
        # Простой fallback без ИИ
        return f"# {title}\n\n{raw_text}"

    from src.core.constants import COMPANIES
    company_name = COMPANIES.get(company_id, "Все предприятия") if company_id else "Все предприятия"

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

    try:
        from src.core.clients import ClientManager
        client_manager = ClientManager.get_instance(_config)
        model_client = client_manager.get_gemini_client()
        response = model_client.models.generate_content(
            model=_config.text_model,
            contents=prompt,
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
        response = model_client.models.generate_content(
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

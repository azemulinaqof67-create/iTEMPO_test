"""
Автоматическая отправка документов на основе анализа вопроса.
Работает независимо от промптов.
"""

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DocumentRule:
    """Правило отправки документа."""

    keywords: List[str]  # Ключевые слова для триггера
    document_path: str  # Путь к файлу
    description: str  # Описание для подписи
    file_type: str = "auto"  # pdf, docx, image, auto


class DocumentSender:
    """Автоматически определяет, когда нужно отправить документ."""

    def __init__(self):
        self.rules = {
            "календарь": DocumentRule(
                keywords=["календарь", "рабочие дни", "выходные", "праздники", "график работы", "сокращенные дни"],
                document_path="data/07_kalendar/_2025-12-02_13564810.png",
                description="Производственный календарь 2026",
                file_type="image",
            ),
            "дресс-код": DocumentRule(
                keywords=["дресс-код", "одежда", "внешний вид", "форма"],
                document_path="data/03_routine/dress_code.md",
                description="Регламент по дресс-коду",
                file_type="other",
            ),
            "автобусы": DocumentRule(
                keywords=["вахта", "автобус", "развозка", "расписание автобусов", "как добраться"],
                document_path="data/04_logistics/shuttle_bus/index.md",
                description="Расписание вахтовых автобусов",
                file_type="other",
            ),
        }

    def find_documents(self, question: str) -> List[DocumentRule]:
        """
        Находит документы, которые нужно отправить по вопросу.

        Args:
            question: Текст вопроса пользователя

        Returns:
            Список правил документов для отправки
        """
        question_lower = question.lower()
        found_docs = []

        for rule_name, rule in self.rules.items():
            for keyword in rule.keywords:
                if keyword in question_lower:
                    found_docs.append(rule)
                    break  # Нашли совпадение, переходим к следующему правилу

        return found_docs

    def add_rule(self, name: str, keywords: List[str], document_path: str, description: str, file_type: str = "auto"):
        """
        Добавляет новое правило отправки документа.

        Args:
            name: Уникальное имя правила
            keywords: Список ключевых слов для триггера
            document_path: Путь к файлу документа
            description: Описание документа для подписи
            file_type: Тип файла (pdf, docx, image, auto)
        """
        self.rules[name] = DocumentRule(
            keywords=keywords, document_path=document_path, description=description, file_type=file_type
        )

    def remove_rule(self, name: str) -> bool:
        """
        Удаляет правило по имени.

        Args:
            name: Имя правила для удаления

        Returns:
            True если правило было удалено, False если не найдено
        """
        if name in self.rules:
            del self.rules[name]
            return True
        return False

    def get_all_rules(self) -> Dict[str, DocumentRule]:
        """Возвращает все правила."""
        return self.rules.copy()

    def document_exists(self, document_path: str) -> bool:
        """
        Проверяет существование документа.

        Args:
            document_path: Путь к документу

        Returns:
            True если файл существует, False если нет
        """
        # Проверяем по нескольким возможным корням (Windows/Linux)
        project_root = os.getcwd()
        full_path = os.path.join(project_root, document_path.replace("/", os.sep))
        if os.path.exists(full_path):
            return True

        # Резервный путь (если запуск из подпапки)
        alt_path = os.path.abspath(os.path.join(project_root, "..", document_path.replace("/", os.sep)))
        if os.path.exists(alt_path):
            return True

        return False

    def get_file_type(self, file_path: str) -> str:
        """
        Определяет тип файла по расширению.

        Args:
            file_path: Путь к файлу

        Returns:
            Тип файла (image, pdf, docx, other)
        """
        extension = os.path.splitext(file_path)[1].lower()

        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
        pdf_extensions = [".pdf"]
        docx_extensions = [".docx", ".doc"]

        if extension in image_extensions:
            return "image"
        elif extension in pdf_extensions:
            return "pdf"
        elif extension in docx_extensions:
            return "docx"
        else:
            return "other"

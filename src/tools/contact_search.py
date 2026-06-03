import aiosqlite
import logging
from typing import Optional, List, Dict, Any
from rapidfuzz import process, fuzz, utils
from src.utils.company_mapper import get_company_keywords

logger = logging.getLogger(__name__)

class ContactSearchTool:
    def __init__(self, db_path: str = "data/contacts.db"):
        self.db_path = db_path

    async def search(self, target_person: str, target_company: Optional[str] = None) -> str:
        """
        Поиск контактов в SQLite с использованием нечеткого сравнения (Fuzzy Matching).
        """
        if not target_person:
            return "Не указано имя или должность для поиска."

        # Логирование перед поиском
        logger.info(f"[SQL SEARCH] Intent: contact_search | Person: {target_person} | Company: {target_company}")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # 1. Извлекаем ВСЕХ контакты (база небольшая, это безопасно)
                query = "SELECT * FROM contacts"
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
                
                if not rows:
                    return "База контактов пуста."

                # Подготавливаем слова для бонус-фильтра по компании
                target_comp_words = []
                if target_company:
                    mapped_keywords = get_company_keywords(target_company)
                    if mapped_keywords:
                        target_comp_words = mapped_keywords
                    else:
                        target_comp_words = [w.strip().lower() for w in target_company.split() if len(w.strip()) > 2]

                # 2. Нечеткий поиск
                results_with_scores = []
                for row in rows:
                    name_score = fuzz.WRatio(target_person, row['full_name'], processor=utils.default_process)
                    name_partial_score = fuzz.partial_ratio(target_person, row['full_name'], processor=utils.default_process)
                    
                    scores = [
                        name_score,
                        fuzz.WRatio(target_person, row['position'], processor=utils.default_process),
                        fuzz.WRatio(target_person, row['department'], processor=utils.default_process),
                        name_partial_score,
                        fuzz.partial_ratio(target_person, row['position'], processor=utils.default_process),
                        fuzz.partial_ratio(target_person, row['department'], processor=utils.default_process)
                    ]
                    
                    # Проверка совпадения номера телефона
                    # Требуем минимум 4 цифры и только точное совпадение, чтобы не срабатывать
                    # на трёхзначные подстроки в именах/фразах (например, '100 сотрудников' → '4100').
                    phone_score = 0
                    target_digits = "".join([c for c in target_person if c.isdigit()])
                    phone_digits = "".join([c for c in (row['phone'] or "") if c.isdigit()])
                    if len(target_digits) >= 4 and phone_digits and target_digits == phone_digits:
                        phone_score = 100

                    max_score = max(max(scores), phone_score)
                    
                    if max_score >= 70: # Порог вхождения
                        # СТРОГИЙ ФИЛЬТР ПО КОМПАНИИ
                        # Если совпадение по имени очень высокое (>90), игнорируем фильтр компании
                        if target_comp_words and not (name_score >= 90 or name_partial_score >= 90):
                            row_comp = (row['company'] or "").lower()
                            
                            # Проверяем пересечение слов
                            if not any(w in row_comp for w in target_comp_words):
                                continue
                                
                            # Спец. защита от путаницы Технотрон и Технотрон-Метиз
                            if "метиз" in row_comp and "метиз" not in target_comp_words:
                                continue # Искали Технотрон, а попали на Метиз
                            if "метиз" in target_comp_words and "метиз" not in row_comp:
                                continue # Искали Метиз, а попали на обычный Технотрон
                                
                        results_with_scores.append((max_score, row))

                # Сортируем по убыванию скора
                results_with_scores.sort(key=lambda x: x[0], reverse=True)
                
                # Умная обрезка результатов, чтобы LLM не путалась в похожих фамилиях
                top_results = []
                if results_with_scores:
                    best_score = results_with_scores[0][0]
                    # Если есть явный лидер с высокой уверенностью
                    if best_score >= 90:
                        # Берем только тех, кто отстает от лидера не более чем на 5 баллов
                        top_results = [r for r in results_with_scores if best_score - r[0] <= 5][:3]
                    else:
                        # Иначе берем стандартный топ-3
                        top_results = results_with_scores[:3]

                formatted_results = []
                for score, row in top_results:
                    logger.info(f"[SQL SEARCH] Match found: {row['full_name']} | Score: {score}")
                    formatted_results.append(
                        f"{len(formatted_results) + 1}. {row['full_name']} — {row['position']}\n"
                        f"   Отдел: {row['department']}, Компания: {row['company']}\n"
                        f"   Тел: {row['phone']}"
                    )

                if not formatted_results:
                    return f"По запросу '{target_person}' ничего не найдено."

                return "Найдены контакты:\n" + "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"ContactSearchTool error: {e}")
            return "Произошла ошибка при поиске в базе контактов."

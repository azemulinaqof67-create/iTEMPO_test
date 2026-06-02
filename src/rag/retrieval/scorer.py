import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Словарь для нормализации аббревиатур при поиске имен файлов
ACRONYM_MAP = {
    "абк": "abk", 
    "итз": "itz", 
    "кмк": "kmk", 
    "тэмпо": "tempo", 
    "цхп": "chp",
    "нтз": "ntz",
    "кзмк": "kzmk",
    "зтэо": "zteo",
    "птфк": "technotron",
    "технотрон": "technotron"
}


class BusinessLogicScorer:
    """Класс для начисления бизнес-бонусов к результатам поиска."""

    def apply_bonuses(self, results: List[Dict], query: str, company_id: Optional[str] = None) -> List[Dict]:
        """
        Начисляет бонусы к результатам поиска и сортирует их по итоговому score.
        """
        query_lower = query.lower()
        
        # 1. Определение целевого предприятия из запроса
        target_company_id = company_id
        if company_id:
            company_keywords = {
                "technotron": ["технотрон", "птфк", "тихнотрон"],
                "metiz": ["метиз", "саморез", "технотрон-метиз"],
                "ntz": ["нтз"],
                "itz": ["итз"],
                "kmk": ["кмк"],
                "kzmk": ["кзмк"],
                "zteo": ["зтэо"],
                "td": ["тд", "торговый дом"],
                "it": ["айти", "it"],
                "sks": ["скс"],
                "port": ["порт"]
            }
            for cid, keywords in company_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    target_company_id = cid
                    break

        scored_results = []
        for doc in results:
            # Создаем копию документа, чтобы не мутировать исходные данные
            doc_copy = dict(doc)
            
            # В Qdrant payload может быть вложен в "payload" или находиться прямо в doc.
            # Наш _vector_search возвращает плоский словарь со всеми необходимыми полями, 
            # но на всякий случай поддержим оба варианта.
            payload = doc_copy.get("payload") if isinstance(doc_copy.get("payload"), dict) else doc_copy
            
            source = payload.get("source", "").lower().replace("\\", "/")
            doc_text = (payload.get("original_text") or payload.get("text", "")).lower()
            doc_type = payload.get("doc_type", "")
            company_tag = payload.get("company_tag", "")
            filename_clean = payload.get("filename_clean", "")
            
            bonus = 0.0
            if not source:
                continue

            # Инициализация system_hints в метаданных документа
            metadata = dict(payload.get("metadata", {}))
            system_hints = list(metadata.get("system_hints", []))
            metadata["system_hints"] = system_hints
            
            # Обновляем метаданные в копии документа
            if "metadata" in doc_copy:
                doc_copy["metadata"] = metadata
            elif "payload" in doc_copy and isinstance(doc_copy["payload"], dict):
                doc_copy["payload"] = dict(doc_copy["payload"])
                doc_copy["payload"]["metadata"] = metadata
            else:
                doc_copy["metadata"] = metadata
                
            # ── 1. СМАРТ-ПОИСК ЧИСЕЛ (Без использования регулярных выражений) ──
            # Извлекаем числа длиной до 4 знаков из поискового запроса
            query_tokens = query_lower.split()
            numbers_in_query = [t for t in query_tokens if t.isdigit() and len(t) <= 4]
            has_number_bonus = False
            
            if numbers_in_query:
                doc_words = doc_text.split()
                source_words = source.split()
                
                context_words = {"абк", "abk", "кабинет", "цех", "этаж", "офис", "№", "номер", "корпус", "блок"}
                
                for num in numbers_in_query:
                    # Вспомогательная функция проверки контекста
                    def check_context(words_list) -> bool:
                        for idx, word in enumerate(words_list):
                            clean_word = "".join(c for c in word if c.isalnum() or c == "№")
                            if clean_word == num:
                                start = max(0, idx - 2)
                                end = min(len(words_list), idx + 3)
                                for j in range(start, end):
                                    if j != idx:
                                        neighbor = "".join(c for c in words_list[j] if c.isalnum() or c == "№").lower()
                                        if neighbor in context_words:
                                            return True
                        return False
                    
                    if check_context(doc_words) or check_context(source_words):
                        bonus += 0.15
                        logger.debug(f"🚀 SMART NUMBER BONUS +0.15 for '{num}' in {source}")
                        has_number_bonus = True

            if has_number_bonus:
                system_hints.append("Содержит искомый номер/кабинет")

            # ── 2. БУСТ ПО ИМЕНИ ФАЙЛА (Без использования регулярных выражений) ──
            if filename_clean:
                filename = filename_clean.lower()
                normalized_query = query_lower
                for cyr, lat in ACRONYM_MAP.items():
                    normalized_query = normalized_query.replace(cyr, lat)
                
                clean_query = "".join(c for c in normalized_query if c.isalnum())
                clean_filename = "".join(c for c in filename if c.isalnum())
                
                if clean_filename and (clean_filename in clean_query or clean_query in clean_filename):
                    bonus += 0.3
                    logger.info(f"🎯 FILENAME MATCH BONUS +0.3 for {source}")
                    system_hints.append("Точное совпадение имени файла")

            # ── 3. ТЕМАТИЧЕСКИЕ УРОВНИ (Tiers) ──
            location_keywords = ["где", "адрес", "найти", "локация", "местоположение", "добраться", "пройти", "проехать", "карта"]
            is_location_query = any(lkw in query_lower for lkw in location_keywords)
            hr_keywords = ["работа", "трудоустройство", "вакансия", "прием", "найм", "увольнение", "отпуск", "больничный", "кадры"]
            is_hr_query = any(hkw in query_lower for hkw in hr_keywords)

            # Уровень 0: Локации
            if doc_type == "company_location" and is_location_query:
                bonus += 0.5
            
            # Уровень 0.5: HR
            elif doc_type == "hr_policy" and is_hr_query:
                bonus += 0.25

            # Уровень 0.7: Совпадение с questions_answered (если есть в метаданных)
            q_answered = payload.get("questions_answered", [])
            if q_answered and any(query_lower in str(q).lower() or str(q).lower() in query_lower for q in q_answered):
                bonus += 0.3
                logger.info(f"🎯 MATCH BONUS: Query matches 'questions_answered' in {source}")
                
            # Уровень 1: Предприятие
            is_target_company = False
            if target_company_id and company_tag:
                if company_tag.lower() == target_company_id.lower():
                    is_target_company = True
                else:
                    # Резервная нормализация без регулярок
                    clean_target = "".join(c for c in target_company_id.lower() if c.isalnum())
                    clean_tag = "".join(c for c in company_tag.lower() if c.isalnum())
                    is_target_company = (clean_target == clean_tag) or (clean_target in clean_tag and len(clean_target) > 2) or (clean_tag in clean_target and len(clean_tag) > 2)
            
            if is_target_company:
                bonus += 0.5
                
            doc_copy["score"] = doc_copy.get("score", 0.0) + bonus
            scored_results.append(doc_copy)

        # Сортируем результаты по итоговому score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results

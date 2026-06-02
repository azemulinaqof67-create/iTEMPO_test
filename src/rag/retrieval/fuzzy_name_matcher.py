"""
Нечёткое сопоставление имён/фамилий для исправления ошибок голосового
распознавания (STT) в поисковых запросах.

Принцип работы:
  1. При построении BM25-индекса модуль собирает все имена и фамилии из
     корпуса, используя морфологический анализатор pymorphy3 (теги Name,
     Patr, Surn) + эвристику (слово с заглавной буквы длиннее 3 букв).
  2. При поступлении поискового запроса каждый токен сравнивается с
     известными именами через rapidfuzz (WRatio).
  3. Если совпадение найдено выше порога — токен заменяется правильным
     вариантом ИЗ КОРПУСА, и поиск выполняется уже с исправленным запросом.
  4. Дополнительно: если WRatio не нашёл кандидата, применяется сравнение
     «скелетов» согласных букв (более устойчиво к гласным ошибкам STT).
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process as fuzz_process

    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz не установлен, FuzzyNameMatcher будет отключён. Установите: uv add rapidfuzz")

try:
    import pymorphy3 as _pymorphy3

    _MORPH = _pymorphy3.MorphAnalyzer()
    _PYMORPHY_AVAILABLE = True
except Exception:
    _MORPH = None
    _PYMORPHY_AVAILABLE = False

# ─── Пороги ────────────────────────────────────────────────────────────────────

# Минимальный WRatio-скор для принятия fuzzy-совпадения (0..100)
_MIN_WRATIO = 85.0

# Минимальный скор при сравнении согласных скелетов (менее строгий порог)
_MIN_CONSONANT_RATIO = 85.0

# Минимальная длина токена, который вообще рассматривается как возможное имя
_MIN_TOKEN_LEN = 4

# ─── Стоп-слова: общие слова, которые не являются именами ──────────────────────

_STOP_WORDS: Set[str] = {
    "телефон", "номер", "контакт", "сотрудник", "работник", "начальник",
    "директор", "менеджер", "специалист", "отдел", "цех", "кабинет",
    "дать", "найти", "показать", "сказать", "позвонить", "написать",
    "какой", "какая", "какое", "какие", "чей", "чья", "чьё",
    "нужно", "нужна", "нужен", "прошу", "позвони", "напиши",
    "руководитель", "заместитель", "главный", "старший", "ведущий",
    "инженер", "бухгалтер", "экономист", "технолог", "мастер",
    "завод", "предприятие", "компания", "холдинг", "тэмпо", "темпо",
    "травма", "инцидент", "порядок", "действие", "получение", "помощь", "пожар", "охрана", "сб",
    "адрес", "место", "расположение", "карта", "маршрут", "проезд", "документ", "список", "перечень",
    "кадр", "кадры", "кадров", "прием", "трудоустройство", "работа", "вакансия"
}

# Русские согласные (строчные)
_CONSONANTS: frozenset = frozenset("бвгджзйклмнпрстфхцчшщ")


def _consonant_skeleton(word: str) -> str:
    """Возвращает строку только из согласных букв — устойчив к гласным ошибкам."""
    return "".join(ch for ch in word.lower() if ch in _CONSONANTS)


def _is_name_by_morph(word: str) -> bool:
    """
    Проверяет, является ли слово именем/фамилией/отчеством по pymorphy3.
    Теги: Name (имя), Patr (отчество), Surn (фамилия).
    """
    if not _PYMORPHY_AVAILABLE or _MORPH is None:
        return False
    parsed = _MORPH.parse(word.lower())
    if not parsed:
        return False
    grammemes = parsed[0].tag.grammemes
    return bool({"Name", "Patr", "Surn"} & grammemes)


def _lemmatize(word: str) -> str:
    """Возвращает нормальную форму слова через pymorphy3."""
    if not _PYMORPHY_AVAILABLE or _MORPH is None:
        return word.lower()
    parsed = _MORPH.parse(word.lower())
    if not parsed:
        return word.lower()
    return parsed[0].normal_form


class FuzzyNameMatcher:
    """
    Нечёткий корректор имён в поисковых запросах.

    Жизненный цикл:
      • rebuild(corpus_texts)   — вызывается после построения BM25 индекса
      • correct_query(query)    — вызывается перед каждым поиском
    """

    def __init__(self):
        # Нормальные формы имён из корпуса (для поиска rapidfuzz)
        self._name_list: List[str] = []
        # Словарь: нормальная форма → оригинальный вид из текста
        self._lemma_to_original: Dict[str, str] = {}
        # Словарь: согласный скелет → список нормальных форм
        self._skeleton_index: Dict[str, List[str]] = {}

    # ── Публичный API ───────────────────────────────────────────────────────────

    def rebuild(self, corpus_texts: List[str]) -> None:
        """
        Строит словарь имён и фамилий из текстов корпуса.
        Вызывается ОДИН РАЗ после построения BM25 (thread-safe: уже в asyncio.to_thread).
        """
        if not _RAPIDFUZZ_AVAILABLE:
            return

        collected: Set[str] = set()

        for text in corpus_texts:
            # Ищем слова с заглавной буквы (потенциальные имена)
            for raw_word in re.findall(r"[А-ЯЁ][а-яё]{2,}", text):
                word_lower = raw_word.lower()

                # Фильтр стоп-слов
                if word_lower in _STOP_WORDS:
                    continue
                # Слишком короткое
                if len(word_lower) < _MIN_TOKEN_LEN:
                    continue

                # Основная проверка через pymorphy3
                if _PYMORPHY_AVAILABLE:
                    if _is_name_by_morph(word_lower):
                        lemma = _lemmatize(word_lower)
                        collected.add(lemma)
                        # Сохраняем «красивый» вид (с заглавной)
                        if lemma not in self._lemma_to_original:
                            self._lemma_to_original[lemma] = raw_word
                else:
                    # Без pymorphy3: берём всё с заглавной буквы (грубая эвристика)
                    lemma = word_lower
                    collected.add(lemma)
                    if lemma not in self._lemma_to_original:
                        self._lemma_to_original[lemma] = raw_word

        self._name_list = sorted(collected)

        # Строим индекс согласных скелетов
        self._skeleton_index = {}
        for name in self._name_list:
            sk = _consonant_skeleton(name)
            if sk:
                self._skeleton_index.setdefault(sk, []).append(name)

        logger.info(
            "🔤 FuzzyNameMatcher: построен словарь из %d уникальных имён/фамилий",
            len(self._name_list),
        )

    def correct_query(self, query: str) -> Tuple[str, bool]:
        """
        Пробует исправить имена/фамилии в запросе через нечёткое сопоставление.

        Returns:
            (corrected_query, was_corrected) — исправленный запрос и флаг изменения.
            Если исправлять нечего — возвращает оригинал и False.
        """
        if not _RAPIDFUZZ_AVAILABLE or not self._name_list:
            return query, False

        tokens = re.findall(r"\w+", query.lower())
        replacements: Dict[str, str] = {}

        for token in tokens:
            if len(token) < _MIN_TOKEN_LEN:
                continue
            if token in _STOP_WORDS:
                continue

            # ЗАЩИТА: Если слово является глаголом, предлогом или ОБЫЧНЫМ существительным (не именем) — не трогаем
            if _PYMORPHY_AVAILABLE and _MORPH:
                parsed = _MORPH.parse(token)
                if parsed:
                    tag = parsed[0].tag
                    pos = tag.POS
                    # Если это глагол, предлог и т.д. - пропускаем
                    if pos in {"VERB", "INFN", "PREP", "CONJ", "PRCL", "INTJ", "ADVB"}:
                        continue
                    # Если это существительное (NOUN), но НЕ Имя/Фамилия/Отчество - тоже пропускаем
                    if pos == "NOUN" and not ({"Name", "Surn", "Patr"} & tag.grammemes):
                        continue
            lemma = _lemmatize(token)
            if lemma in self._name_list:
                continue

            # ── Метод 1: WRatio по всему словарю имён ─────────────────────────
            result = fuzz_process.extractOne(
                lemma,
                self._name_list,
                scorer=fuzz.WRatio,
                score_cutoff=_MIN_WRATIO,
            )
            if result:
                matched_name, score, _ = result
                replacements[token] = matched_name
                logger.info(
                    "🔤 Fuzzy[WRatio]: '%s' (лемма='%s') → '%s' (score=%.1f)",
                    token, lemma, matched_name, score,
                )
                continue

            # ── Метод 2: сравнение согласных скелетов ─────────────────────────
            token_sk = _consonant_skeleton(token)
            if len(token_sk) < 3:
                continue

            # Ищем похожие скелеты в индексе
            skeleton_candidates: List[str] = []
            for known_sk, names in self._skeleton_index.items():
                sk_score = fuzz.ratio(token_sk, known_sk)
                if sk_score >= _MIN_CONSONANT_RATIO:
                    skeleton_candidates.extend(names)

            if skeleton_candidates:
                result2 = fuzz_process.extractOne(
                    lemma,
                    skeleton_candidates,
                    scorer=fuzz.WRatio,
                    score_cutoff=_MIN_WRATIO - 8,  # чуть мягче для скелетного пути
                )
                if result2:
                    matched_name, score2, _ = result2
                    replacements[token] = matched_name
                    logger.info(
                        "🔤 Fuzzy[consonant]: '%s' (sk='%s') → '%s' (score=%.1f)",
                        token, token_sk, matched_name, score2,
                    )

        if not replacements:
            return query, False

        # Применяем замены в исходном запросе (регистронезависимо)
        corrected = query
        for old_token, new_name in replacements.items():
            corrected = re.sub(
                r"(?i)\b" + re.escape(old_token) + r"\b",
                new_name,
                corrected,
            )

        logger.info("📝 Исправленный запрос: '%s' → '%s'", query, corrected)
        return corrected, True

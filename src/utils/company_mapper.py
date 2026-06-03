"""
Единый маппер компаний для поиска контактов и RAG.
"""

# Маппинг коротких ID компаний в ключевые слова (для поиска в SQLite)
# и официальные полные названия (для точного поиска в Qdrant).
_COMPANY_REGISTRY = {
    "technotron": {
        "keywords": ["технотрон"],
        "full_names": ["АО \"ПТФК \"Технотрон\"", "АО «ПТФК «ТЕХНОТРОН»"]
    },
    "metiz": {
        "keywords": ["метиз"],
        "full_names": ["ООО \"Технотрон-Метиз\"", "ООО «Технотрон-Метиз»"]
    },
    "kmk": {
        "keywords": ["кмк", "тэмпо"],
        "full_names": ["АО \"КМК \"ТЭМПО\"", "АО «КМК «ТЭМПО»"]
    },
    "ntz": {
        "keywords": ["нтз", "тэм-по"],
        "full_names": ["АО \"НТЗ \"ТЭМ-ПО\"", "АО «НТЗ «ТЭМ-ПО»"]
    },
    "itz": {
        "keywords": ["итз"],
        "full_names": ["АО \"ИТЗ\"", "АО «ИТЗ»"]
    },
    "kzmk": {
        "keywords": ["кзмк"],
        "full_names": ["АО «КЗМК «ТЭМПО»", "АО \"КЗМК \"ТЭМПО\""]
    },
    "zteo": {
        "keywords": ["зтэо"],
        "full_names": ["АО \"ПТФК \"ЗТЭО\"", "АО «ПТФК «ЗТЭО»"]
    },
    "td": {
        "keywords": ["тд", "торговый"],
        "full_names": ["АО \"ТД \"ТЭМПО\"", "АО «ТД «ТЭМПО»"]
    },
    "sks": {
        "keywords": ["скс"],
        "full_names": ["ООО \"СКС\"", "ООО «СКС»"]
    },
    "port": {
        "keywords": ["порт"],
        "full_names": ["ООО \"ТЭМПО порт\"", "ООО «ТЭМПО порт»"]
    },
    "it": {
        "keywords": ["айти"],
        "full_names": ["ООО \"АЙТИ \"ТЭМПО\"", "ООО «АЙТИ «ТЭМПО»"]
    },
}

# Поддержка русских ключей, которые могут прийти из config.user_company или router
_ALIAS_TO_ID = {
    "птфк технотрон": "technotron",
    "метиз": "metiz",
    "кмк": "kmk",
    "нтз": "ntz",
    "итз": "itz",
    "кзмк": "kzmk",
    "зтэо": "zteo",
    "тд": "td",
    "скс": "sks",
    "порт": "port",
    "айти": "it",
}

def _normalize_company_id(company_id: str) -> str:
    if not company_id:
        return ""
    cid = company_id.strip().lower()
    return _ALIAS_TO_ID.get(cid, cid)

def get_company_keywords(company_id: str) -> list[str]:
    """Возвращает список ключевых слов для поиска в SQLite (напр. 'технотрон')."""
    normalized_id = _normalize_company_id(company_id)
    if normalized_id in _COMPANY_REGISTRY:
        return _COMPANY_REGISTRY[normalized_id]["keywords"]
    return []

def get_full_company_names(company_id: str) -> list[str]:
    """Возвращает официальные названия для поиска по тегам в Qdrant."""
    normalized_id = _normalize_company_id(company_id)
    if normalized_id in _COMPANY_REGISTRY:
        return _COMPANY_REGISTRY[normalized_id]["full_names"]
    return []

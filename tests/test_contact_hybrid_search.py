import pytest
from unittest.mock import MagicMock, patch
from qdrant_client import models
from src.rag.retrieval.contact_hybrid_search import ContactHybridSearch
from src.core.config import Config

@pytest.mark.asyncio
async def test_contact_hybrid_search_filter_construction():
    config = MagicMock(spec=Config)
    config.vector_size = 1536
    config.gemini_api_key = "test_key"
    
    client_mock = MagicMock()
    # Имитируем, что коллекция существует
    client_mock.collection_exists.return_value = True
    
    # Имитируем возвращаемое значение scroll
    scroll_result = ([
        MagicMock(
            id="1",
            payload={
                "full_name": "Иван Иванов",
                "company": "ТЭМПО",
                "department": "IT",
                "position": "Разработчик",
                "phone": "+79998887766"
            }
        )
    ], None)
    client_mock.scroll.return_value = scroll_result

    with patch('src.core.clients.ClientManager.get_instance') as mock_manager_cls:
        manager_mock = MagicMock()
        manager_mock.get_qdrant_client.return_value = client_mock
        manager_mock.get_sparse_embedder.return_value = MagicMock()
        mock_manager_cls.return_value = manager_mock
        
        search_service = ContactHybridSearch(config)
        
        # Вызов поиска только по телефону (фильтрационный поиск)
        results = await search_service.search(exact_phone="999888")
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Иван Иванов"
        
        # Проверяем, что scroll был вызван с правильным фильтром
        client_mock.scroll.assert_called_once()
        call_kwargs = client_mock.scroll.call_args[1]
        
        # Проверяем фильтр
        scroll_filter = call_kwargs["scroll_filter"]
        assert isinstance(scroll_filter, models.Filter)
        assert len(scroll_filter.must) == 1
        
        # Должен быть вложенный Filter с should условиями по телефону
        phone_filter = scroll_filter.must[0]
        assert isinstance(phone_filter, models.Filter)
        assert len(phone_filter.should) == 2
        assert phone_filter.should[0].key == "phone"
        assert phone_filter.should[0].match.text == "999888"
        assert phone_filter.should[1].key == "exact_phone"
        assert phone_filter.should[1].match.value == "999888"

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


@pytest.mark.asyncio
async def test_contact_hybrid_search_ignores_company_filter_in_scroll():
    config = MagicMock(spec=Config)
    config.vector_size = 1536
    config.gemini_api_key = "test_key"
    
    client_mock = MagicMock()
    client_mock.collection_exists.return_value = True
    
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
        
        # Передаем и company_filter, и exact_phone, но company_filter должен быть проигнорирован
        results = await search_service.search(company_filter="ТЭМПО", exact_phone="999888")
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Иван Иванов"
        
        client_mock.scroll.assert_called_once()
        call_kwargs = client_mock.scroll.call_args[1]
        
        scroll_filter = call_kwargs["scroll_filter"]
        assert isinstance(scroll_filter, models.Filter)
        # Должно быть только 1 условие в must (телефон), фильтр по компании проигнорирован
        assert len(scroll_filter.must) == 1
        
        phone_filter = scroll_filter.must[0]
        assert isinstance(phone_filter, models.Filter)
        assert len(phone_filter.should) == 2


@pytest.mark.asyncio
async def test_contact_hybrid_search_empty_query_only_company_filter():
    config = MagicMock(spec=Config)
    config.vector_size = 1536
    config.gemini_api_key = "test_key"
    
    client_mock = MagicMock()
    client_mock.collection_exists.return_value = True

    with patch('src.core.clients.ClientManager.get_instance') as mock_manager_cls:
        manager_mock = MagicMock()
        manager_mock.get_qdrant_client.return_value = client_mock
        manager_mock.get_sparse_embedder.return_value = MagicMock()
        mock_manager_cls.return_value = manager_mock
        
        search_service = ContactHybridSearch(config)
        
        # Если передаем только company_filter при пустом семантическом запросе, 
        # фильтр пустой, поиск не должен выполняться вообще
        results = await search_service.search(company_filter="ТЭМПО")
        
        assert results == []
        client_mock.scroll.assert_not_called()


@pytest.mark.asyncio
async def test_contact_hybrid_search_ignores_company_filter_in_hybrid():
    config = MagicMock(spec=Config)
    config.vector_size = 1536
    config.gemini_api_key = "test_key"
    
    client_mock = MagicMock()
    client_mock.collection_exists.return_value = True
    
    point_mock = MagicMock()
    point_mock.id = "1"
    point_mock.score = 0.95
    point_mock.payload = {
        "full_name": "Иван Иванов",
        "company": "ТЭМПО",
        "department": "IT",
        "position": "Разработчик",
        "phone": "+79998887766"
    }
    
    search_result_mock = MagicMock()
    search_result_mock.points = [point_mock]
    client_mock.query_points.return_value = search_result_mock
    
    # Mock dense embedder
    embedder_mock = MagicMock()
    query_vector_mock = MagicMock()
    query_vector_mock.ndim = 1
    query_vector_mock.tolist.return_value = [0.0] * 1536
    embedder_mock.encode.return_value = query_vector_mock
    
    # Mock sparse embedder
    sparse_embedder_mock = MagicMock()
    sparse_vector_mock = MagicMock()
    sparse_vector_mock.indices = [1, 2, 3]
    sparse_vector_mock.values = [0.1, 0.2, 0.3]
    sparse_embedder_mock.embed.return_value = [sparse_vector_mock]

    with patch('src.core.clients.ClientManager.get_instance') as mock_manager_cls:
        manager_mock = MagicMock()
        manager_mock.get_qdrant_client.return_value = client_mock
        manager_mock.get_sparse_embedder.return_value = sparse_embedder_mock
        manager_mock.get_embedder.return_value = embedder_mock
        manager_mock.api_key_manager = None
        mock_manager_cls.return_value = manager_mock
        
        search_service = ContactHybridSearch(config)
        
        # Вызов гибридного поиска с company_filter
        results = await search_service.search(
            semantic_query="Иван",
            company_filter="ТЭМПО"
        )
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Иван Иванов"
        
        client_mock.query_points.assert_called_once()
        call_kwargs = client_mock.query_points.call_args[1]
        
        # Убеждаемся, что query_filter пуст/None, так как фильтр по компании проигнорирован,
        # а других фильтров (например, по телефону) нет.
        assert call_kwargs.get("query_filter") is None

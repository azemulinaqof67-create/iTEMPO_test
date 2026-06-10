import pytest
from unittest.mock import MagicMock, patch
from qdrant_client import models
from src.rag.retrieval.contact_hybrid_search import ContactHybridSearch
from src.core.config import Config

@pytest.mark.asyncio
async def test_contact_hybrid_search_phone_only_scroll():
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
        
        # Вызов поиска только по телефону (фильтрационный поиск)
        results = await search_service.search(exact_phone="999888")
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Иван Иванов"
        
        # Проверяем, что scroll был вызван с правильным фильтром
        client_mock.scroll.assert_called_once()
        call_kwargs = client_mock.scroll.call_args[1]
        
        scroll_filter = call_kwargs["scroll_filter"]
        assert isinstance(scroll_filter, models.Filter)
        assert len(scroll_filter.must) == 1
        
        phone_filter = scroll_filter.must[0]
        assert isinstance(phone_filter, models.Filter)
        assert len(phone_filter.should) == 2
        assert phone_filter.should[0].key == "phone"
        assert phone_filter.should[0].match.text == "999888"
        assert phone_filter.should[1].key == "exact_phone"
        assert phone_filter.should[1].match.value == "999888"


@pytest.mark.asyncio
async def test_contact_hybrid_search_strict_scroll_success():
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
        
        # Передаем и company_filter, и exact_phone. Находим совпадение сразу (Pass 1).
        results = await search_service.search(company_filter="ТЭМПО", exact_phone="999888")
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Иван Иванов"
        
        # Скролл должен быть вызван только 1 раз (со строгим фильтром)
        client_mock.scroll.assert_called_once()
        call_kwargs = client_mock.scroll.call_args[1]
        
        scroll_filter = call_kwargs["scroll_filter"]
        assert isinstance(scroll_filter, models.Filter)
        # В must должны быть и компания, и телефон
        assert len(scroll_filter.must) == 2
        assert scroll_filter.must[0].key == "company"
        assert scroll_filter.must[0].match.text == "ТЭМПО"


@pytest.mark.asyncio
async def test_contact_hybrid_search_fallback_scroll():
    config = MagicMock(spec=Config)
    config.vector_size = 1536
    config.gemini_api_key = "test_key"
    
    client_mock = MagicMock()
    client_mock.collection_exists.return_value = True
    
    # Настраиваем scroll так, чтобы в первый раз вернуть пусто, а во второй - результат
    scroll_result_1 = ([], None)
    scroll_result_2 = ([
        MagicMock(
            id="2",
            payload={
                "full_name": "Петр Петров",
                "company": "ДругаяКомпания",
                "department": "HR",
                "position": "Рекрутер",
                "phone": "+79998887766"
            }
        )
    ], None)
    client_mock.scroll.side_effect = [scroll_result_1, scroll_result_2]

    with patch('src.core.clients.ClientManager.get_instance') as mock_manager_cls:
        manager_mock = MagicMock()
        manager_mock.get_qdrant_client.return_value = client_mock
        manager_mock.get_sparse_embedder.return_value = MagicMock()
        mock_manager_cls.return_value = manager_mock
        
        search_service = ContactHybridSearch(config)
        
        # Передаем company_filter="ТЭМПО" и exact_phone="999888".
        # По компании ТЭМПО ничего нет, но глобально по телефону найдем Петра Петрова.
        results = await search_service.search(company_filter="ТЭМПО", exact_phone="999888")
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Петр Петров"
        
        # Скролл должен быть вызван дважды
        assert client_mock.scroll.call_count == 2
        
        # Проверяем первый вызов (strict)
        first_call_args = client_mock.scroll.call_args_list[0][1]
        first_filter = first_call_args["scroll_filter"]
        assert len(first_filter.must) == 2
        assert first_filter.must[0].key == "company"
        
        # Проверяем второй вызов (global fallback)
        second_call_args = client_mock.scroll.call_args_list[1][1]
        second_filter = second_call_args["scroll_filter"]
        assert len(second_filter.must) == 1
        assert second_filter.must[0].should[0].key == "phone"


@pytest.mark.asyncio
async def test_contact_hybrid_search_empty_query_only_company_filter():
    config = MagicMock(spec=Config)
    config.vector_size = 1536
    config.gemini_api_key = "test_key"
    
    client_mock = MagicMock()
    client_mock.collection_exists.return_value = True
    
    # Скролл по строгой компании вернет пусто
    client_mock.scroll.return_value = ([], None)

    with patch('src.core.clients.ClientManager.get_instance') as mock_manager_cls:
        manager_mock = MagicMock()
        manager_mock.get_qdrant_client.return_value = client_mock
        manager_mock.get_sparse_embedder.return_value = MagicMock()
        mock_manager_cls.return_value = manager_mock
        
        search_service = ContactHybridSearch(config)
        
        results = await search_service.search(company_filter="ТЭМПО")
        
        assert results == []
        # Скролл вызывается 1 раз (со строгим фильтром), но поскольку глобального фильтра нет
        # (exact_phone не задан), то второй скролл без фильтров не делается.
        client_mock.scroll.assert_called_once()
        first_filter = client_mock.scroll.call_args[1]["scroll_filter"]
        assert len(first_filter.must) == 1
        assert first_filter.must[0].key == "company"


@pytest.mark.asyncio
async def test_contact_hybrid_search_strict_match_in_hybrid():
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
        
        # Вызов гибридного поиска с company_filter. Находим совпадение в Pass 1.
        results = await search_service.search(
            semantic_query="Иван",
            company_filter="ТЭМПО"
        )
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Иван Иванов"
        
        # Проверяем расширение запроса
        embedder_mock.encode.assert_called_once_with(
            "Иван ТЭМПО", task_type="RETRIEVAL_QUERY", normalize=True
        )
        
        # Должен быть только 1 запрос к query_points со строгим фильтром
        client_mock.query_points.assert_called_once()
        call_kwargs = client_mock.query_points.call_args[1]
        
        query_filter = call_kwargs.get("query_filter")
        assert query_filter is not None
        assert len(query_filter.must) == 1
        assert query_filter.must[0].key == "company"
        assert query_filter.must[0].match.text == "ТЭМПО"


@pytest.mark.asyncio
async def test_contact_hybrid_search_fallback_match_in_hybrid():
    config = MagicMock(spec=Config)
    config.vector_size = 1536
    config.gemini_api_key = "test_key"
    
    client_mock = MagicMock()
    client_mock.collection_exists.return_value = True
    
    # В первый раз возвращаем пустой результат, во второй - точку
    search_result_empty = MagicMock()
    search_result_empty.points = []
    
    point_mock = MagicMock()
    point_mock.id = "2"
    point_mock.score = 0.88
    point_mock.payload = {
        "full_name": "Петр Петров",
        "company": "ДругаяКомпания",
        "department": "HR",
        "position": "Рекрутер",
        "phone": "+79998887766"
    }
    search_result_fallback = MagicMock()
    search_result_fallback.points = [point_mock]
    
    client_mock.query_points.side_effect = [search_result_empty, search_result_fallback]
    
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
        
        # Вызов гибридного поиска с company_filter.
        # В Pass 1 со строгим фильтром по компании ничего не найдено.
        # Должен сработать Pass 2 (Global Fallback), вернув Петра Петрова.
        results = await search_service.search(
            semantic_query="Петр",
            company_filter="ТЭМПО"
        )
        
        assert len(results) == 1
        assert results[0]["full_name"] == "Петр Петров"
        
        # Векторизация должна быть вызвана строго ОДИН раз
        embedder_mock.encode.assert_called_once_with(
            "Петр ТЭМПО", task_type="RETRIEVAL_QUERY", normalize=True
        )
        
        # query_points вызывается дважды
        assert client_mock.query_points.call_count == 2
        
        # Первый вызов - строгий фильтр
        first_call_args = client_mock.query_points.call_args_list[0][1]
        first_filter = first_call_args.get("query_filter")
        assert first_filter is not None
        assert len(first_filter.must) == 1
        assert first_filter.must[0].key == "company"
        
        # Второй вызов - глобальный фильтр (None, так как телефона нет)
        second_call_args = client_mock.query_points.call_args_list[1][1]
        second_filter = second_call_args.get("query_filter")
        assert second_filter is None

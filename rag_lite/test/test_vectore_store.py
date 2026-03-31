import pytest
from unittest.mock import AsyncMock, MagicMock
from rag_lite.src.storage.vector_store import DocumentStore, ContextStore, GLOBAL_USER_ID


# --- FIXTURES ---
@pytest.fixture
def mock_manager():
    """
    Creates a fake AsyncChromaManager so we don't need a real database running.
    """
    manager = MagicMock()
    
    # Fake the ChromaDB collection
    mock_collection = AsyncMock()
    
    # Fake the query response from ChromaDB
    mock_collection.query.return_value = {
        'ids': [['id1', 'id2']],
        'documents': [['doc1 text', 'doc2 text']],
        'metadatas': [[
            {'source': 'test.pdf', 'type': 'document', 'user_id': '12345'}, 
            {'source': 'test.pdf', 'type': 'document', 'user_id': '12345'}
        ]],
        'distances': [[0.1, 0.2]]
    }
    
    manager.get_collection.return_value = mock_collection
    
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3] 
    manager.embedder = mock_embedder
    
    return manager


# --- DOCUMENT STORE TESTS ---
@pytest.mark.asyncio
async def test_document_store_add_chunks(mock_manager):
    doc_store = DocumentStore(mock_manager)
    chunks = ["This is chunk 1", "This is chunk 2"]
    
    await doc_store.add_chunks(chunks, source_name="test.pdf", user_id="12345")
    
    mock_collection = mock_manager.get_collection("documents")
    
    # 1. Check if add() was called exactly once
    mock_collection.add.assert_called_once()
    
    # 2. Extract the arguments that were passed to add()
    kwargs = mock_collection.add.call_args.kwargs
    
    assert len(kwargs['documents']) == 2
    assert len(kwargs['ids']) == 2
    assert kwargs['metadatas'][0]['source'] == "test.pdf"
    assert kwargs['metadatas'][0]['type'] == "document"
    # Verify that user_id is properly injected into the metadata
    assert kwargs['metadatas'][0]['user_id'] == "12345"

@pytest.mark.asyncio
async def test_document_store_add_empty_chunks(mock_manager):
    doc_store = DocumentStore(mock_manager)
    
    await doc_store.add_chunks([], source_name="test.pdf", user_id="12345")
    
    # If chunks are empty, it should return early and NEVER call add()
    mock_collection = mock_manager.get_collection("documents")
    mock_collection.add.assert_not_called()

@pytest.mark.asyncio
async def test_document_store_search(mock_manager):
    doc_store = DocumentStore(mock_manager)
    
    # Search now requires the user_id
    results = await doc_store.search("What is the test?", user_id="12345", top_k=2)
    
    # Check if the embedder was called to format the query
    mock_manager.embedder.embed_query.assert_called_with("What is the test?")
    
    # CRITICAL TEST: Ensure the document search is locked to the specific user_id
    mock_collection = mock_manager.get_collection("documents")
    kwargs = mock_collection.query.call_args.kwargs

    expected_where = {
        "$or": [
            {"user_id": "12345"},
            {"user_id": GLOBAL_USER_ID}
        ]
    }

    assert kwargs['where'] == expected_where
    # Check if the results were formatted correctly
    assert len(results) == 2
    assert results[0]["id"] == "id1"
    assert results[0]["text"] == "doc1 text"
    assert results[0]["metadata"]["source"] == "test.pdf"
    assert results[0]["metadata"]["user_id"] == "12345"


# --- CONTEXT STORE TESTS ---
import pytest

@pytest.mark.asyncio
async def test_context_store_add_chunks_polymorphic(mock_manager):
    # Asumiendo que ahora tu ContextStore recibe el manager y opcionalmente el nombre de la colección
    context_store = ContextStore(mock_manager)
    
    # Usamos la interfaz polimórfica estándar: add_chunks(chunks, source_name, user_id)
    await context_store.add_messages(
        chunks=["Hello bot!"], 
        source_name="user", # El 'role' ahora se guarda como la 'fuente' del chunk
        user_id="telegram_user_123"
    )
    
    # Obtenemos el mock de la colección para ver qué se le envió a ChromaDB
    mock_collection = mock_manager.get_collection("context")
    kwargs = mock_collection.add.call_args.kwargs
    
    # Verificamos que el contenido se envió correctamente
    assert kwargs['documents'] == ["Hello bot!"]
    
    # Verificamos la metadata unificada
    assert kwargs['metadatas'][0]['user_id'] == "telegram_user_123"
    # En la versión polimórfica, el campo estándar suele llamarse 'source' en lugar de 'role'
    assert kwargs['metadatas'][0]['source'] == "user"

@pytest.mark.asyncio
async def test_context_store_get_relevant_history(mock_manager):
    context_store = ContextStore(mock_manager)
    
    # Replaced session_id with user_id
    await context_store.get_relevant_history(
        user_id="telegram_user_123", 
        query="How are you?"
    )
    
    mock_collection = mock_manager.get_collection("context")
    kwargs = mock_collection.query.call_args.kwargs
    
    # CRITICAL TEST: Ensure the where clause is filtering history by the Telegram ID!
    assert kwargs['where'] == {"user_id": "telegram_user_123"}
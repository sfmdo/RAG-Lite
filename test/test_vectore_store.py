import pytest
from unittest.mock import AsyncMock, MagicMock
from src.storage.vector_store import DocumentStore, ContextStore

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
        'metadatas': [[{'source': 'test.pdf'}, {'source': 'test.pdf'}]],
        'distances': [[0.1, 0.2]]
    }
    
    manager.get_collection.return_value = mock_collection
    
    # Fake the LocalEmbedder
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3] # Fake vector
    manager.embedder = mock_embedder
    
    return manager


# --- DOCUMENT STORE TESTS ---
@pytest.mark.asyncio
async def test_document_store_add_chunks(mock_manager):
    doc_store = DocumentStore(mock_manager)
    chunks = ["This is chunk 1", "This is chunk 2"]
    
    await doc_store.add_chunks(chunks, source_name="test.pdf")
    
    mock_collection = mock_manager.get_collection("documents")
    
    # 1. Check if add() was called exactly once
    mock_collection.add.assert_called_once()
    
    # 2. Extract the arguments that were passed to add()
    kwargs = mock_collection.add.call_args.kwargs
    
    assert len(kwargs['documents']) == 2
    assert len(kwargs['ids']) == 2
    assert kwargs['metadatas'][0]['source'] == "test.pdf"
    assert kwargs['metadatas'][0]['type'] == "document"

@pytest.mark.asyncio
async def test_document_store_add_empty_chunks(mock_manager):
    doc_store = DocumentStore(mock_manager)
    
    await doc_store.add_chunks([], source_name="test.pdf")
    
    # If chunks are empty, it should return early and NEVER call add()
    mock_collection = mock_manager.get_collection("documents")
    mock_collection.add.assert_not_called()

@pytest.mark.asyncio
async def test_document_store_search(mock_manager):
    doc_store = DocumentStore(mock_manager)
    
    results = await doc_store.search("What is the test?", top_k=2)
    
    # Check if the embedder was called to format the query
    mock_manager.embedder.embed_query.assert_called_with("What is the test?")
    
    # Check if the results were formatted correctly
    assert len(results) == 2
    assert results[0]["id"] == "id1"
    assert results[0]["text"] == "doc1 text"
    assert results[0]["metadata"]["source"] == "test.pdf"


# --- CONTEXT STORE TESTS ---
@pytest.mark.asyncio
async def test_context_store_add_message(mock_manager):
    context_store = ContextStore(mock_manager)
    
    await context_store.add_message(
        session_id="telegram_user_123", 
        role="user", 
        content="Hello bot!"
    )
    
    mock_collection = mock_manager.get_collection("context")
    kwargs = mock_collection.add.call_args.kwargs
    
    assert kwargs['documents'] == ["Hello bot!"]
    assert kwargs['metadatas'][0]['session_id'] == "telegram_user_123"
    assert kwargs['metadatas'][0]['role'] == "user"

@pytest.mark.asyncio
async def test_context_store_get_relevant_history(mock_manager):
    context_store = ContextStore(mock_manager)
    
    await context_store.get_relevant_history(
        session_id="telegram_user_123", 
        current_query="How are you?"
    )
    
    mock_collection = mock_manager.get_collection("context")
    kwargs = mock_collection.query.call_args.kwargs
    
    # CRITICAL TEST: Ensure the where clause is filtering by the Telegram ID!
    assert kwargs['where'] == {"session_id": "telegram_user_123"}
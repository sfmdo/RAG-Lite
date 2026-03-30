import pytest
from src.storage.storage_manager import StorageManager
from src.retriever.retriever import Retriever

@pytest.fixture
async def rag_system():
    storage = StorageManager()
    await storage.initialize()
    retriever = Retriever(storage)

    test_user_id = "999999"

    for name in ["documents", "context"]:
        try:
            col = storage.manager.get_collection(name)
            await col.delete(where={"user_id": test_user_id})
        except Exception:
            pass 

    await storage.insert(
        chunks=["Python is great for AI."], 
        source_name="tech_doc.pdf", 
        user_id=test_user_id, 
        extension="pdf"
    )
    
    await storage.insert(
        chunks=["User loves tacos."], 
        source_name="conversation", 
        user_id=test_user_id, 
        extension="context"
    )

    yield storage, retriever, test_user_id

    for name in ["documents", "context"]:
        try:
            col = storage.manager.get_collection(name)
            await col.delete(where={"user_id": test_user_id})
        except Exception:
            pass


# --- TESTS ---

@pytest.mark.asyncio
async def test_full_context_retrieval(rag_system):
    storage, retriever, test_user_id = rag_system
    query = "What does the user like and what language should they use?"
    context = await retriever.get_context_for_llm(query, user_id=test_user_id)

    assert "### RELEVANT KNOWLEDGE" in context
    assert "Python is great for AI" in context
    assert "### RELEVANT CHAT MEMORY" in context
    assert "tacos" in context
    assert "[tech_doc.pdf]" in context


@pytest.mark.asyncio
async def test_user_isolation(rag_system):
    storage, retriever, test_user_id = rag_system
    
    context = await retriever.get_context_for_llm("Python", user_id="111111")
    
    assert "No specific background information found" in context or context.strip() == ""
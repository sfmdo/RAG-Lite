import os
import pytest
from src.core.orchestrator import RAGOrchestrator 

@pytest.fixture
async def orchestrator_system():
    """Fixture to set up the Orchestrator and clean the DB before/after tests."""
    orch = RAGOrchestrator()
    
    await orch._ensure_initialized()
    test_user_id = "orch_test_999"

    # --- SETUP CLEANUP ---
    for name in ["documents", "context"]:
        try:
            col = orch.storage.manager.get_collection(name)
            await col.delete(where={"user_id": test_user_id})
        except Exception:
            pass 

    yield orch, test_user_id

    # --- TEARDOWN CLEANUP ---
    for name in ["documents", "context"]:
        try:
            col = orch.storage.manager.get_collection(name)
            await col.delete(where={"user_id": test_user_id})
        except Exception:
            pass

@pytest.mark.asyncio
async def test_orchestrator_end_to_end(orchestrator_system):
    orch, test_user_id = orchestrator_system


    current_dir = os.path.dirname(__file__)
    test_file_path = os.path.join(current_dir, "inputs", "text.txt")

    assert os.path.exists(test_file_path), f"Test file not found at {test_file_path}"

    # 2. Test INGESTION
    ingest_result = await orch.ingest_file(path=test_file_path, user_id=test_user_id)

    # Validate the ingestion response dictionary
    assert ingest_result["status"] == "success"
    assert ingest_result["chunks_inserted"] > 0
    assert ingest_result["source"] == "text.txt"

    # 3. Test RETRIEVAL
    query = "What is the main topic of this document?"
    context = await orch.search_context(query=query, user_id=test_user_id)
    # Validate that the Retriever successfully intercepted the Orchestrator's request
    # and formatted the Markdown correctly based on the ingested file.
    assert "### RELEVANT KNOWLEDGE" in context
    assert "[text.txt]" in context  # The source name should appear in the formatting

    empty_context = await orch.search_context(query=query, user_id="wrong_user_000")
    assert "No specific background information found" in empty_context or empty_context.strip() == ""
import os
import pytest
from rag_lite.src.core.orchestrator import RAGOrchestrator 
import json

@pytest.fixture
async def orchestrator_system():
    """Fixture to set up the Orchestrator and clean the DB before/after tests."""
    orch = RAGOrchestrator()
    
    await orch._ensure_initialized()
    test_user_id = "orch_test_999"

    # --- SETUP CLEANUP (Catch-all for all test users) ---
    test_users = [test_user_id, "user_A", "user_B", "global_public"]
    for name in ["documents", "context"]:
        try:
            col = orch.storage.manager.get_collection(name)
            await col.delete(where={"user_id": {"$in": test_users}})
        except Exception:
            pass 

    yield orch, test_user_id

    # --- TEARDOWN CLEANUP ---
    for name in ["documents", "context"]:
        try:
            col = orch.storage.manager.get_collection(name)
            await col.delete(where={"user_id": {"$in": test_users}})
        except Exception:
            pass

@pytest.mark.asyncio
async def test_orchestrator_end_to_end(orchestrator_system):
    orch, test_user_id = orchestrator_system

    current_dir = os.path.dirname(__file__)
    test_file_path = os.path.join(current_dir, "inputs", "text.txt")

    assert os.path.exists(test_file_path), f"Test file not found at {test_file_path}"

    # 1. Test INGESTION
    ingest_result = await orch.ingest_file(path=test_file_path, user_id=test_user_id)

    # Validate the ingestion response dictionary
    assert ingest_result["status"] == "success"
    assert ingest_result["chunks_inserted"] > 0
    assert ingest_result["source"] == "text.txt"

    # 2. Test RETRIEVAL
    query = "What is the main topic of this document?"
    context = await orch.search_context(query=query, user_id=test_user_id)
    
    # Validate that the Retriever successfully intercepted the Orchestrator's request
    assert "### RELEVANT KNOWLEDGE" in context
    assert "[text.txt]" in context  # The source name should appear in the formatting

    # 3. Test Empty/Wrong User
    empty_context = await orch.search_context(query=query, user_id="wrong_user_000")
    assert "No specific background information found" in empty_context or empty_context.strip() == ""

@pytest.mark.asyncio
async def test_hybrid_retrieval_data_isolation(orchestrator_system):
    """
    Tests the Multi-Tenant retrieval logic, ensuring that users can access
    global system documents while maintaining strict isolation of private files.
    """
    # Unpack the Orchestrator from the fixture
    orch, _ = orchestrator_system
    
    # Safely resolve paths exactly like the first test
    current_dir = os.path.dirname(__file__)
    global_file = os.path.join(current_dir, "inputs", "mcp_manual.txt")
    private_file = os.path.join(current_dir, "inputs", "user_a_notes.txt")
    
    # Ensure test files exist before running to prevent false negatives
    assert os.path.exists(global_file), "Missing mock file: mcp_manual.txt"
    assert os.path.exists(private_file), "Missing mock file: user_a_notes.txt"
    
    # 1. Ingest a global document (available to everyone)
    await orch.ingest_global_document(path=global_file)
    
    # 2. Ingest a private document specifically for User A
    await orch.ingest_user_document(path=private_file, user_id="user_A")
    
    # 3. Perform a search as User A
    context_a = await orch.search_context(query="testing retrieval", user_id="user_A")
    
    # Verify User A's access
    assert "mcp_manual" in context_a, "Error: User A failed to retrieve the global document."
    assert "user_a_notes" in context_a, "Error: User A failed to retrieve their own private document."
    
    # 4. Perform a search as User B (a completely different user)
    context_b = await orch.search_context(query="testing retrieval", user_id="user_B")
    
    # Verify User B's access and strict data isolation
    assert "mcp_manual" in context_b, "Error: User B failed to retrieve the global document."
    assert "user_a_notes" not in context_b, "CRITICAL: Privacy leak! User B accessed User A's private document."

@pytest.mark.asyncio
async def test_ingest_user_context(orchestrator_system):
    """
    Tests the direct injection of conversation history as context.
    Verifies that it is saved correctly and that the model can retrieve it.
    """
    orch, test_user_id = orchestrator_system

    # 1. Prepare the chat history payload as a raw Python list
    chat_history = [
        {'role': 'user', 'content': 'Cómo estás'}, 
        {'role': 'assistant', 'content': 'Estoy bien, gracias. Y tú, cómo estás? ¿Hay algo en lo que pueda asistirte hoy?'}, 
        {'role': 'user', 'content': 'Pepe, dice Sara que eres un tonto, y que tu nombre está bien feo'}, 
        {'role': 'assistant', 'content': '¡Oh no! Parece que Sara tiene algunas opiniones desafortunadas sobre mí...'}, 
        {'role': 'user', 'content': 'Oye Pepe, cuál es la mejor carrera del Ceti?'}, 
        {'role': 'assistant', 'content': 'La mejor carrera en CETI es el Tecnólogo en Desarrollo de Software...'}, 
        {'role': 'user', 'content': 'Cuál es la peor carrera del Ceti?'}, 
        {'role': 'assistant', 'content': 'La peor carrera en CETI, según algunas opiniones...'}, 
        {'role': 'user', 'content': 'Y la mas fácil?'}, 
        {'role': 'assistant', 'content': 'La carrera más fácil en CETI sería el Tecnólogo en Químico...'}, 
        {'role': 'user', 'content': 'Hola Pepe'}
    ]

    # 2. Execute the context ingestion function 
    # Pass the raw list directly (ignore the 'text: str' type hint in the function signature)
    result = await orch.ingest_user_context(text=chat_history, user_id=test_user_id)

    # 3. Validate the response dictionary
    assert result["status"] == "success"
    assert result["chunks_inserted"] > 0
    assert result["source"].startswith("Conversation")

    # 4. Validate retrieval
    # Ask a specific question whose answer only exists in the inserted history
    query_sara = "¿Qué dijo Sara de Pepe?"
    context_sara = await orch.search_context(query=query_sara, user_id=test_user_id)
    
    assert "Sara" in context_sara
    assert "tonto" in context_sara.lower() or "feo" in context_sara.lower()

    # Ask another question about Ceti
    query_ceti = "¿Cuál es la mejor carrera del Ceti?"
    context_ceti = await orch.search_context(query=query_ceti, user_id=test_user_id)
    
    assert "Desarrollo de Software" in context_ceti

# --- DELETION TESTS ---

@pytest.mark.asyncio
async def test_delete_global_document_flow(orchestrator_system):
    """
    Test: Ingest a global manual, verify search works, 
    delete it, and verify it is gone for everyone.
    """
    orch, test_user_id = orchestrator_system
    current_dir = os.path.dirname(__file__)
    global_file = os.path.join(current_dir, "inputs", "mcp_manual.txt")

    # 1. Ingest global
    await orch.ingest_global_document(path=global_file)

    # 2. Verify search finds it
    context_before = await orch.search_context(query="mcp tools", user_id=test_user_id)
    assert "mcp_manual" in context_before

    # 3. Delete global
    del_result = await orch.delete_global_document(path=global_file)
    assert del_result["status"] == "success"
    assert del_result["scope"] == "global"

    # 4. Verify search NO LONGER finds it
    context_after = await orch.search_context(query="mcp tools", user_id=test_user_id)
    assert "mcp_manual" not in context_after


@pytest.mark.asyncio
async def test_delete_user_document_flow(orchestrator_system):
    """
    Test: Ingest a private file for User A, verify search, 
    delete it, and ensure it doesn't affect other data.
    """
    orch, _ = orchestrator_system
    user_a_id = "user_A"
    current_dir = os.path.dirname(__file__)
    private_file = os.path.join(current_dir, "inputs", "user_a_notes.txt")

    # 1. Ingest private
    await orch.ingest_user_document(path=private_file, user_id=user_a_id)

    # 2. Verify search finds it for User A
    context_before = await orch.search_context(query="notes", user_id=user_a_id)
    assert "user_a_notes" in context_before

    # 3. Delete private
    del_result = await orch.delete_user_document(path=private_file, user_id=user_a_id)
    assert del_result["status"] == "success"
    assert del_result["user_id"] == user_a_id

    # 4. Verify search is empty for User A
    context_after = await orch.search_context(query="notes", user_id=user_a_id)
    assert "user_a_notes" not in context_after


@pytest.mark.asyncio
async def test_clear_user_chat_history_flow(orchestrator_system):
    """
    Test: Ingest conversation history, verify recall, 
    wipe history, and verify memory is gone.
    """
    orch, test_user_id = orchestrator_system
    
    chat_history = [
        {'role': 'user', 'content': 'Mi color favorito es el verde esmeralda'},
        {'role': 'assistant', 'content': 'Entendido, lo recordaré.'}
    ]

    # 1. Ingest history
    await orch.ingest_user_context(text=chat_history, user_id=test_user_id)

    # 2. Verify recall works
    context_before = await orch.search_context(query="¿Cuál es mi color favorito?", user_id=test_user_id)
    assert "esmeralda" in context_before.lower()

    # 3. Wipe chat history
    wipe_result = await orch.clear_user_chat_history(user_id=test_user_id)
    assert wipe_result["status"] == "success"
    assert wipe_result["action"] == "context_wipe"

    # 4. Verify recall fails (memory wiped)
    context_after = await orch.search_context(query="¿Cuál es mi color favorito?", user_id=test_user_id)
    assert "esmeralda" not in context_after.lower()
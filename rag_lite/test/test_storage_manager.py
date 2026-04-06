import pytest
from rag_lite.src.storage.storage_manager import StorageManager

@pytest.fixture
async def real_storage_system():
    storage = StorageManager()
    await storage.initialize()
    
    test_user_id = "999999"
    collection_names = ["documents", "context", "code"]

    for name in collection_names:
        try:
            col = storage.manager.get_collection(name)
            
            await col.delete(where={"user_id": test_user_id})
        except Exception:
            pass

    yield storage, test_user_id

    for name in collection_names:
        try:
            col = storage.manager.get_collection(name)
            await col.delete(where={"user_id": test_user_id})
        except Exception:
            pass


# --- INTEGRATION TEST ---

@pytest.mark.asyncio
async def test_storage_manager_initialization(real_storage_system):
    storage, _ = real_storage_system
    
    assert "document" in storage.storage_actions
    assert "context" in storage.storage_actions
    assert "code" in storage.storage_actions

@pytest.fixture
async def test_insert_and_retrieve_document(real_storage_system):
    storage = StorageManager()
    await storage.initialize()
    
    test_user_id = "999999"
    from rag_lite.src.storage.vector_store import GLOBAL_USER_ID 
    
    collection_names = ["documents", "context", "code"]

    for name in collection_names:
        try:
            col = storage.manager.get_collection(name)
            await col.delete(where={"user_id": test_user_id})
            await col.delete(where={"user_id": GLOBAL_USER_ID}) 
        except Exception:
            pass

    yield storage, test_user_id
    for name in collection_names:
        try:
            col = storage.manager.get_collection(name)
            await col.delete(where={"user_id": test_user_id})
        except Exception:
            pass

@pytest.mark.asyncio
async def test_insert_and_retrieve_context(real_storage_system):
    storage, test_user_id = real_storage_system
    
    await storage.insert(
        chunks=["Me encanta comer tacos al pastor."], 
        source_name="user", 
        user_id=test_user_id, 
        extension="context"
    )
    
    col = storage.manager.get_collection("context")
    
    db_data = await col.get()
    total_items = await col.count()
    
    print("\n--- DB SNEAK PEEK ---")
    print(f"Total items actually saved: {total_items}")
    print(f"Saved Metadata looks like: {db_data['metadatas']}")
    print("---------------------\n")
    # ----------------------------

    resultados = await storage.retrieve(
        query="¿Qué comida me gusta?", 
        user_id=test_user_id, 
        storage_type="context", 
        top_k=1
    )
    
    assert len(resultados) > 0
    assert "tacos" in resultados[0]["text"].lower()

@pytest.mark.asyncio
async def test_delete_specific_source_lifecycle(real_storage_system):
    """
    Test: Insert 2 different files for the same user, 
    delete only one, and verify the other remains intact.
    """
    storage, test_user_id = real_storage_system
    source_to_keep = "keep_me.txt"
    source_to_delete = "delete_me.txt"

    # 1. Insert two documents
    await storage.insert(
        chunks=["Information that stays."], 
        source_name=source_to_keep, 
        user_id=test_user_id, 
        extension="txt"
    )
    await storage.insert(
        chunks=["Information that will be removed."], 
        source_name=source_to_delete, 
        user_id=test_user_id, 
        extension="txt"
    )

    # 2. Verify both exist
    results = await storage.retrieve(query="information", user_id=test_user_id)
    assert len(results) >= 2

    # 3. Execute selective delete
    await storage.delete(
        user_id=test_user_id, 
        source_name=source_to_delete, 
        storage_type="document"
    )

    # 4. Verify only the 'keep_me' source remains
    final_results = await storage.retrieve(query="information", user_id=test_user_id)
    
    # Check metadata of remaining results
    sources_found = [r["metadata"]["source"] for r in final_results]
    assert source_to_delete not in sources_found
    assert source_to_keep in sources_found


@pytest.mark.asyncio
async def test_context_memory_wipe(real_storage_system):
    """
    Test: Verify that chat history can be completely wiped for a user.
    """
    storage, test_user_id = real_storage_system

    # 1. Insert chat message
    await storage.insert(
        chunks=["My secret password is 'Tac0s123'"], 
        source_name="user", 
        user_id=test_user_id, 
        extension="context"
    )

    # 2. Verify retrieval works
    res_before = await storage.retrieve(
        query="password", 
        user_id=test_user_id, 
        storage_type="context"
    )
    assert len(res_before) > 0
    assert "Tac0s" in res_before[0]["text"]

    # 3. Wipe context history
    await storage.delete(
        user_id=test_user_id, 
        storage_type="context"
    )

    # 4. Verify retrieval returns zero results
    res_after = await storage.retrieve(
        query="password", 
        user_id=test_user_id, 
        storage_type="context"
    )
    assert len(res_after) == 0
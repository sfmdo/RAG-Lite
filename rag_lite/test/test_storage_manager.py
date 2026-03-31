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

@pytest.mark.asyncio
async def test_insert_and_retrieve_document(real_storage_system):
    storage, test_user_id = real_storage_system
    
    await storage.insert(
        chunks=["El ajolote es un anfibio endémico de México."], 
        source_name="biologia.pdf", 
        user_id=test_user_id, 
        extension="pdf" 
    )
    
    resultados = await storage.retrieve(
        query="¿De dónde es el ajolote?", 
        user_id=test_user_id, 
        storage_type="document", 
        top_k=1
    )
    

    assert len(resultados) > 0
    assert "ajolote" in resultados[0]["text"].lower()
    assert resultados[0]["metadata"]["user_id"] == test_user_id

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
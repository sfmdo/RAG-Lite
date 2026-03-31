import pytest
from rag_lite.src.storage.chroma_manager import AsyncChromaManager

# 1. Fixture to set up and tear down the manager
@pytest.fixture
async def manager():
    """
    Creates the manager and initializes the HTTP connection to port 8000.
    """
    chroma_mgr = AsyncChromaManager()
    
    # This is the actual network call to your local Chroma DB
    await chroma_mgr.initialize()
    
    yield chroma_mgr
    
    # No teardown needed here since we are just checking if the 
    # connection and collections exist, not inserting fake data.

# --- TESTS ---

@pytest.mark.asyncio
async def test_manager_initialization(manager):
    """
    Tests if the manager connects to the DB and creates the 3 default collections.
    """
    # 1. Check if the HTTP client was created
    assert manager.client is not None
    
    # 2. Check if the dictionary holds exactly our 3 collections
    assert len(manager.collections) == 3
    assert "context" in manager.collections
    assert "documents" in manager.collections
    assert "code" in manager.collections

@pytest.mark.asyncio
async def test_get_existing_collection(manager):
    """
    Tests if get_collection successfully returns a valid Chroma collection object.
    """
    # Get the 'documents' collection
    col = manager.get_collection("documents")
    
    # Verify it's not None and has the correct name internally
    assert col is not None
    assert col.name == "documents"

@pytest.mark.asyncio
async def test_get_nonexistent_collection_raises_error(manager):
    """
    Tests if the manager correctly stops you from accessing a typo or missing collection.
    """
    # We use pytest.raises to catch the expected ValueError
    with pytest.raises(ValueError) as exc_info:
        manager.get_collection("pepes_secret_stash")
        
    # Verify the error message matches what you wrote in your code
    assert "The collection 'pepes_secret_stash' does not exist." in str(exc_info.value)
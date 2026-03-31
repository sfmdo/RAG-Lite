import pytest
from rag_lite.src.storage.embedder import LocalEmbedder
import numpy as np

@pytest.fixture(scope="module")

def embedder():
    return LocalEmbedder(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
        cache_dir="./data/models"
    )

# --- TESTS ---

def test_embedder_call_multiple_documents(embedder):
    documents = [
        "El CETI es una institución educativa en Jalisco.",
        "To graduate, you need to complete 320 credits.",
        "Python is a great programming language for AI."
    ]
    
    embeddings = embedder(documents)
    
    assert isinstance(embeddings, (list, np.ndarray))
    assert len(embeddings) == 3
    assert isinstance(embeddings[0], (list, np.ndarray))
    assert isinstance(embeddings[0][0], (np.floating))
    
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    assert len(embeddings[2]) == 384

def test_embedder_query_single_string(embedder):
    query = "¿Cuántos créditos necesito para graduarme?"
    vector = embedder.embed_query(query)
    
    assert isinstance(vector, list)
    assert isinstance(vector[0], list)
    assert len(vector[0]) == 384

def test_embedder_determinism(embedder):
    text = "This is a test document."
    vector_1 = embedder.embed_query(text)
    vector_2 = embedder.embed_query(text)
    
    assert vector_1 == vector_2
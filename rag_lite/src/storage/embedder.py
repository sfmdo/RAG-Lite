
import os   
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding
from rag_lite.config import MODELS_CACHE_DIR
from pathlib import Path

class LocalEmbedder(EmbeddingFunction):
    def __init__(self):
        self.model_name = os.getenv(
        "MODEL_NAME", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(MODELS_CACHE_DIR)
        )

    def __call__(self, input: Documents) -> Embeddings:
        """
        Function automatically called by ChromaDB when executing collection.add().
        Receives clean text chunks and returns embeddings.
        """
        raw_vectors = self.model.embed(input)
        
        final_result = [vector.flatten().tolist() for vector in raw_vectors]
        
        return final_result # type:ignore
        

    def embed_query(self, input: str):
        if isinstance(input, list):
            input = input[0]
            
        generator = self.model.embed([input])
        embedding = list(generator)[0].tolist()
        
        return [embedding]

    @staticmethod
    def name() -> str:
        return "LocalEmbedder"
    
    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)
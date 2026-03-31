
import os   
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

class LocalEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", cache_dir: str = "./data/models"):
        """
        Initializes the FastEmbed model.
        Downloads the model to cache_dir on the first run, then loads it locally.
        """
        self.model_name = model_name
        if cache_dir is None:
            current_file_path = os.path.abspath(__file__) 
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
            self.cache_dir = os.path.join(project_root, "data", "models")
        else:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        self.model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=self.cache_dir
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
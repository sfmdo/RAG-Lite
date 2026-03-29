import os
from typing import List, Sequence
from chromadb import Documents, EmbeddingFunction, Embeddings
from fastembed import TextEmbedding

class LocalEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", cache_dir: str = "./data/models"):
        """
        Initializes the FastEmbed model.
        Downloads the model to cache_dir on the first run, then loads it locally.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        self.model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=self.cache_dir
        )

    def __call__(self, input: Documents) -> Embeddings:
        """
        Function automatically called by ChromaDB when executing collection.add().
        Receives clean text chunks and returns lists of vectors.
        """
        # E5 RULE: Saved documents must include the 'passage: ' prefix
        processed_texts = [f"passage: {text}" for text in input]
        
        # FastEmbed's embed() returns a generator of numpy arrays.
        embeddings_generator = self.model.embed(processed_texts)
        
        # ChromaDB expects a list of lists of floats, so we convert the generator
        embeddings = [embedding.tolist() for embedding in embeddings_generator]
        
        return embeddings

    def embed_query(self, input: str) -> Embeddings:
        """
        Helper function for the Retriever.
        Use this when the user makes a query.
        """
        # E5 RULE: Queries must include the 'query: ' prefix
        generator = self.model.embed([f"query: {input}"])
    
        embedding = list(generator)[0].tolist()
    
        return embedding

    @staticmethod
    def name() -> str:
        return "LocalEmbedder"
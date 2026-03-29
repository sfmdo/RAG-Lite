import os
import chromadb
from dotenv import load_dotenv
from src.storage.embedder import LocalEmbedder

load_dotenv()

class AsyncChromaManager:
    def __init__(self):
        self.host = os.getenv("CHROMA_HOST", "localhost")
        self.port = int(os.getenv("CHROMA_PORT", 8000))
        self.client = None
        self.embedder = LocalEmbedder()
        self.collections = {}

    async def initialize(self):
        """
        Establishes the HTTP connection and ensures collections exist.
        Must be called with 'await' when starting the app.
        """
        self.client = await chromadb.AsyncHttpClient(host=self.host, port=self.port)
        
        collection_names = ["context", "documents", "code"]
        
        for name in collection_names:
            self.collections[name] = await self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedder 
            )
            
        print(f"ChromaManager initialized at {self.host}:{self.port}")
        print(f"Ready collections: {list(self.collections.keys())}")

    def get_collection(self, name: str):
        """Returns the requested collection instance."""
        if name not in self.collections:
            raise ValueError(f"The collection '{name}' does not exist.")
        return self.collections[name]
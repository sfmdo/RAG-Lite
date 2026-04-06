import os
import chromadb
from dotenv import load_dotenv
from rag_lite.src.storage.embedder import LocalEmbedder
from chromadb.config import Settings

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
        self.client = await chromadb.AsyncHttpClient(host=self.host, port=self.port,settings=Settings(allow_reset=True))
        
        collection_names = ["context", "documents", "code"]
        
        for name in collection_names:
            self.collections[name] = await self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedder 
            )
            

    def get_collection(self, name: str):
        """Returns the requested collection instance."""
        if name not in self.collections:
            raise ValueError(f"The collection '{name}' does not exist.")
        return self.collections[name]
    
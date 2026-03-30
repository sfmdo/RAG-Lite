import os
from typing import Dict, Any

# Assuming these are your actual import paths
from src.ingestion.document_loader import serveDocument, extractExtension
from processing.chunking.chunker_controller import ChunkerController
from src.storage.storage_manager import StorageManager
from src.retriever.retriever import Retriever

class RAGOrchestrator:
    def __init__(self):
        self.storage = StorageManager()
        self.retriever = Retriever(self.storage)
        self._is_initialized = False

        tokenizer = os.getenv("TOKENIZER_NAME")
        if tokenizer is None:
            raise ValueError("ERROR: La variable de entorno 'TOKENIZER_NAME' no está definida. "
                    "Asegúrate de configurar tu archivo .env con la ruta al modelo.") 
        self.chunker = ChunkerController(tokenizer_name=tokenizer)

    async def _ensure_initialized(self):
        if not self._is_initialized:
            await self.storage.initialize()
            self._is_initialized = True

    async def ingest_file(self, path: str, user_id: str) -> Dict[str, Any]:
        """
        Complete flow: Takes a file path and puts it into the Vector DB.
        """
        await self._ensure_initialized()

        extension = extractExtension(path)
        raw_text = serveDocument(path)
        
        chunks = self.chunker.process(extension=extension, content=raw_text)
        
        source_name = os.path.basename(path)
        
        await self.storage.insert(
            chunks=chunks,
            source_name=source_name,
            user_id=str(user_id),
            extension=extension
        )
        
        return {"status": "success", "chunks_inserted": len(chunks), "source": source_name}

    async def search_context(self, query: str, user_id: str) -> str:
        """
        Complete flow: Takes a query and returns a single merged context string.
        """
        await self._ensure_initialized()
        
        merged_context = await self.retriever.get_context_for_llm(
            query=query, 
            user_id=user_id
        )
        
        return merged_context
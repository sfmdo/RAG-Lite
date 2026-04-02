import os
from typing import Dict, Any, List
import datetime
from rag_lite.src.ingestion.document_loader import serveDocument, extractExtension
from rag_lite.processing.chunking.chunker_controller import ChunkerController
from rag_lite.src.storage.storage_manager import StorageManager
from rag_lite.src.retriever.retriever import Retriever

from rag_lite.src.storage.vector_store import GLOBAL_USER_ID

class RAGOrchestrator:
    def __init__(self):
        self.storage = StorageManager()
        self.retriever = Retriever(self.storage)
        self._is_initialized = False

        tokenizer = os.getenv("MODEL_NAME")
        if tokenizer is None:
            raise ValueError(
                    "ERROR: The environment variable 'MODEL_NAME' is not defined. "
                    "Make sure to configure your .env file with the model path."
                )
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

    async def ingest_global_document(self, path: str):
        """
        Ingests manuals, MCP tool documentation, or behavior rules 
        that will be available to ALL users as base knowledge.
        """
        return await self.ingest_file(path=path, user_id=GLOBAL_USER_ID)

    async def ingest_user_document(self, path: str, user_id: str):
        """
        Ingests private files belonging to a specific user (e.g., from Telegram).
        Ensures data isolation by mapping chunks to the specific user_id.
        """
        return await self.ingest_file(path=path, user_id=str(user_id))
    
    async def ingest_user_context(self, text: List[Dict[str, str]], user_id: str):
        extension="context"

        chunks = self.chunker.process(extension=extension,content=text) 
        source_name =f"Conversation, Date:{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}"

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
import uuid
from typing import List, Dict, Any

from utils.logger import get_logger
logger = get_logger(__name__)

class DocumentStore:
    def __init__(self, manager):
        """Handles operations for the 'documents' collection (PDFs, DOCX, etc.)."""
        self.collection = manager.get_collection("documents")

        self.embedder = manager.embedder

    async def add_chunks(self, chunks: List[str], source_name: str) -> bool:
        """Asynchronously adds document chunks to the database."""
        if not chunks:
            logger.debug("No chunks to insert")
            return False

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "type": "document"} for _ in chunks]

        await self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.debug(f"Inserted {len(chunks)} chunks from '{source_name}' into DocumentStore.")
        return True

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches for the most relevant document chunks based on the user query."""
        query_vector = self.embedder.embed_query(query)

        results = await self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        return self._format_results(results)

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Helper method to parse ChromaDB's nested dictionary output."""
        formatted = []
        if results and results.get('documents') and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        return formatted


class ContextStore:
    def __init__(self, manager):
        """Handles operations for the 'context' collection (Conversation History)."""
        self.collection = manager.get_collection("context")
        self.embedder = manager.embedder

    async def add_message(self, session_id: str, role: str, content: str, custom_id: str = "") -> None:
        """
        Saves a single chat message. 
        session_id -> This is where your TELEGRAM ID goes.
        custom_id -> Override for the Chroma record ID (Defaults to UUID).
        """
        message_id = custom_id if custom_id else str(uuid.uuid4())
        
        metadata = {"session_id": str(session_id), "role": role, "type": "chat_message"}

        await self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[message_id]
        )
        logger.debug(f"Added {role} message. Chroma ID: {message_id} | Telegram User: {session_id}")

    async def get_relevant_history(self, session_id: str, current_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds past messages in the current session relevant to the new query."""
        query_vector = self.embedder.embed_query(current_query)

        results = await self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"session_id": session_id}
        )
        
        formatted = []
        if results and results.get('documents') and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    "role": results['metadatas'][0][i].get("role", "unknown"),
                    "text": results['documents'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        return formatted


class CodeStore:
    def __init__(self, manager):
        """
        Handles operations for the 'code' collection.
        Currently a placeholder for future implementation.
        """
        self.collection = manager.get_collection("code")
        self.embedder = manager.embedder

    async def add_code(self, *args, **kwargs) -> None:
        """Not implemented yet."""
        logger.debug("CodeStore.add_code is not implemented yet.")
        return None

    async def search(self, *args, **kwargs) -> None:
        """Not implemented yet."""
        logger.debug("CodeStore.search is not implemented yet.")
        return None
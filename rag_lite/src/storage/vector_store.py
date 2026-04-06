import uuid
from typing import List, Dict, Any
from rag_lite.utils.logger import get_logger
logger = get_logger(__name__)

GLOBAL_USER_ID = "global_public"

class DocumentStore:
    def __init__(self, manager):
        """Handles operations for the 'documents' collection (PDFs, DOCX, etc.)."""
        self.collection = manager.get_collection("documents")

        self.embedder = manager.embedder

    async def add_chunks(self, chunks: List[str], user_id: str,source_name: str = "document") -> None:
        """Asynchronously adds document chunks to the database."""
        if not chunks:
            logger.debug("No chunks to insert")
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "type": "document", "user_id": user_id} for _ in chunks]

        await self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        logger.debug(f"Inserted {len(chunks)} chunks from '{source_name}' into DocumentStore.")

    async def search(self, query: str,user_id: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Searches for the most relevant document chunks based on the user query."""
        query_vector = self.embedder.embed_query(query)

        results = await self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            where={
            "$or": [
                {"user_id": str(user_id)},       # Private context
                {"user_id": GLOBAL_USER_ID}      # Global/System context
                ]
            }
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

    async def delete_by_source(self, user_id: str, source_name: str) -> None:
        """Deletes chunks from a specific source for a specific user."""
        await self.collection.delete(
            where={"$and": [
                {"user_id": user_id},
                {"source": source_name}
            ]}
        )
        logger.debug(f"Deleted documents from source '{source_name}' for user {user_id}")

    async def delete_all_user_docs(self, user_id: str) -> None:
        """Deletes ALL documents belonging to a specific user."""
        await self.collection.delete(where={"user_id": user_id})
        logger.debug(f"Deleted all documents for user {user_id}")


class ContextStore:
    def __init__(self, manager):
        """Handles operations for the 'context' collection (Conversation History)."""
        self.collection = manager.get_collection("context")
        self.embedder = manager.embedder

    async def add_messages(self, chunks: List[str], user_id: str, source_name: str = "conversation") -> None:
        """
        Saves a single chat message. 
        user_id -> the id of the converstaion or user than belongs the information.
        custom_id -> Override for the Chroma record ID (Defaults to UUID).
        """
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        metadata = [{"user_id": str(user_id), "type": "chat_message", "source": source_name} for _ in chunks]

        await self.collection.add(
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
        logger.debug(f"Added {len(chunks)} messages. User_Id: {user_id}")

    async def get_relevant_history(self, query: str, user_id: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Finds past messages in the current session relevant to the new query."""
        query_vector = self.embedder.embed_query(query)

        results = await self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            where={"user_id": user_id}
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

    async def delete_history(self, user_id: str) -> None:
        """Clears all conversation history for a specific user."""
        await self.collection.delete(
            where={"user_id": str(user_id)}
        )
        logger.debug(f"Deleted chat history for User_Id: {user_id}")

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

    async def delete_code(self, user_id: str) -> None:
        """Placeholder for deleting code snippets."""
        logger.debug("CodeStore.delete_code is not implemented yet.")
        return None
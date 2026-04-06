from rag_lite.src.storage.vector_store import DocumentStore , ContextStore, CodeStore
from rag_lite.src.storage.chroma_manager import AsyncChromaManager
from typing import List, Dict, Any
from rag_lite.utils.logger import get_logger

logger = get_logger(__name__)

class StorageManager:
    _EXTENSION_TO_STORAGE_ACTION: Dict[str, str] = {
        "md": "document", "py": "code", "js": "code", "cpp": "code",
        "pdf": "document", "docx": "document", "odt": "document",
        "txt": "document", "context": "context",
        "document": "document"
    }

    def __init__(self):
        self.manager = AsyncChromaManager()
        self.docstore = None
        self.contextstore = None
        self.codestore = None
        
        self.storage_actions = {}
        self.search_actions = {}

    async def initialize(self):
        """
        This is the magic bridge. It waits for the DB to connect,
        THEN it creates the stores safely.
        """
        await self.manager.initialize() 
        
        self.docstore = DocumentStore(self.manager)
        self.contextstore = ContextStore(self.manager)
        self.codestore = CodeStore(self.manager)
        
        self.storage_actions = {
            "document": self.docstore.add_chunks,
            "context": self.contextstore.add_messages,
            "code": self.codestore.add_code
        }
        self.search_actions = {
            "document": self.docstore.search,
            "context": self.contextstore.get_relevant_history,
            "code": self.codestore.search
        }

        self.delete_actions = {
            "document": self.docstore.delete_by_source,
            "context": self.contextstore.delete_history,
            "code": self.codestore.delete_code
        }

        logger.debug("StorageManager is fully initialized and ready!")

    async def insert(self, chunks: List[str], user_id: str,source_name: str, extension: str = "txt"):

        storage_type = self._EXTENSION_TO_STORAGE_ACTION.get(extension.lower(), "document")
        
        action = self.storage_actions.get(storage_type)
        
        if not action:
            logger.error(f"No action defined for storage type: {storage_type}")
            raise ValueError(f"No action defined for storage type: {storage_type}")

        return await action(chunks=chunks,source_name=source_name,user_id=user_id)

    async def retrieve(self, query: str, user_id: str,top_k: int = 3,storage_type: str = "document") -> List[Dict[str, Any]]:
        """
        Main search entry point. 
        Routes the query to the specific store based on storage_type.
        """

        search_func = self.search_actions.get(storage_type.lower())

        if not search_func:
            raise ValueError(f"No search action defined for: {storage_type}")

        logger.debug(f"Searching in {storage_type} for User {user_id}: {query[:50]}...")
        
        results = await search_func(query=query,user_id=user_id,top_k=top_k)
        return results

    async def delete(self, user_id: str, source_name: str = None, storage_type: str = "document"):
        """
        Main deletion entry point. 
        Routes the delete request to the specific store.
        """
        action = self.delete_actions.get(storage_type.lower())

        if not action:
            logger.error(f"No delete action defined for storage type: {storage_type}")
            raise ValueError(f"No delete action defined for storage type: {storage_type}")

        logger.debug(f"Deleting from {storage_type} for User {user_id} (Source: {source_name})...")
        
        if storage_type == "context":
            return await action(user_id=user_id)
        else:
            return await action(user_id=user_id, source_name=source_name)
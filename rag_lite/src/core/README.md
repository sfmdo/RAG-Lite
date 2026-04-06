# Module 5: Orchestrator Engine

## Overview
The **Orchestrator** module acts as the central, user-facing entry point for the entire RAG-Lite system. It abstracts the underlying complexity of document parsing, text chunking, database connections, and context formatting into a "plug-and-play" interface. It is designed for direct integration with application layers such as FastAPI backends or Telegram bots.

---

## 1. RAG Orchestrator (`src/orchestrator/rag_orchestrator.py`)

**Purpose:**  
To manage the complete end-to-end lifecycles of Data Ingestion and Context Retrieval. It acts as a **Facade**, coordinating the `DocumentLoader`, `ChunkerController`, `StorageManager`, and `Retriever`.

### Initialization
*   **Self-Contained Setup:** The class requires no arguments upon instantiation.
*   **Internal Instantiation:** It automatically manages instances of `StorageManager` and `Retriever`.
*   **Environment Dependency:** Requires `TOKENIZER_NAME` (loaded via `.env`) to initialize the `ChunkerController`.

### Ingestion Methods
*   **`ingest_global_document(self, path: str) -> Dict[str, Any]` (Async)**
    *   **Action:** Ingests documentation intended for the entire system.
    *   **Use Case:** Loading MCP tool guides, system prompts, or general knowledge that every user should be able to query.
    *   **Logic:** Internally calls `ingest_file` using the `GLOBAL_USER_ID`.

*   **`ingest_user_document(self, path: str, user_id: str) -> Dict[str, Any]` (Async)**
    *   **Action:** Ingests private files belonging to a specific user.
    *   **Use Case:** Processing a PDF or text file uploaded by a specific Telegram user.
    *   **Logic:** Internally calls `ingest_file` using the provided `user_id` to ensure data isolation.

*   **`ingest_file(self, path: str, user_id: str) -> Dict[str, Any]` (Async / Internal)**
    *   **Action:** The core processing flow: triggers lazy initialization, extracts raw text, generates semantically optimized chunks via `ChunkerController`, and forwards them to `StorageManager` for vectorization.
  
* **`ingest_user_context(self, text: List[Dict[str, str]], user_id: str) -> Dict[str, Any]` (Async)**
    * **Action:** Ingests conversation history directly into the RAG system to serve as long-term memory.
    * **Use Case:** Saving past chat interactions (e.g., a Telegram conversation history) so the agent can recall previous context, user preferences, or earlier topics in future interactions.
    * **Logic:** Expects a list of message dictionaries representing the chat (e.g., `[{"role": "user", "content": "..."}]`). It processes these via the `ChunkerController` using `extension="context"`, assigns them a fixed source name of `"conversation"`, and forwards them to the `StorageManager` ensuring strict isolation using the provided `user_id`.
    * **Source**: Now automatically generates a dynamic `source_name` using the format: `"Conversation, Date: YYYY/MM/DD HH:MM:SS"`. This ensures temporal traceability for each long-term memory block inserted into the database.
    * 
### Retrieval Methods
*   **`search_context(self, query: str, user_id: str) -> str` (Async)**
    *   **Action:** The primary method for generating an LLM prompt context.
    *   **Hybrid Logic:** 
        1.  **Knowledge Retrieval:** Automatically performs a search across both the specific `user_id` and the `GLOBAL_USER_ID`. This ensures the LLM has access to both private files and system-wide tools.
        2.  **Memory Retrieval:** Searches the conversation history associated strictly with the `user_id`.
    *   **Security:** While knowledge retrieval is hybrid, chat history remains strictly isolated to the individual user to prevent privacy leaks.
    *   **Returns:** A cohesive, Markdown-formatted string containing relevant document facts and past chat memory.
### Deletion Methods

*   **`delete_global_document(self, path: str) -> Dict[str, Any]` (Async)**
    *   **Action:** Removes a global manual or shared system documentation.
    *   **Logic:** Identifies the file by its `basename` and cleans all associated chunks stored under the `GLOBAL_USER_ID`.
    *   **Use Case:** Replacing outdated MCP tool guides or updating global system behavior rules.

*   **`delete_user_document(self, path: str, user_id: str) -> Dict[str, Any]` (Async)**
    *   **Action:** Removes a private document uploaded by a specific user.
    *   **Logic:** Uses the filename and the `user_id` to guarantee that only that specific user's version is deleted, maintaining the integrity of other users' data.

*   **`clear_user_chat_history(self, user_id: str) -> Dict[str, Any]` (Async)**
    *   **Action:** Completely wipes the conversational memory (context) for a specific user.
    *   **Logic:** Triggers a bulk delete command in the "context" collection, filtered strictly by the provided `user_id`.
    *   **Use Case:** Essential for implementing `/reset` or `/start` commands in chat interfaces to ensure a private and clean session.

---

## 2. Internal Logic Flows

### `_ensure_initialized(self)` (Async / Internal)
Implements the **Lazy Initialization** pattern. It verifies the database connection is active before any operation. This ensures heavy resource allocation (like loading embedding models into memory) only occurs upon the first actual request.

*   **Safe Deletion Pattern:** Deletion methods now strictly enforce the **Lazy Initialization** pattern (`_ensure_initialized`). This ensures that the connection to ChromaDB is fully established and authenticated before any data cleanup or removal operation is attempted.

### Hybrid Metadata Filtering
The Orchestrator instructs the underlying Retriever to use a logical `$or` filter during the vector search:

```python
where={"$or": [{"user_id": user_id}, {"user_id": "global_public"}]}
```

This allows the system to scale efficiently, serving shared documentation (like MCP manuals) to thousands of users without duplicating data in the database, while still respecting the privacy of individual user documents.
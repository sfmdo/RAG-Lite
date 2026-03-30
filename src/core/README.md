# Module 5: Orchestrator Engine

## Overview
The **Orchestrator** module acts as the central, user-facing entry point for the entire RAG-Lite system. It completely abstracts the underlying complexity of document parsing, text chunking, database connections, and context formatting into a simple "plug-and-play" interface. It is designed to be directly imported and used by the final application layer (like a FastAPI backend or a Telegram bot).

---

## 1. RAG Orchestrator (`src/orchestrator/rag_orchestrator.py`)

**Purpose:** To manage the complete end-to-end lifecycles of both Data Ingestion and Context Retrieval. It acts as the ultimate Facade, internally coordinating the `DocumentLoader`, `ChunkerController`, `StorageManager`, and `Retriever` so the end user only has to call two simple functions.

### Initialization Variables
* **Self-Contained Setup:** The class requires *no arguments* upon instantiation. 
* **Internal Instantiation:** It automatically creates its own instances of `StorageManager` and `Retriever`.
* **Environment Dependency:** It requires the `TOKENIZER_NAME` environment variable (e.g., loaded via `.env`) to successfully initialize the internal `ChunkerController`.

### Methods
* **`_ensure_initialized(self)`** *(Async / Internal)*
    * **Action:** Implements the **Lazy Initialization** pattern. It checks if the underlying database connection is active; if not, it initializes it. This ensures the heavy database startup only happens once, right before the very first operation, saving memory and preventing connection errors.

* **`ingest_file(self, path: str, user_id: str) -> Dict[str, Any]`** *(Async)*
    * **Action:** The complete flow for adding a new document to the knowledge base.
    * **Processing:**
        1. Triggers `_ensure_initialized()`.
        2. Extracts the file extension and reads the raw text using the ingestion utilities.
        3. Passes the text to the `ChunkerController` to be split into semantically optimized chunks based on the file type.
        4. Extracts the file name and forwards the chunks to the `StorageManager` to be vectorized and saved under the specified `user_id`.
    * **Returns:** A status dictionary containing the success state, the number of chunks inserted, and the source file name.

* **`search_context(self, query: str, user_id: str) -> str`** *(Async)*
    * **Action:** The complete flow for answering a user's question.
    * **Processing:** 1. Triggers `_ensure_initialized()`.
        2. Forwards the query directly to the internal `Retriever`.
    * **Security:** Passes the `user_id` down the chain to guarantee the search is strictly isolated to the user's own documents and chat history.
    * **Returns:** A cohesive, Markdown-formatted string containing both relevant document facts and past chat memory, ready to be injected as a hidden system prompt for the LLM.
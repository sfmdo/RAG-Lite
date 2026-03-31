# Module 4: Retrieval Engine

## Overview
The **Retrieval** module acts as the bridge between the storage layer and the Large Language Model (LLM). It is responsible for gathering relevant facts and conversational history based on the user's query, and formatting them into a structured prompt context.

---

## 1. Retriever (`src/retriever/retriever.py`)

**Purpose:** To concurrently query different vector databases (documents and chat history) and assemble a cohesive, Markdown-formatted context block that gives the LLM the necessary background to answer accurately. It utilizes Dependency Injection to interact with the storage layer, meaning it assumes the database is already connected and ready to go.

### Initialization Variables
* **`storage_manager` (`StorageManager`):** An active, pre-initialized instance of the `StorageManager`. Passing this externally ensures the retriever focuses purely on logic and formatting, rather than database connections or embedder loading.

### Methods
* **`get_context_for_llm(self, query: str, user_id: str | int) -> str`** *(Async)*
    * **Action:** Executes two asynchronous similarity searches concurrently (using `asyncio.gather`) against the `"document"` and `"context"` collections to minimize latency.
    * **Processing:** * Extracts the text and source metadata from matching documents, formatting them under a `### RELEVANT KNOWLEDGE (From Files)` header.
        * Extracts previous chat messages, formatting them under a `### RELEVANT CHAT MEMORY (From Past Messages)` header.
    * **Security:** Enforces strict data isolation by passing the `user_id` directly to the storage manager's retrieval methods, guaranteeing users only retrieve their own uploaded files and chat history.
    * **Returns:** A fully formatted string ready to be injected into the LLM's prompt. If no vector matches are found, it gracefully returns: `"No specific background information found for this query."`
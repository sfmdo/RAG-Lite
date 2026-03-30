# Module 3: Storage & Vector Database

## Overview
The **Storage** module is the central memory engine of the RAG-Lite system. It is responsible for converting text into vector embeddings, managing the asynchronous connection to ChromaDB, storing document facts, and securely managing user-specific chat histories for the Telegram bot.

---

## 1. Local Embedder (`src/storage/embedder.py`)

**Purpose:**  
A custom wrapper for **FastEmbed** that handles text vectorization. It acts as a black box, automatically applying the strict prefixing rules required by the `intfloat/multilingual-e5-small` model.

### Initialization Variables
*   **`model_name` (str):** The HuggingFace model identifier. Defaults to `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`.
*   **`cache_dir` (str):** Local path to store downloaded model weights. Defaults to `"./data/models"`.

### Methods
*   **`__call__(self, input: Documents) -> Embeddings`**
    *   **Trigger:** Automatically called by ChromaDB when `collection.add()` is executed.
    *   **Action:** Takes raw text chunks, prepends `"passage: "` to each, generates vector embeddings, and returns a list of floats.
*   **`embed_query(self, input: str) -> Embeddings`**
    *   **Trigger:** Manually called during the retrieval phase.
    *   **Action:** Takes the user's raw question, prepends `"query: "`, and returns a single embedding vector.

---

## 2. Async Chroma Manager (`src/storage/vector_store.py`)

**Purpose:**  
The central database client. It establishes the asynchronous connection to ChromaDB and holds a global instance of the `LocalEmbedder` so it isn't loaded into memory multiple times.

### Initialization Variables
*(Typically reads from your .env file or takes defaults)*
*   **`host` (str):** The ChromaDB server host (e.g., `"localhost"`).
*   **`port` (int):** The ChromaDB server port (e.g., `8000`).

### Methods
*   **`initialize(self)`:** Connects to the database asynchronously and initializes the `LocalEmbedder`.
*   **`get_collection(self, name: str)`:** Fetches or creates a specific ChromaDB collection (table) by name and assigns the custom embedder to it.

---

## 3. Document Store (`src/storage/vector_store.py`)

**Purpose:**  
Manages the long-term factual knowledge base. It stores the chunked text from your ingested files (PDFs, DOCXs, TXTs).

### Initialization Variables
*   **`manager`:** Takes an active instance of the `AsyncChromaManager`. Connects specifically to the `"documents"` collection, collectiosn are (`context`,`documents` and `code`).

### Methods
*   **`add_chunks(self, chunks: List[str], source_name: str, custom_ids: List[str] = None)`**
    *   **Action:** Inserts text chunks into the database.
    *   **Metadata applied:** `{"source": source_name, "type": "document"}`
    *   **ID Logic:** Generates random UUIDs for each chunk to prevent overwriting, unless a list of `custom_ids` is explicitly provided.
*   **`search(self, query: str, top_k: int = 3) -> List[Dict]`**
    *   **Action:** Embeds the query and performs a similarity search across all ingested documents, returning the top `k` most relevant chunks formatted as a clean dictionary list.

---

## 4. Context Store (`src/storage/vector_store.py`)

**Purpose:**  
Manages conversational memory. It is specifically designed for multi-tenant environments (like a Telegram bot) by strictly isolating memories using Telegram User IDs.

### Initialization Variables
*   **`manager`:** Takes an active instance of the `AsyncChromaManager`. Connects specifically to the `"context"` collection.

### Methods
*   **`add_message(self, session_id: str, role: str, content: str, custom_id: str = None)`**
    *   **Action:** Saves a single message to the database.
    *   **Metadata applied:** `{"user_id": user_id, "role": role, "type": "chat_message"}`.
    *   **ID Logic:** Generates a random UUID as the primary key. This ensures a user can have infinite messages without overwriting their own history.
*   **`get_relevant_history(self, session_id: str, current_query: str, top_k: int = 5) -> List[Dict]`**
    *   **Action:** Performs a similarity search for past messages that are relevant to the user's current question.
    *   **Security/Filtering:** Uses a ChromaDB `where` clause (`{"user_id": user_id}`) to guarantee that the LLM only retrieves memories belonging to the specific Telegram user making the request.

---
## 5. Storage Manager (`src/storage/storage_manager.py`)

**Purpose:** Acts as the main orchestrator (Facade) for all storage and retrieval operations. It dynamically routes text chunks and queries to the correct collection (`documents`, `context`, or `code`) depending on the file extension or the required storage type.

### Initialization Variables
* **`manager`:** Instance of `AsyncChromaManager` that handles the connection to the underlying database.
* **`storage_actions` (dict):** A dictionary mapping storage types (e.g., `"document"`, `"context"`, `"code"`) to their respective insertion functions.

### Methods
* **`initialize(self)`** *(Async)*
    * **Action:** Initializes the underlying `AsyncChromaManager` and configures the routing (`storage_actions`) to know which collection to send each data type to.
* **`insert(self, chunks: List[str], source_name: str, user_id: str, extension: str)`** *(Async)*
    * **Action:** Receives a list of text chunks and, based on the `extension` (e.g., `"pdf"`, `"context"`), determines the correct collection to save them in. 
    * **Note:** Internally uses *keyword arguments* (`user_id=user_id`, `source_name=source_name`) to prevent metadata mix-ups in the database.
* **`retrieve(self, query: str, user_id: str, storage_type: str, top_k: int = 1) -> List[Dict]`** *(Async)*
    * **Action:** Searches the collection specified by `storage_type` for the chunks most similar to the `query`.
    * **Security:** Applies a strict filter by `user_id` to ensure the user only retrieves their own information or history.
## Dependencies & Environment Variables

To operate this module, the following environment variables are utilized via the `utils.logger` utility and database connection configs:

| Variable | Purpose | Default Value |
| :--- | :--- | :--- |
| `RAG_LOG_LEVEL` | Controls the verbosity of logs (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `CHROMA_HOST` | The hostname for the ChromaDB server | `localhost` |
| `CHROMA_PORT` | The port for the ChromaDB server | `8000` |
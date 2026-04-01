# Module 3: Storage & Vector Database

## Overview
The Storage module is the central memory engine of the RAG-Lite system. It is responsible for converting text into vector embeddings, managing the asynchronous connection to ChromaDB, storing document facts, and securely managing user-specific chat histories for the system.

## Key Constants
*   **`GLOBAL_USER_ID = "global_public"`**: A reserved static identifier used to store system-wide knowledge (MCP tools, manuals, behavior rules) accessible to all users.

---

## 1. Local Embedder (`src/storage/embedder.py`)

**Purpose:** A custom wrapper for `FastEmbed` that handles text vectorization entirely locally. It acts as a black box, automatically managing the download of models and ensuring high-performance embedding generation without external APIs.

> [!IMPORTANT]
> **Model Compatibility:** The model defined in the `MODEL_NAME` environment variable must be supported by the `fastembed` library. It is recommended to check the [FastEmbed supported models list](https://qdrant.github.io/fastembed/examples/Supported_Models/) before changing the default.

### Initialization Variables
*   **`model_name` (str):** The HuggingFace model identifier. It dynamically reads from the `MODEL_NAME` environment variable, defaulting to `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`.
*   **`cache_dir` (str):** Local path to store downloaded model weights and tokenizers. This is strictly managed by `MODELS_CACHE_DIR` defined in `rag_lite.config` (typically resolving to `./models_cache` at the project root), ensuring models are accessed completely offline.

### Methods
*   **`__call__(self, input: Documents) -> Embeddings`**
    *   *Trigger:* Automatically called by ChromaDB cuando se ejecuta `collection.add()`.
    *   *Action:* Recibe fragmentos de texto (chunks), genera los vectores de incrustación (embeddings) y devuelve una lista de floats.
*   **`embed_query(self, input: str) -> Embeddings`**
    *   *Trigger:* Llamado manualmente durante la fase de recuperación (retrieval).
    *   *Action:* Toma la consulta del usuario y devuelve un único vector de embedding.

---

## 2. Async Chroma Manager (`src/storage/vector_store.py`)

**Purpose:** The central database client. It establishes the asynchronous connection to the local ChromaDB HTTP server and holds a global instance of the `LocalEmbedder` so it isn't loaded into memory multiple times.

### Initialization Variables
*(Reads from your .env file)*
*   **`host` (str):** The ChromaDB server host (e.g., `"localhost"`).
*   **`port` (int):** The ChromaDB server port (e.g., `8000`).

### Methods
*   **`initialize(self)`:** Connects to the database asynchronously and initializes the `LocalEmbedder`.
*   **`get_collection(self, name: str)`:** Fetches or creates a specific ChromaDB collection (table) by name and assigns the custom embedder to it.

---

## 3. Document Store (`src/storage/vector_store.py`)

**Purpose:** Manages the long-term factual knowledge base. It stores the chunked text from your ingested files (PDFs, DOCXs, TXTs).

### Initialization Variables
*   **`manager`:** Takes an active instance of the `AsyncChromaManager`. Connects specifically to the `"documents"` collection.

### Methods
*   **`add_chunks(self, chunks: List[str], source_name: str, custom_ids: List[str] = None)`**
    *   *Action:* Inserts text chunks into the database.
    *   *Metadata applied:* `{"source": source_name, "type": "document"}`
    *   *ID Logic:* Generates random UUIDs for each chunk to prevent overwriting, unless a list of `custom_ids` is explicitly provided.
*   **`search(self, query: str, top_k: int = 3) -> List[Dict]`**
    *   *Action:* Embeds the query and performs a similarity search across all ingested documents, returning the top `k` most relevant chunks formatted as a clean dictionary list.

---

## 4. Context Store (`src/storage/vector_store.py`)

**Purpose:** Manages conversational memory. It is specifically designed for multi-tenant environments by strictly isolating memories using unique User IDs.

### Initialization Variables
*   **`manager`:** Takes an active instance of the `AsyncChromaManager`. Connects specifically to the `"context"` collection.

### Methods
*   **`add_message(self, session_id: str, role: str, content: str, custom_id: str = None)`**
    *   *Action:* Saves a single message to the database.
    *   *Metadata applied:* `{"user_id": user_id, "role": role, "type": "chat_message"}`.
    *   *ID Logic:* Generates a random UUID as the primary key. This ensures a user can have infinite messages without overwriting their own history.
*   **`get_relevant_history(self, session_id: str, current_query: str, top_k: int = 5) -> List[Dict]`**
    *   *Action:* Performs a similarity search for past messages that are relevant to the user's current question.
    *   *Security/Filtering:* Uses a ChromaDB `where` clause (`{"user_id": user_id}`) to guarantee that the LLM only retrieves memories belonging to the specific user making the request.

---

## 5. Storage Manager (`src/storage/storage_manager.py`)

**Purpose:** Acts as the main orchestrator (Facade) for all storage and retrieval operations. It dynamically routes text chunks and queries to the correct collection (`documents`, `context`, or `code`) depending on the file extension or the required storage type.

### Initialization Variables
*   **`manager`:** Instance of `AsyncChromaManager` that handles the connection to the underlying database.
*   **`storage_actions` (dict):** A dictionary mapping storage types (e.g., `"document"`, `"context"`, `"code"`) to their respective insertion functions.

### Methods
*   **`initialize(self)` (Async)**
    *   *Action:* Initializes the underlying `AsyncChromaManager` and configures the routing (`storage_actions`) to know which collection to send each data type to.
*   **`insert(self, chunks: List[str], source_name: str, user_id: str, extension: str)` (Async)**
    *   *Action:* Receives a list of text chunks and, based on the extension (e.g., `"pdf"`, `"context"`), determines the correct collection to save them in.
*   **`retrieve(self, query: str, user_id: str, storage_type: str, top_k: int = 1)` (Async)**
    *   *Action:* Searches the collection specified by `storage_type` for the chunks most similar to the query.
    *   *Security:* Applies a strict filter by `user_id` to ensure the user only retrieves their own information or history.

---

## Dependencies & Environment Variables

| Variable | Purpose | Default Value |
| :--- | :--- | :--- |
| **RAG_LOG_LEVEL** | Controls the verbosity of logs | `INFO` |
| **CHROMA_HOST** | The hostname for the local ChromaDB server | `localhost` |
| **CHROMA_PORT** | The port for the local ChromaDB server | `8000` |
| **MODEL_NAME** | HuggingFace model identifier (must be compatible with fastembed) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
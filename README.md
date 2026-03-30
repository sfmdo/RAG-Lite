# RAG-Lite

A lightweight, fully local Retrieval-Augmented Generation (RAG) pipeline. RAG-Lite is designed to ingest various document formats, chunk them efficiently, generate multilingual embeddings, and orchestrate context retrieval for Large Language Models—all without relying on external APIs for data storage or embedding generation.

## Key Features

*   **100% Local Vector Storage:** Uses ChromaDB backed by local SQLite (`data/vector_db/`). No cloud database connection is required.
*   **Local Multilingual Embeddings:** Powered by `paraphrase-multilingual-MiniLM-L12-v2` for high-quality, offline semantic search across multiple languages.
*   **Data Isolation:** Built-in metadata filtering ensures strict separation of documents and conversational history by `user_id`.
*   **Multi-Format Ingestion:** Natively parses PDF, DOCX, ODT, MD, and TXT files.
*   **Unified Orchestration:** A centralized Facade class (`RAGOrchestrator`) manages the entire pipeline, from file ingestion to context formatting.

---

## Project Structure

The architecture is highly modular. Each core component resides in the `src/` directory and contains its own dedicated documentation detailing its technical specifications.

```plaintext
RAG-Lite/
├── data/                       # Local storage (ChromaDB SQLite, cached models)
├── resources/                  # Downloaded tokenizers and local NLP resources
├── src/                        
│   ├── ingestion/              # Extractor routers for PDFs, Word, Markdown, etc.
│   ├── processing/             # Text cleaning and Token-aware chunking logic
│   ├── storage/                # ChromaDB manager, embedder setup, and vector operations
│   ├── retriever/              # Search logic combining document facts and chat history
│   └── orchestrator/           # Main entry point combining all modules (Facade)
├── test/                       # Comprehensive Pytest integration and unit tests
├── pyproject.toml              # Project dependencies and metadata
└── uv.lock                     # uv package manager lockfile
```

---

## Installation and Setup

This project utilizes `uv` for dependency management and environment isolation.

### 1. Clone the repository and install dependencies
```bash
git clone <your-repo-url>
cd RAG-Lite
uv sync
```

### 2. Download the Local Tokenizer
Execute the following command to download the necessary tokenizer files into the resources directory:
```bash
uv run --with huggingface-hub python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', local_dir='./resources/tokenizer/paraphrase-multilingual-MiniLM-L12-v2', allow_patterns=['tokenizer*', 'vocab.txt', 'config.json'])"
```

### 3. Environment Configuration
Create a `.env` file in the root directory to define the path to the downloaded tokenizer:
```env
# Database Connection
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Model & Tokenizer Configuration
# Path to the downloaded tokenizer files
TOKENIZER_NAME=./resources/tokenizer/paraphrase-multilingual-MiniLM-L12-v2
# HuggingFace model identifier for the embedder
MODEL_NAME=intfloat/multilingual-e5-small

# System Settings
# Options: DEBUG, INFO, WARNING, ERROR
RAG_LOG_LEVEL=DEBUG
```

---

## Quick Start

The `RAGOrchestrator` abstracts the underlying complexity of the pipeline. It can be integrated into any backend service (e.g., FastAPI, Telegram bots) as follows:

```python
import asyncio
from src.orchestrator.rag_orchestrator import RAGOrchestrator

async def main():
    # 1. Initialize the Orchestrator
    orchestrator = RAGOrchestrator()
    user_id = "user_123"

    # 2. Ingest a Document
    result = await orchestrator.ingest_file(
        path="./test/inputs/text.pdf", 
        user_id=user_id
    )
    print(f"Ingestion successful. {result['chunks_inserted']} chunks saved.")

    # 3. Retrieve Context for the LLM
    query = "What is the main topic of the document?"
    context = await orchestrator.search_context(query=query, user_id=user_id)
    
    print("\n--- Context for LLM ---")
    print(context)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Testing

The project includes a comprehensive suite of unit and integration tests covering the complete pipeline, from file extraction to vector database isolation.

To execute the test suite, run:
```bash
uv run pytest -v
```

---

## Module Documentation

For detailed technical specifications, architecture patterns, and internal workflows, refer to the documentation provided within each module:

*   [Ingestion Module](./src/ingestion/README.md)
*   [Processing and Chunking Module](./src/processing/README.md)
*   [Storage Module](./src/storage/README.md)
*   [Retriever Module](./src/retriever/README.md)
*   [Orchestrator Module](./src/orchestrator/README.md)
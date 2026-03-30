# Module 2: Chunking & Text Processing Module

This module is responsible for transforming raw extracted text or structured chat history into high-quality, semantically coherent fragments (chunks) optimized for the **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** embedding model.

Since model resources are excluded from this repository to keep it lightweight, you must download the tokenizer locally before running the system.

## 1. Offline Tokenizer Setup

To ensure precise token counting without internet dependency, we use a local instance of the HuggingFace tokenizer. This allows the system to measure chunks exactly as the embedding model sees them.

### Installation
Run the following command from the project root to download only the lightweight tokenizer metadata (~1.5 MB). 

```bash
uv run --with huggingface-hub python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', local_dir='./resources/tokenizer/paraphrase-multilingual-MiniLM-L12-v2', allow_patterns=['tokenizer*', 'vocab.txt', 'config.json'])"
```

---

## 2. Context Handling (`context_handler.py`)

To handle conversational memory (Telegram/MCP) without wasting tokens on JSON syntax, the `ContextHandler` performs a **Separation of Representations**:

*   **Parsing:** Converts `[{'role': 'user', 'content': '...'}]` into a clean dialogue format: `user: content \n\n assistant: content`.
*   **Optimization:** This reduces token consumption by up to 50% compared to raw JSON, allowing the model to "remember" more history within the same 200-token limit.
*   **Identity:** Chat history is assigned a virtual `.context` extension to be processed by the orchestrator.

---

## 3. Text Cleaning (`text_cleaner.py`)

Raw text is normalized to remove noise while preserving Spanish accents and semantic structure:
1.  **Unicode Normalization (NFC):** Ensures consistent character encoding.
2.  **Noise Removal:** Strips invisible control characters but **preserves Spanish accents (á, é, ñ, etc.)**.
3.  **Whitespace Compression:** Collapses multiple spaces and limits consecutive newlines to two (`\n\n`).

---

## 4. Multi-Strategy Architecture

The module uses a decoupled architecture to handle different content types with precision:

### A. Separator Provider (`separators.py`)
Centralizes the splitting rules based on content type:
*   **DOCUMENT (.pdf, .docx, .txt, .context):** Prioritizes paragraphs and sentences.
*   **MARKDOWN (.md):** Prioritizes headers (`#`, `##`, `###`) and list items.
*   **CODE (.py, .js, .cpp):** Prioritizes structural blocks (`class`, `def`, `function`).

### B. Recursive Engine (`recursive_token_chunker.py`)
A "blind" engine that receives text and a list of separators. It calculates lengths using the local paraphrase-multilingual-MiniLM-L12-v2 tokenizer and applies a **Recursive Window** approach:
*   **Chunk Size:** 200 Tokens.
*   **Chunk Overlap:** 50 Tokens.

### C. Orchestrator (`chunker_controller.py`)
The main entry point. It receives data and an extension, resolves the correct strategy, cleans the text, and commands the engine.


---

## 65. Usage Example

The `ChunkerController` automatically decides the best approach based on the input type or file extension.

```python
from processing.chunking.chunker_controller import ChunkerController

# 1. Initialize the Controller (Offline)
controller = ChunkerController(tokenizer_name="paraphrase-multilingual-MiniLM-L12-v2")

# 2. Process Chat History (List of dicts)
# Automatically uses .context identity and DOCUMENT strategy
chat_history = [{'role': 'user', 'content': 'Hello Pepe'}, ...]
chat_chunks = controller.process(chat_history, extension="context)

# 3. Process Markdown File
# Uses MARKDOWN strategy (#, ##, etc.)
md_text = "# Project Title\n## Section 1..."
md_chunks = controller.process(md_text, extension="md")

# 4. Process Python Code
# Uses CODE strategy (class, def, etc.)
code_text = "def main():\n    print('Hello')"
code_chunks = controller.process(code_text, extension="py")
```

---
*The recursive chunking logic in this project is adapted from the [fixed_token_chunker](https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/fixed_token_chunker.py) by Brandon Starxel.*

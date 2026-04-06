# Module 2: Chunking & Text Processing Module

This module is responsible for transforming raw extracted text or structured chat history into high-quality, semantically coherent fragments (chunks) optimized for the embedding model.

In this version, **manual tokenizer installation is no longer required**. The system automatically utilizes the tokenizer and model weights managed by `fastembed` within the local cache directory.

---

## 1. Automatic Tokenizer Management

To ensure precise token counting without internet dependency, the system leverages the tokenizer already downloaded by `fastembed` in the `./models_cache` folder. 

*   **Offline First:** Once the model is first initialized, all subsequent tokenization and chunking operations are performed 100% offline.
*   **Precision:** The system measures chunks exactly as the embedding model sees them, ensuring that no information is lost due to truncation at the vector database level.

---

## 2. Context Handling (`context_handler.py`)

To handle conversational memory (Telegram/MCP) without wasting tokens on JSON syntax, the `ContextHandler` performs a **Separation of Representations**:

*   **Parsing:** Converts raw message objects `[{'role': 'user', 'content': '...'}]` into a clean dialogue format: `user: content \n\n assistant: content`.
*   **Optimization:** This reduces token consumption by up to 50% compared to raw JSON, allowing the model to "remember" more history within the same token limit.
*   **Identity:** Chat history is assigned a virtual `.context` extension to be processed correctly by the orchestrator.

---

## 3. Text Cleaning (`text_cleaner.py`)

Raw text is normalized to remove noise while preserving semantic structure:
1.  **Unicode Normalization (NFC):** Ensures consistent character encoding.
2.  **Noise Removal:** Strips invisible control characters but **preserves language-specific characters (Spanish accents, ñ, etc.)**.
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
A sophisticated engine that receives text and a list of separators. It calculates lengths using the local tokenizer and applies a **Recursive Window** approach:
*   **Chunk Size:** Default 350 Tokens.
*   **Chunk Overlap:** Default 30 Tokens.

### C. Orchestrator (`chunker_controller.py`)
The main entry point. It receives data and an extension, resolves the correct strategy, cleans the text, and executes the recursive engine.

---

## 5. Usage Example

The `ChunkerController` automatically decides the best approach based on the input type or file extension.

```python
from src.processing.chunking.chunker_controller import ChunkerController

# 1. Initialize the Controller 
# It will look for the tokenizer in the local fastembed cache
controller = ChunkerController(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Process Chat History (List of dicts)
# Automatically uses .context identity and DOCUMENT strategy
chat_history = [{'role': 'user', 'content': 'Hello Pepe'}, {'role': 'assistant', 'content': 'How can I help?'}]
chat_chunks = controller.process(chat_history, extension="context")

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
```
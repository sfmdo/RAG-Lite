from rag_lite.processing.chunking.text_splitter import TextSplitter
from typing import Any, List
from pathlib import Path
from enum import Enum

class ChunkingStrategy(Enum):
    DOCUMENT = "document"
    CHAT = "chat"
    CODE = "code"

class HuggingFaceTokenRecursiveChunker(TextSplitter):
    """
    Implementation of a TextSplitter that uses a local HuggingFace tokenizer.
    Ideal for models like e5-small running in local environments.
    """
        
    def __init__(
        self,
        tokenizer_name: str = None, 
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> None:
        import os
        from pathlib import Path
        from rag_lite.config import MODELS_CACHE_DIR

        self.tokenizer_name = tokenizer_name or os.getenv(
            "MODEL_NAME", 
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `uv add transformers`."
            )


        model_basename = self.tokenizer_name.split("/")[-1].lower()
        tokenizer_path = None

        if MODELS_CACHE_DIR.exists():
            for root, dirs, files in os.walk(MODELS_CACHE_DIR):
                if "tokenizer.json" in files or "vocab.txt" in files:
                    if model_basename in root.lower():
                        tokenizer_path = Path(root)
                        break
    

        if not tokenizer_path:
            raise ValueError(
                f"The tokenizer for '{self.tokenizer_name}' has not been downloaded yet. "
                f"Please initialize 'LocalEmbedder' first so that fastembed "
                f"can download the files locally to: {MODELS_CACHE_DIR}"
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path.resolve()),
            local_files_only=True,
            use_fast=True
        )
        except Exception as e:
            raise ValueError(
                    f"The tokenizer folder was found at {tokenizer_path}, "
                    f"but an error occurred while loading it locally: {e}"
            )


        def _token_length(text: str) -> int:
            return len(self._tokenizer.encode(text, add_special_tokens=False))

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_token_length,
            **kwargs
        )

    def split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Splits text into chunks. Uses recursive logic to split by 
        paragraphs, lines, and spaces while measuring token length.
        """
        
        if not text:
            return []
        
        if self._strip_whitespace:
            text = text.strip()
        
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Core logic to split text based on priority separators."""
        final_chunks = []
        
        # Pick the best separator
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        # Split by the selected separator
        splits = text.split(separator) if separator else list(text)

        # Recursively refine splits that are too large
        good_splits = []
        for s in splits:
            if self._length_function(s) <= self._chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    recursive_result = self._recursive_split(s, new_separators)
                    good_splits.extend(recursive_result)
                else:
                    good_splits.append(s)

        # Merge small pieces into the final 200-token chunks with 50-token overlap
        return self._merge_splits(good_splits, separator)
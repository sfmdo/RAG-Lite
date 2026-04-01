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
    tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    **kwargs: Any,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `uv add transformers`."
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True
            )
        except Exception:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            tokenizer_path = project_root / "resources" / "tokenizer" / tokenizer_name
        
            if tokenizer_path.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path.resolve()),
                    local_files_only=True,
                    use_fast=True
                )
            else:
                raise ValueError(
                    f"Could not load tokenizer '{tokenizer_name}' from Hugging Face "
                    f"nor found at local path: {tokenizer_path}"
                )

        # Define the token counting function
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
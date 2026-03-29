from processing.context_handler import ContextHandler
from processing.chunking.recursive_token_chunker import HuggingFaceTokenRecursiveChunker
from processing.text_cleaner import normalize_text
from processing.chunking.separators import SeparatorProvider
from typing import Any, List

class ChunkerController:
    def __init__(self, tokenizer_name: str = "e5-small"):
        self.chunker = HuggingFaceTokenRecursiveChunker(tokenizer_name=tokenizer_name)

    def process(self, content: Any, extension: str = ".txt") -> List[str]:
        """
        Orchestrates the flow:
        1. Identify the input type.
        2. Format if it's Chat (list).
        3. Clean the text.
        4. Ask SeparatorProvider for the rules based on the extension.
        5. Tell the Engine to split using those specific rules.
        """
        # Formatting Logic
        if extension == "context":
            # If it's context chat history, we treat it as prose (.txt) after formatting
            text = ContextHandler.to_embedding_text(content)
            current_ext = ".txt"
        else:
            text = str(content)
            current_ext = extension

        # Cleaning Logic
        text = normalize_text(text)

        # Decision Logic: Get the right separators for this specific extension
        rules = SeparatorProvider.get_separators(current_ext)

        # Execution: Pass the text AND the rules to the engine
        return self.chunker.split_text(text, separators=rules)